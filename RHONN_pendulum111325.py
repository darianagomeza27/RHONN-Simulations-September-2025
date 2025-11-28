#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified RHONN Identification for Pendulum Dynamics with PSO Optimization

Features:
- Simple pendulum nonlinear dynamics
- Simplified RHONN neural network  
- EKF/UKF/PF identification comparison
- Particle Swarm Optimization (PSO) for automatic parameter tuning
- No control needed - free oscillation with damping

Usage:
- Set run_pso_optimization = True for automatic parameter optimization
- Set run_pso_optimization = False for fast execution with defaults
- Adjust pso_particles and pso_iterations for optimization trade-offs

Author: [Your Name]
Date: 2025
"""

import numpy as np
import time

# Optional: Only import plotly if available (for flexibility)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available. Skipping plots.")

# ==============================================================================
# 0) Seeding for Reproducibility (with variation)
# ==============================================================================
RANDOM_SEED = int((time.time() * 1000000) % 100000)
np.random.seed(RANDOM_SEED)
print("🎲" + "="*60)
print(f"🎯 SEMILLA ACTUAL: {RANDOM_SEED}")
print("   ⚡ ANOTA ESTA SEMILLA SI OBTIENES BUENOS RESULTADOS")
print("   🔄 Para reproducir: cambia línea 19 a RANDOM_SEED = {0}".format(RANDOM_SEED))
print("="*62)

# ==============================================================================
# 1) Simple Pendulum Dynamics (No Control)
# ==============================================================================
def pendulum_dynamics(x, g=9.81, L=1.0, b=0.1):
    """
    Simple pendulum dynamics: d²θ/dt² = -(g/L)sin(θ) - b*dθ/dt
    State: x = [θ, θ_dot] (angle, angular velocity)
    """
    theta, theta_dot = x
    
    # Nonlinear pendulum equation
    theta_ddot = -(g/L) * np.sin(theta) - b * theta_dot
    
    return np.array([theta_dot, theta_ddot])

def plant(x_k, dt=0.01, process_noise_std=0.01):
    """
    Integrate pendulum dynamics with noise
    """
    # Runge-Kutta 4th order integration
    k1 = pendulum_dynamics(x_k)
    k2 = pendulum_dynamics(x_k + 0.5 * dt * k1)
    k3 = pendulum_dynamics(x_k + 0.5 * dt * k2)
    k4 = pendulum_dynamics(x_k + dt * k3)
    
    x_kp1 = x_k + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Add process noise
    if process_noise_std > 0:
        noise = np.random.normal(0, process_noise_std, size=2)
        x_kp1 += noise
    
    # Wrap angle to [-π, π]
    x_kp1[0] = np.arctan2(np.sin(x_kp1[0]), np.cos(x_kp1[0]))
    
    return x_kp1

# ==============================================================================
# 2) Simplified RHONN Structure for Pendulum
# ==============================================================================
def sigmoidal(z, beta=1.0):
    """Sigmoid activation function"""
    z = np.clip(z, -10, 10)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-beta * z))

def construct_z_vector(x_est):
    """
    Simplified feature vector for pendulum [θ, θ_dot]
    Features: [sin(θ), cos(θ), θ_dot, θ_dot², 1]
    """
    theta, theta_dot = x_est
    
    features = [
        np.sin(theta),        # Nonlinear angle term
        np.cos(theta),        # Nonlinear angle term  
        theta_dot,            # Angular velocity
        theta_dot**2,         # Quadratic velocity term
        1.0                   # Bias term
    ]
    
    return np.array(features)

def RHONN_predict(x_state, w_neuron):
    """Predict next state using RHONN"""
    z_i = construct_z_vector(x_state)
    return np.dot(w_neuron, z_i)

# ==============================================================================
# 4) Filters: EKF, UKF, PF
# ==============================================================================
class EKF_RHONN_Trainer:
    def __init__(self, num_neurons, num_weights_per_neuron, initial_weights=None,
                 Q_init=1e-4, R_init=1e-2, P_init=1.0, eta=1.0):
        self.num_neurons = num_neurons
        self.num_weights_per_neuron = num_weights_per_neuron
        self.eta = eta
        self.weights = []
        self.P = []
        self.Q = []
        self.R = []
        for i in range(num_neurons):
            w_i = np.copy(initial_weights[i]) if initial_weights is not None else np.random.randn(num_weights_per_neuron) * 0.1
            self.weights.append(w_i)
            self.P.append(np.eye(num_weights_per_neuron) * P_init)
            self.Q.append(np.eye(num_weights_per_neuron) * Q_init)
            self.R.append(np.array([R_init]))

    def update(self, chi_kp1, chi_k, x_hat_previous):
        x_state_for_z = np.copy(x_hat_previous)
        x_state_for_z[0] = chi_k[0]
        z_i = construct_z_vector(x_state_for_z)
        H_i = z_i.reshape(-1, 1)
        for i in range(self.num_neurons):
            P_pred = self.P[i] + self.Q[i] + np.eye(self.num_weights_per_neuron) * 1e-8
            M_i = self.R[i][0] + (H_i.T @ P_pred @ H_i)[0, 0]
            M_i = max(M_i, 1e-10)
            x_hat_pred_i = self.weights[i] @ z_i
            e_i = np.clip(chi_kp1[i] - x_hat_pred_i, -10.0, 10.0)
            K_i = (P_pred @ H_i).flatten() / M_i
            adaptive_eta = self.eta * (1.0 / (1.0 + np.abs(e_i) * 0.1))
            self.weights[i] += adaptive_eta * K_i * e_i
            I_KH = np.eye(self.num_weights_per_neuron) - np.outer(K_i, H_i.ravel())
            self.P[i] = I_KH @ P_pred @ I_KH.T + np.outer(K_i, K_i) * self.R[i][0]
            self.P[i] = 0.5 * (self.P[i] + self.P[i].T)
            if np.min(np.linalg.eigvals(self.P[i])) <= 0:
                self.P[i] += np.eye(self.num_weights_per_neuron) * 1e-6

class UKF_RHONN_Trainer:
    def __init__(self, num_neurons, num_weights_per_neuron, initial_weights=None,
                 Q_init=1e-4, R_init=1e-2, P_init=1.0, eta=1.0,
                 alpha=1e-3, beta=2.0, kappa=None):
        self.num_neurons = num_neurons
        self.num_weights_per_neuron = num_weights_per_neuron
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa if kappa is not None else 3 - num_weights_per_neuron
        self.n = num_weights_per_neuron
        self.lambda_ = alpha**2 * (self.n + self.kappa) - self.n
        self.Wm = np.zeros(2 * self.n + 1)
        self.Wc = np.zeros(2 * self.n + 1)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - alpha**2 + beta)
        for i in range(1, 2 * self.n + 1):
            weight = 1.0 / (2 * (self.n + self.lambda_))
            self.Wm[i] = weight
            self.Wc[i] = weight
        self.weights = []
        self.P = []
        self.Q = []
        self.R = []
        for i in range(num_neurons):
            w_i = np.copy(initial_weights[i]) if initial_weights is not None else np.random.randn(num_weights_per_neuron) * 0.1
            self.weights.append(w_i)
            self.P.append(np.eye(num_weights_per_neuron) * P_init)
            self.Q.append(np.eye(num_weights_per_neuron) * Q_init)
            self.R.append(np.array([R_init]))

    def _generate_sigma_points(self, mean, covariance):
        n = len(mean)
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mean
        try:
            sqrt = np.linalg.cholesky((n + self.lambda_) * covariance)
        except np.linalg.LinAlgError:
            U, s, Vh = np.linalg.svd(covariance)
            sqrt = U @ np.diag(np.sqrt(np.maximum(s, 0))) @ Vh
            sqrt *= np.sqrt(n + self.lambda_)
        for i in range(n):
            sigma_points[i + 1] = mean + sqrt[i]
            sigma_points[i + 1 + n] = mean - sqrt[i]
        return sigma_points

    def _measurement_function(self, weight_sigma_point, z_vector):
        return np.dot(weight_sigma_point, z_vector)

    def update(self, chi_kp1, chi_k, x_hat_previous):
        x_state_for_z = np.copy(x_hat_previous)
        x_state_for_z[0] = chi_k[0]
        z_i = construct_z_vector(x_state_for_z)
        for i in range(self.num_neurons):
            sigma_points = self._generate_sigma_points(self.weights[i], self.P[i])
            predicted_mean = np.sum(self.Wm[:, None] * sigma_points, axis=0)
            predicted_cov = self.Q[i].copy()
            for j in range(2 * self.n + 1):
                diff = sigma_points[j] - predicted_mean
                predicted_cov += self.Wc[j] * np.outer(diff, diff)
            measurement_sigma_points = np.array([
                self._measurement_function(sigma_points[j], z_i) for j in range(2 * self.n + 1)
            ])
            predicted_measurement = np.sum(self.Wm * measurement_sigma_points)
            innovation_cov = self.R[i][0] + np.sum(self.Wc * (measurement_sigma_points - predicted_measurement)**2)
            innovation_cov = max(innovation_cov, 1e-12)
            cross_cov = np.sum(self.Wc * (sigma_points - predicted_mean).T * (measurement_sigma_points - predicted_measurement), axis=1)
            K = cross_cov / innovation_cov
            innovation = chi_kp1[i] - predicted_measurement
            self.weights[i] = predicted_mean + self.eta * K * innovation
            self.P[i] = predicted_cov - np.outer(K, K) * innovation_cov
            self.P[i] = 0.5 * (self.P[i] + self.P[i].T)
            if np.min(np.linalg.eigvals(self.P[i])) <= 0:
                self.P[i] += np.eye(self.n) * 1e-6

class PF_RHONN_Trainer:
    def __init__(self, num_neurons, num_weights_per_neuron, n_particles=500,
                 initial_weights=None, Q_std=None, R_std=None, ess_threshold=None):
        self.num_neurons = num_neurons
        self.num_weights_per_neuron = num_weights_per_neuron
        self.n_particles = n_particles
        self.Q_std = [Q_std] * num_neurons if isinstance(Q_std, (int, float)) else (Q_std or [0.2, 0.1, 0.8])
        self.R_std = [R_std] * num_neurons if isinstance(R_std, (int, float)) else (R_std or [0.2, 0.1, 0.8])
        self.R_var = [r**2 for r in self.R_std]
        self.ess_threshold = ess_threshold if ess_threshold is not None else n_particles * 0.5
        self.weights = [np.copy(w) for w in initial_weights] if initial_weights else [np.random.randn(num_weights_per_neuron)*0.01 for _ in range(num_neurons)]
        self.particles = []
        self.weights_pf = []
        for i in range(num_neurons):
            base = self.weights[i]
            init_std = max(0.01, min(0.1, np.std(base) * 0.5)) if np.std(base) > 0 else 0.05
            particles_i = base[None, :] + np.random.randn(n_particles, num_weights_per_neuron) * init_std
            self.particles.append(particles_i)
            self.weights_pf.append(np.ones(n_particles) / n_particles)

    def _ess(self, w):
        w_norm = w / (np.sum(w) + 1e-100)
        return 1.0 / (np.sum(w_norm**2))

    def _resample_systematic(self, neuron_index):
        w = self.weights_pf[neuron_index]
        p = self.particles[neuron_index]
        w_norm = w / (np.sum(w) + 1e-100)
        N = len(w_norm)
        cdf = np.cumsum(w_norm)
        u = np.random.rand() / N
        positions = u + np.arange(N) / N
        indexes = np.searchsorted(cdf, positions)
        indexes = np.clip(indexes, 0, N - 1)
        self.particles[neuron_index] = p[indexes]
        self.weights_pf[neuron_index] = np.ones(N) / N

    def update(self, chi_kp1, chi_k, x_hat_previous):
        x_state_for_z = np.copy(x_hat_previous)
        x_state_for_z[0] = chi_k[0]
        z = construct_z_vector(x_state_for_z)
        for i in range(self.num_neurons):
            self.particles[i] += np.random.randn(self.n_particles, self.num_weights_per_neuron) * self.Q_std[i]
        for i in range(self.num_neurons):
            x_pred_particles = self.particles[i] @ z
            innov = chi_kp1[i] - x_pred_particles
            var_robust = max(self.R_var[i], 1e-6)
            ll = -0.5 * (innov**2) / var_robust - 0.5 * np.log(2 * np.pi * var_robust)
            ll_normalized = ll - np.max(ll)
            like = np.exp(np.clip(ll_normalized, -20, 0)) + 1e-15
            self.weights_pf[i] *= like
            w_sum = np.sum(self.weights_pf[i])
            if w_sum < 1e-15:
                self.weights_pf[i] = np.ones(self.n_particles) / self.n_particles
            else:
                self.weights_pf[i] /= w_sum
            if self._ess(self.weights_pf[i]) < self.ess_threshold:
                self._resample_systematic(i)
        for i in range(self.num_neurons):
            w_norm = self.weights_pf[i] / (np.sum(self.weights_pf[i]) + 1e-15)
            self.weights[i] = np.sum(w_norm[:, None] * self.particles[i], axis=0)

    def get_estimate(self):
        return self.weights

# ==============================================================================
# 5) Particle Swarm Optimization for Filter Parameters
# ==============================================================================
class PSO_Optimizer:
    """
    Particle Swarm Optimization for EKF/UKF/PF parameter tuning
    Optimizes: [Q_init, R_init, P_init, eta_ekf, eta_ukf, alpha_ukf, Q_std_theta, Q_std_theta_dot, R_std_theta, R_std_theta_dot]
    """
    
    def __init__(self, n_particles=20, n_iterations=50, bounds=None):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.bounds = bounds if bounds is not None else [
            (1e-6, 1e-2),   # Q_init (EKF/UKF)
            (1e-6, 1e-1),   # R_init (EKF/UKF)
            (0.1, 5.0),     # P_init (EKF/UKF)
            (0.01, 1.0),    # eta_ekf
            (0.01, 1.0),    # eta_ukf
            (1e-4, 0.1),    # alpha_ukf
            (0.001, 0.2),   # Q_std_theta (PF process noise for angle)
            (0.001, 0.2),   # Q_std_theta_dot (PF process noise for angular velocity)
            (0.001, 0.2),   # R_std_theta (PF measurement noise for angle)
            (0.001, 0.2)    # R_std_theta_dot (PF measurement noise for angular velocity)
        ]
        self.n_dims = len(self.bounds)
        
        # PSO parameters
        self.w = 0.9        # Inertia weight
        self.c1 = 2.0       # Cognitive parameter
        self.c2 = 2.0       # Social parameter
        
        # Initialize particles
        self.positions = np.zeros((n_particles, self.n_dims))
        self.velocities = np.zeros((n_particles, self.n_dims))
        self.best_positions = np.zeros((n_particles, self.n_dims))
        self.best_scores = np.full(n_particles, np.inf)
        self.global_best_position = np.zeros(self.n_dims)
        self.global_best_score = np.inf
        
        # Initialize random positions within bounds
        for i in range(self.n_dims):
            low, high = self.bounds[i]
            self.positions[:, i] = np.random.uniform(low, high, n_particles)
            self.velocities[:, i] = np.random.uniform(-(high-low)*0.1, (high-low)*0.1, n_particles)
        
        self.best_positions = self.positions.copy()
    
    def _clip_to_bounds(self, positions):
        """Ensure particles stay within bounds"""
        clipped = positions.copy()
        for i in range(self.n_dims):
            low, high = self.bounds[i]
            clipped[:, i] = np.clip(clipped[:, i], low, high)
        return clipped
    
    def objective_function(self, params):
        """
        Evaluate filter performance for given parameters
        Returns RMSE (to minimize)
        """
        try:
            # Unpack parameters
            Q_init, R_init, P_init, eta_ekf, eta_ukf, alpha_ukf, Q_std_theta, Q_std_theta_dot, R_std_theta, R_std_theta_dot = params
            
            # Quick evaluation simulation
            n_eval = 150  # Shorter for fast evaluation since we're evaluating 3 filters now
            dt_eval = 0.02
            x_true_eval = np.zeros((n_eval, 2))
            x_true_eval[0] = [np.pi/4, 0.0]  # 45 degrees start
            
            # Initialize filters with test parameters
            num_neurons = 2
            num_features = 5
            common_weights_eval = [np.random.uniform(-0.5, 0.5, num_features) for _ in range(num_neurons)]
            
            ekf_eval = EKF_RHONN_Trainer(num_neurons, num_features, common_weights_eval,
                                        Q_init=Q_init, R_init=R_init, P_init=P_init, eta=eta_ekf)
            ukf_eval = UKF_RHONN_Trainer(num_neurons, num_features, common_weights_eval,
                                        Q_init=Q_init, R_init=R_init, P_init=P_init, eta=eta_ukf, alpha=alpha_ukf)
            pf_eval = PF_RHONN_Trainer(num_neurons, num_features, n_particles=100,  # Reduced for speed
                                      initial_weights=common_weights_eval,
                                      Q_std=[Q_std_theta, Q_std_theta_dot], 
                                      R_std=[R_std_theta, R_std_theta_dot], 
                                      ess_threshold=50)
            
            x_hat_ekf_eval = np.zeros_like(x_true_eval)
            x_hat_ukf_eval = np.zeros_like(x_true_eval)
            x_hat_pf_eval = np.zeros_like(x_true_eval)
            x_hat_ekf_eval[0] = x_true_eval[0]
            x_hat_ukf_eval[0] = x_true_eval[0]
            x_hat_pf_eval[0] = x_true_eval[0]
            
            # Run evaluation simulation
            for k in range(n_eval - 1):
                # Simulate true pendulum
                x_true_eval[k+1] = plant(x_true_eval[k], dt_eval, 0.005)
                
                # Add measurement noise
                meas_noise = np.random.normal(0, [0.005, 0.02], size=2)
                x_measured = x_true_eval[k+1] + meas_noise
                
                # Update filters
                ekf_eval.update(x_measured, x_hat_ekf_eval[k], x_hat_ekf_eval[k])
                ukf_eval.update(x_measured, x_hat_ukf_eval[k], x_hat_ukf_eval[k])
                pf_eval.update(x_measured, x_hat_pf_eval[k], x_hat_pf_eval[k])
                
                # Predict next states
                x_hat_ekf_eval[k+1, 0] = RHONN_predict(x_hat_ekf_eval[k], ekf_eval.weights[0])
                x_hat_ekf_eval[k+1, 1] = RHONN_predict(x_hat_ekf_eval[k], ekf_eval.weights[1])
                x_hat_ukf_eval[k+1, 0] = RHONN_predict(x_hat_ukf_eval[k], ukf_eval.weights[0])
                x_hat_ukf_eval[k+1, 1] = RHONN_predict(x_hat_ukf_eval[k], ukf_eval.weights[1])
                
                pf_estimates = pf_eval.get_estimate()
                x_hat_pf_eval[k+1, 0] = RHONN_predict(x_hat_pf_eval[k], pf_estimates[0])
                x_hat_pf_eval[k+1, 1] = RHONN_predict(x_hat_pf_eval[k], pf_estimates[1])
            
            # Calculate RMSE for all three filters
            rmse_ekf = np.sqrt(np.mean((x_true_eval - x_hat_ekf_eval)**2))
            rmse_ukf = np.sqrt(np.mean((x_true_eval - x_hat_ukf_eval)**2))
            rmse_pf = np.sqrt(np.mean((x_true_eval - x_hat_pf_eval)**2))
            
            # Combined objective (minimize average RMSE of all three filters)
            objective = (rmse_ekf + rmse_ukf + rmse_pf) / 3.0
            
            # Add penalty for extreme parameters
            penalty = 0
            if eta_ekf > 0.8 or eta_ukf > 0.8:
                penalty += 1.0
            if alpha_ukf > 0.05:
                penalty += 0.5
            # Penalty for extreme PF noise parameters
            if Q_std_theta > 0.15 or Q_std_theta_dot > 0.15 or R_std_theta > 0.15 or R_std_theta_dot > 0.15:
                penalty += 0.5
                
            return objective + penalty
            
        except Exception as e:
            print(f"⚠️ Error in PSO evaluation: {e}")
            return 10.0  # Large penalty for failed parameters
    
    def optimize(self, verbose=True):
        """Run PSO optimization"""
        if verbose:
            print("🚀 Starting PSO optimization for filter parameters...")
            print(f"   Particles: {self.n_particles}, Iterations: {self.n_iterations}")
        
        for iteration in range(self.n_iterations):
            # Evaluate all particles
            for i in range(self.n_particles):
                score = self.objective_function(self.positions[i])
                
                # Update personal best
                if score < self.best_scores[i]:
                    self.best_scores[i] = score
                    self.best_positions[i] = self.positions[i].copy()
                
                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()
            
            # Update velocities and positions
            for i in range(self.n_particles):
                # Random factors
                r1, r2 = np.random.random(self.n_dims), np.random.random(self.n_dims)
                
                # Update velocity
                cognitive = self.c1 * r1 * (self.best_positions[i] - self.positions[i])
                social = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                
                # Update position
                self.positions[i] += self.velocities[i]
            
            # Clip to bounds
            self.positions = self._clip_to_bounds(self.positions)
            
            # Progress update
            if verbose and (iteration + 1) % 10 == 0:
                print(f"   Iteration {iteration + 1}/{self.n_iterations}: Best RMSE = {self.global_best_score:.6f}")
        
        if verbose:
            print(f"✅ PSO optimization completed! Final best RMSE: {self.global_best_score:.6f}")
        
        return self.global_best_position, self.global_best_score

def optimize_filter_parameters_pso(run_optimization=True, n_particles=20, n_iterations=50):
    """
    Optimize EKF, UKF, and PF parameters using PSO
    
    Returns:
    - optimized_params: Dictionary with optimized parameters for all filters
    """
    if not run_optimization:
        print("🔧 Using default filter parameters (no PSO optimization)")
        return {
            'EKF': [1e-4, 1e-3, 1.0, 0.5],              # [Q_init, R_init, P_init, eta]
            'UKF': [1e-4, 1e-3, 1.0, 0.5, 1e-2],        # [Q_init, R_init, P_init, eta, alpha]
            'PF': [0.05, 0.05, 0.05, 0.05],              # [Q_std_theta, Q_std_theta_dot, R_std_theta, R_std_theta_dot]
            'PSO_score': None
        }
    
    print("🔧" + "="*60)
    
    # Initialize PSO
    pso = PSO_Optimizer(n_particles=n_particles, n_iterations=n_iterations)
    
    # Run optimization
    best_params, best_score = pso.optimize(verbose=True)
    
    # Extract optimized parameters
    Q_opt, R_opt, P_opt, eta_ekf, eta_ukf, alpha_ukf, Q_std_theta, Q_std_theta_dot, R_std_theta, R_std_theta_dot = best_params
    
    ekf_params = [Q_opt, R_opt, P_opt, eta_ekf]
    ukf_params = [Q_opt, R_opt, P_opt, eta_ukf, alpha_ukf]
    pf_params = [Q_std_theta, Q_std_theta_dot, R_std_theta, R_std_theta_dot]
    
    print(f"📊 Optimized EKF params: Q={Q_opt:.2e}, R={R_opt:.2e}, P={P_opt:.3f}, η={eta_ekf:.3f}")
    print(f"📊 Optimized UKF params: Q={Q_opt:.2e}, R={R_opt:.2e}, P={P_opt:.3f}, η={eta_ukf:.3f}, α={alpha_ukf:.2e}")
    print(f"📊 Optimized PF params: Q_std=[{Q_std_theta:.3f}, {Q_std_theta_dot:.3f}], R_std=[{R_std_theta:.3f}, {R_std_theta_dot:.3f}]")
    print("="*62)
    
    return {
        'EKF': ekf_params,
        'UKF': ukf_params,
        'PF': pf_params,
        'PSO_score': best_score
    }

# ==============================================================================
# 6) Main Simulation
# ==============================================================================
def main():
    # Settings
    n_steps = 1000
    dt = 0.02
    t_history = np.linspace(0, (n_steps-1) * dt, n_steps)
    process_noise_std = 0.01

    # PSO Optimization Settings (COMMENTED OUT - Using pre-optimized values)
    # run_pso_optimization = True   # Set to False for fast execution with defaults
    # pso_particles = 15           # Number of PSO particles (reduce for faster optimization)
    # pso_iterations = 40          # Number of PSO iterations

    # Pendulum & RHONN setup
    x_true = np.zeros((n_steps, 2))  # [θ, θ_dot]
    x_true[0] = [np.pi/3, 0.0]  # Start at 60 degrees with zero velocity
    
    num_neurons = 2  # Two states for pendulum
    num_features = 5  # Simplified feature vector size
    common_initial_weights = [np.random.uniform(-0.5, 0.5, num_features) for _ in range(num_neurons)]

    # Use pre-optimized parameters (from previous PSO optimization)
    # OPTIMIZED PARAMETER SUMMARY:
    #    EKF: Q=5.37e-03, R=1.26e-02, P=5.000, η=0.773
    #    UKF: Q=5.37e-03, R=1.26e-02, P=5.000, η=0.723, α=6.62e-04
    #    PF:  Q_std=[0.053, 0.067], R_std=[0.073, 0.125] (PSO optimized)
    
    # Comment out PSO optimization and use direct values
    # optimized_params = optimize_filter_parameters_pso(run_pso_optimization, pso_particles, pso_iterations)
    
    # Extract optimized parameters (using pre-computed values)
    ekf_params = [5.37e-03, 1.26e-02, 5.000, 0.773]              # [Q_init, R_init, P_init, eta]
    ukf_params = [5.37e-03, 1.26e-02, 5.000, 0.723, 6.62e-04]   # [Q_init, R_init, P_init, eta, alpha]
    pf_params = [0.1, 0.1, 0.1, 0.1]                     # [Q_std_theta, Q_std_theta_dot, R_std_theta, R_std_theta_dot]
    pso_score = None  # No PSO optimization run

    # Initialize filters with optimized parameters
    ekf_trainer = EKF_RHONN_Trainer(num_neurons, num_features, common_initial_weights,
                                    Q_init=ekf_params[0], R_init=ekf_params[1], 
                                    P_init=ekf_params[2], eta=ekf_params[3])
    ukf_trainer = UKF_RHONN_Trainer(num_neurons, num_features, common_initial_weights,
                                    Q_init=ukf_params[0], R_init=ukf_params[1], 
                                    P_init=ukf_params[2], eta=ukf_params[3], alpha=ukf_params[4])
    pf_trainer = PF_RHONN_Trainer(num_neurons, num_features, n_particles=300,
                                  initial_weights=common_initial_weights,
                                  Q_std=[pf_params[0], pf_params[1]], 
                                  R_std=[pf_params[2], pf_params[3]], ess_threshold=150)

    x_hat_ekf = np.zeros_like(x_true); x_hat_ekf[0] = x_true[0]
    x_hat_ukf = np.zeros_like(x_true); x_hat_ukf[0] = x_true[0]
    x_hat_pf  = np.zeros_like(x_true); x_hat_pf[0]  = x_true[0]

    print("\n✅ Starting pendulum RHONN identification...")
    print(f"Initial angle: {x_true[0][0]:.2f} rad ({np.degrees(x_true[0][0]):.1f}°)")
    
    for k in range(n_steps - 1):
        # Simulate true pendulum (no control input needed)
        x_true[k+1] = plant(x_true[k], dt, process_noise_std)

        # Add measurement noise (simulated sensor)
        meas_noise = np.random.normal(0, [0.01, 0.05], size=2)  # Small noise on angle, larger on velocity
        x_measured = x_true[k+1] + meas_noise

        # Update filters using noisy measurement
        ekf_trainer.update(x_measured, x_hat_ekf[k], x_hat_ekf[k])
        ukf_trainer.update(x_measured, x_hat_ukf[k], x_hat_ukf[k])
        pf_trainer.update(x_measured, x_hat_pf[k], x_hat_pf[k])

        # Predict next state using RHONN for each filter
        x_hat_ekf[k+1, 0] = RHONN_predict(x_hat_ekf[k], ekf_trainer.weights[0])
        x_hat_ekf[k+1, 1] = RHONN_predict(x_hat_ekf[k], ekf_trainer.weights[1])
        
        x_hat_ukf[k+1, 0] = RHONN_predict(x_hat_ukf[k], ukf_trainer.weights[0])
        x_hat_ukf[k+1, 1] = RHONN_predict(x_hat_ukf[k], ukf_trainer.weights[1])
        
        pf_estimates = pf_trainer.get_estimate()
        x_hat_pf[k+1, 0] = RHONN_predict(x_hat_pf[k], pf_estimates[0])
        x_hat_pf[k+1, 1] = RHONN_predict(x_hat_pf[k], pf_estimates[1])

        if k % 100 == 0:
            print(f"Step {k}/{n_steps-1}: θ={x_true[k+1][0]:.3f}, θ̇={x_true[k+1][1]:.3f}")

    # Calculate RMSE
    def rmse(a, b): return np.sqrt(np.mean((a - b)**2, axis=0))
    rmse_ekf = rmse(x_true, x_hat_ekf)
    rmse_ukf = rmse(x_true, x_hat_ukf)
    rmse_pf  = rmse(x_true, x_hat_pf)
    total_ekf = np.linalg.norm(rmse_ekf)
    total_ukf = np.linalg.norm(rmse_ukf)
    total_pf  = np.linalg.norm(rmse_pf)
    rmse_totals = {'EKF': total_ekf, 'UKF': total_ukf, 'PF': total_pf}
    best = min(rmse_totals, key=rmse_totals.get)

    print("\n" + "="*60)
    print(f"🏆 BEST FILTER: {best} (Total RMSE: {rmse_totals[best]:.6f})")
    print(f"📊 EKF RMSE: {total_ekf:.6f} | UKF RMSE: {total_ukf:.6f} | PF RMSE: {total_pf:.6f}")
    # if pso_score is not None:
    #     print(f"🔍 PSO Optimization Score: {pso_score:.6f}")
    print(f"🎲 RANDOM_SEED: {RANDOM_SEED}")
    print("\n📋 PRE-OPTIMIZED PARAMETER SUMMARY:")
    print(f"   EKF: Q={ekf_params[0]:.2e}, R={ekf_params[1]:.2e}, P={ekf_params[2]:.3f}, η={ekf_params[3]:.3f}")
    print(f"   UKF: Q={ukf_params[0]:.2e}, R={ukf_params[1]:.2e}, P={ukf_params[2]:.3f}, η={ukf_params[3]:.3f}, α={ukf_params[4]:.2e}")
    print(f"   PF:  Q_std=[{pf_params[0]:.3f}, {pf_params[1]:.3f}], R_std=[{pf_params[2]:.3f}, {pf_params[3]:.3f}] (Pre-optimized values)")
    print("="*60)

    # Plotting
    if PLOTLY_AVAILABLE:
        # State plots
        states = ['θ (angle)', 'θ̇ (angular velocity)']
        units = ['rad', 'rad/s']
        for i, (name, unit) in enumerate(zip(states, units)):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_history, y=x_true[:,i], name='True', line=dict(color='black', width=3)))
            fig.add_trace(go.Scatter(x=t_history, y=x_hat_ekf[:,i], name='EKF', line=dict(dash='dash', color='blue')))
            fig.add_trace(go.Scatter(x=t_history, y=x_hat_ukf[:,i], name='UKF', line=dict(dash='dashdot', color='green')))
            fig.add_trace(go.Scatter(x=t_history, y=x_hat_pf[:,i], name='PF', line=dict(dash='dot', color='red')))
            fig.update_layout(title=f'Pendulum {name} Identification', xaxis_title='Time (s)', yaxis_title=f'{name} ({unit})')
            fig.show()

        # Phase portrait (θ vs θ̇)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_true[:,0], y=x_true[:,1], mode='lines', name='True Trajectory', line=dict(color='black', width=3)))
        fig.add_trace(go.Scatter(x=x_hat_ekf[:,0], y=x_hat_ekf[:,1], mode='lines', name='EKF Estimate', line=dict(dash='dash', color='blue')))
        fig.add_trace(go.Scatter(x=x_hat_ukf[:,0], y=x_hat_ukf[:,1], mode='lines', name='UKF Estimate', line=dict(dash='dashdot', color='green')))
        fig.add_trace(go.Scatter(x=x_hat_pf[:,0], y=x_hat_pf[:,1], mode='lines', name='PF Estimate', line=dict(dash='dot', color='red')))
        fig.update_layout(title='Pendulum Phase Portrait', xaxis_title='θ (rad)', yaxis_title='θ̇ (rad/s)')
        fig.show()

    print("\n✅ Pendulum RHONN identification simulation complete.")

if __name__ == "__main__":
    main()