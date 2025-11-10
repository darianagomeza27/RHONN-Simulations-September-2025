#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master's Thesis-Level Simulation: Adaptive Neural Identification
of a Differential-Drive Mobile Robot using RHONN + EKF/UKF/PF.

Features:
- Automatic parameter optimization for EKF/UKF using Differential Evolution
- Rich excitation signals for system identification  
- Comprehensive RMSE comparison between filters
- Optional parameter tuning (can be disabled for faster execution)

Usage:
- Set run_optimization = True in main() for automatic parameter tuning
- Set run_optimization = False for fast execution with default parameters

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
# 1) True Nonlinear System (Differential Drive Mobile Robot)
# ==============================================================================
def plant_dynamics(x, u, L=0.5, friction_coeff=0.1, slip_factor=0.05):
    x_pos, y_pos, theta = x
    v_l, v_r = u
    v_l_actual = v_l * (1 - slip_factor * np.random.randn())
    v_r_actual = v_r * (1 - slip_factor * np.random.randn())
    v = (v_r_actual + v_l_actual) / 2.0
    omega = (v_r_actual - v_l_actual) / L
    v_friction = v * (1 - friction_coeff * np.abs(v))
    omega_friction = omega * (1 - friction_coeff * np.abs(omega))
    x_dot = v_friction * np.cos(theta)
    y_dot = v_friction * np.sin(theta)
    theta_dot = omega_friction
    return np.array([x_dot, y_dot, theta_dot])

def plant(x_k, u_k, dt=0.01, process_noise_type='laplacian', process_noise_std=0.05,
          terrain_roughness=0.02, sensor_bias=[0.0, 0.0, 0.0]):
    x_dot = plant_dynamics(x_k, u_k)
    x_kp1 = x_k + dt * x_dot

    terrain_noise = terrain_roughness * np.array([
        np.sin(0.5 * x_kp1[0]) * np.random.randn(),
        np.cos(0.3 * x_kp1[1]) * np.random.randn(),
        0.1 * np.sin(x_kp1[2]) * np.random.randn()
    ])

    velocity_magnitude = np.linalg.norm(u_k)
    velocity_noise_factor = 1 + 0.2 * velocity_magnitude

    if process_noise_type == 'mixed':
        gaussian_noise = np.random.normal(0, process_noise_std * velocity_noise_factor, size=3)
        impulse_noise = np.zeros(3)
        if np.random.rand() < 0.02:
            impulse_noise = np.random.normal(0, process_noise_std * 5, size=3)
        laplacian_noise = np.random.laplace(0, process_noise_std * 0.3, size=3)
        total_noise = gaussian_noise + impulse_noise + laplacian_noise
    elif process_noise_type == 'laplacian':
        total_noise = np.random.laplace(0, process_noise_std * velocity_noise_factor, size=3)
    elif process_noise_type == 'uniform':
        a = np.sqrt(3) * process_noise_std * velocity_noise_factor
        total_noise = np.random.uniform(-a, a, size=3)
    else:  # gaussian
        total_noise = np.random.normal(0, process_noise_std * velocity_noise_factor, size=3)

    bias_noise = np.array(sensor_bias) * dt
    enc_res_xy = 0.001
    enc_res_th = np.deg2rad(0.1)
    quantization_noise = np.array([
        enc_res_xy * (np.random.rand() - 0.5),
        enc_res_xy * (np.random.rand() - 0.5),
        enc_res_th * (np.random.rand() - 0.5),
    ])

    x_kp1 += terrain_noise + total_noise + bias_noise + quantization_noise
    return x_kp1

# ==============================================================================
# 2) Excitation Signal Generation for Identification
# ==============================================================================
def generate_excitation_signal(t, signal_type='mixed'):
    """Generate rich excitation signals for system identification"""
    if signal_type == 'mixed':
        # Multi-frequency sinusoidal + step changes
        v_l = 1.0 * np.sin(0.5 * t) + 0.5 * np.sin(1.2 * t) + 0.3 * np.sin(2.0 * t)
        v_r = 1.0 * np.cos(0.8 * t) + 0.4 * np.cos(1.5 * t) + 0.2 * np.cos(2.5 * t)
        
        # Add some step changes for better excitation
        if int(t) % 5 == 0:  # Every 5 seconds
            v_l += 0.5 * np.sign(np.sin(0.1 * t))
            v_r += 0.3 * np.sign(np.cos(0.15 * t))
            
    elif signal_type == 'chirp':
        # Frequency sweep
        f0, f1 = 0.1, 2.0  # Start and end frequencies
        v_l = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / 30) * t)
        v_r = np.cos(2 * np.pi * (f0 + (f1 - f0) * t / 30) * t)
        
    elif signal_type == 'prbs':
        # Pseudo-random binary sequence approximation
        v_l = 1.0 * np.sign(np.sin(17.3 * t) + 0.5 * np.sin(23.7 * t))
        v_r = 1.0 * np.sign(np.cos(19.1 * t) + 0.3 * np.cos(29.3 * t))
        
    else:  # 'simple'
        v_l = 0.8 * np.sin(0.5 * t) + 0.2 * np.cos(1.0 * t)
        v_r = 0.7 * np.cos(0.6 * t) + 0.3 * np.sin(1.2 * t)
    
    # Limit wheel velocities
    v_max = 2.0
    v_l = np.clip(v_l, -v_max, v_max)
    v_r = np.clip(v_r, -v_max, v_max)
    
    return np.array([v_l, v_r])

# ==============================================================================
# 3) RHONN Structure  
# ==============================================================================
def sigmoidal(z, beta=1.0):
    z = np.clip(z, -50, 50)  # Prevent overflow
    return 1.0 / (1.0 + np.exp(-beta * z))

def construct_z_vector(x_est, u_input=None):
    s_x = sigmoidal(x_est[0])
    s_y = sigmoidal(x_est[1])
    s_theta = sigmoidal(x_est[2])
    features = [
        s_x*s_y, s_x*s_theta, s_y*s_theta,
        s_x**2, s_y**2, s_theta**2,
    ]
    if u_input is not None and len(u_input) >= 2:
        s_vl = sigmoidal(u_input[0])
        s_vr = sigmoidal(u_input[1])
        features.append(s_vl * s_vr)
    else:
        features.append(0.0)
    features.extend([x_est[0], x_est[1], 1.0])
    return np.array(features)

def RHONN_predict(x_state_for_z, w_neuron, u_input=None):
    z_i = construct_z_vector(x_state_for_z, u_input)
    if len(z_i) != len(w_neuron):
        raise ValueError(f"Feature/weight dim mismatch: {len(z_i)} vs {len(w_neuron)}")
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

    def update(self, chi_kp1, chi_k, x_hat_previous, u_input=None):
        x_state_for_z = np.copy(x_hat_previous)
        x_state_for_z[0] = chi_k[0]
        z_i = construct_z_vector(x_state_for_z, u_input)
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

    def update(self, chi_kp1, chi_k, x_hat_previous, u_input=None):
        x_state_for_z = np.copy(x_hat_previous)
        x_state_for_z[0] = chi_k[0]
        z_i = construct_z_vector(x_state_for_z, u_input)
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
        return 1.0 / (np.sum(w_norm**2) + 1e-100)

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

    def update(self, chi_kp1, chi_k, x_hat_previous, u_input=None):
        x_state_for_z = np.copy(x_hat_previous)
        x_state_for_z[0] = chi_k[0]
        z = construct_z_vector(x_state_for_z, u_input)
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
# 5) Parameter Optimization for EKF/UKF Tuning
# ==============================================================================
def optimize_filter_parameters(run_optimization=True, n_trials=30, maxiter=50):
    """
    Optimize EKF and UKF parameters using Differential Evolution
    
    Parameters:
    - run_optimization: If False, return default parameters
    - n_trials: Number of optimization trials to run
    - maxiter: Maximum iterations per trial
    
    Returns:
    - optimized_params: Dictionary with optimized EKF, UKF, PF parameters
    """
    if not run_optimization:
        print("🔧 Using default filter parameters (no optimization)")
        return {
            'EKF': [2e-4, 8e-3, 1.5, 0.4],           # [Q_init, R_init, P_init, eta]
            'UKF': [2e-4, 8e-3, 1.5, 0.6, 1e-2],     # [Q_init, R_init, P_init, eta, alpha]
            'PF': [0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.5] # [Q_std_x, Q_std_y, Q_std_theta, R_std_x, R_std_y, R_std_theta, ess_ratio]
        }

    try:
        from scipy.optimize import differential_evolution
        print("🚀 Starting filter parameter optimization with Differential Evolution...")
    except ImportError:
        print("⚠️  scipy not available. Using default parameters.")
        return {
            'EKF': [2e-4, 8e-3, 1.5, 0.4],
            'UKF': [2e-4, 8e-3, 1.5, 0.6, 1e-2],
            'PF': [0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.5]
        }

    def objective_function(params):
        """Objective function for parameter optimization"""
        # Unpack parameters
        # EKF: Q_init, R_init, P_init, eta (4 params)
        # UKF: Q_init, R_init, P_init, eta, alpha (5 params)
        ekf_params = params[:4]
        ukf_params = params[:5]  # Shares first 4 with EKF, adds alpha
        
        # Ensure positive parameters
        ekf_params = np.abs(ekf_params)
        ukf_params = np.abs(ukf_params)
        
        # Quick simulation for evaluation
        n_eval = 500  # Shorter simulation for optimization
        dt_eval = 0.02
        x_true_eval = np.zeros((n_eval, 3))
        x_true_eval[0] = [0.0, 0.0, 0.0]
        
        # Initialize filters with candidate parameters
        num_neurons = 3
        num_features = 10
        common_weights_eval = [np.random.uniform(-0.5, 0.5, num_features) for _ in range(num_neurons)]
        
        try:
            ekf_eval = EKF_RHONN_Trainer(num_neurons, num_features, common_weights_eval,
                                        Q_init=ekf_params[0], R_init=ekf_params[1], 
                                        P_init=ekf_params[2], eta=ekf_params[3])
            ukf_eval = UKF_RHONN_Trainer(num_neurons, num_features, common_weights_eval,
                                        Q_init=ukf_params[0], R_init=ukf_params[1], 
                                        P_init=ukf_params[2], eta=ukf_params[3], alpha=ukf_params[4])
            
            x_hat_ekf_eval = np.zeros_like(x_true_eval)
            x_hat_ukf_eval = np.zeros_like(x_true_eval)
            x_hat_ekf_eval[0] = x_true_eval[0]
            x_hat_ukf_eval[0] = x_true_eval[0]
            
            # Run short simulation
            for k in range(n_eval - 1):
                t = k * dt_eval
                u_eval = generate_excitation_signal(t, 'simple')
                x_true_eval[k+1] = plant(x_true_eval[k], u_eval, dt_eval, 'laplacian', 0.02, 0.005, [0,0,0])
                
                # Add measurement noise
                meas_noise = np.random.normal(0, [0.01, 0.01, np.deg2rad(0.5)], size=3)
                x_measured = x_true_eval[k+1] + meas_noise
                
                # Update filters
                ekf_eval.update(x_measured, x_hat_ekf_eval[k], x_hat_ekf_eval[k], u_eval)
                ukf_eval.update(x_measured, x_hat_ukf_eval[k], x_hat_ukf_eval[k], u_eval)
                
                # Predict next states
                for j in range(num_neurons):
                    x_hat_ekf_eval[k+1, j] = RHONN_predict(x_hat_ekf_eval[k], ekf_eval.weights[j], u_eval)
                    x_hat_ukf_eval[k+1, j] = RHONN_predict(x_hat_ukf_eval[k], ukf_eval.weights[j], u_eval)
            
            # Calculate RMSE
            rmse_ekf = np.sqrt(np.mean((x_true_eval - x_hat_ekf_eval)**2))
            rmse_ukf = np.sqrt(np.mean((x_true_eval - x_hat_ukf_eval)**2))
            
            # Combined objective (minimize average RMSE)
            objective = 0.5 * (rmse_ekf + rmse_ukf)
            
            # Penalize extreme parameters
            penalty = 0
            if ekf_params[3] > 1.0 or ukf_params[3] > 1.0:  # eta > 1
                penalty += 10
            if ukf_params[4] > 0.1:  # alpha > 0.1
                penalty += 5
                
            return objective + penalty
            
        except Exception as e:
            print(f"⚠️  Error in evaluation: {e}")
            return 1000  # Large penalty for failed parameters
    
    # Parameter bounds: [Q_init, R_init, P_init, eta_ekf, eta_ukf, alpha_ukf]
    bounds = [
        (1e-6, 1e-2),   # Q_init
        (1e-6, 1e-1),   # R_init  
        (0.1, 10.0),    # P_init
        (0.01, 1.0),    # eta_ekf
        (0.01, 1.0),    # eta_ukf (can be different from EKF)
        (1e-4, 0.1)     # alpha_ukf
    ]
    
    best_params = None
    best_rmse = np.inf
    
    print(f"🔍 Running {n_trials} optimization trials...")
    
    for trial in range(n_trials):
        try:
            # Set random seed for this trial
            trial_seed = RANDOM_SEED + trial * 1000
            np.random.seed(trial_seed)
            
            result = differential_evolution(
                objective_function, 
                bounds, 
                maxiter=maxiter, 
                popsize=15,
                atol=1e-4,
                seed=trial_seed,
                disp=False
            )
            
            if result.fun < best_rmse:
                best_rmse = result.fun
                best_params = result.x
                
            if (trial + 1) % 10 == 0:
                print(f"   Trial {trial + 1}/{n_trials} completed. Best RMSE: {best_rmse:.6f}")
                
        except Exception as e:
            print(f"   Trial {trial + 1} failed: {e}")
            continue
    
    if best_params is None:
        print("⚠️  Optimization failed. Using default parameters.")
        return {
            'EKF': [2e-4, 8e-3, 1.5, 0.4],
            'UKF': [2e-4, 8e-3, 1.5, 0.6, 1e-2],
            'PF': [0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.5]
        }
    
    # Extract optimized parameters
    ekf_opt = [best_params[0], best_params[1], best_params[2], best_params[3]]
    ukf_opt = [best_params[0], best_params[1], best_params[2], best_params[4], best_params[5]]
    pf_default = [0.1, 0.1, 0.5, 0.1, 0.1, 0.5, 0.5]  # PF parameters not optimized here
    
    print(f"✅ Optimization completed! Best RMSE: {best_rmse:.6f}")
    print(f"📊 Optimized EKF params: Q={ekf_opt[0]:.2e}, R={ekf_opt[1]:.2e}, P={ekf_opt[2]:.3f}, η={ekf_opt[3]:.3f}")
    print(f"📊 Optimized UKF params: Q={ukf_opt[0]:.2e}, R={ukf_opt[1]:.2e}, P={ukf_opt[2]:.3f}, η={ukf_opt[3]:.3f}, α={ukf_opt[4]:.2e}")
    
    return {
        'EKF': ekf_opt,
        'UKF': ukf_opt, 
        'PF': pf_default
    }

# ==============================================================================
# 6) Main Simulation
# ==============================================================================
def main():
    # Settings
    n_steps = 1500
    dt = 0.02
    t_history = np.linspace(0, (n_steps-1) * dt, n_steps)
    process_noise_std = 0.03
    terrain_roughness = 0.01
    sensor_bias = [0.0, 0.0, 0.0]
    excitation_type = 'mixed'  # 'mixed', 'chirp', 'prbs', 'simple'
    
    # Parameter optimization settings
    run_optimization = True  # Set to False to skip optimization and use defaults
    optimization_trials = 20  # Number of optimization trials (reduce for faster execution)
    optimization_maxiter = 30  # Max iterations per trial

    # Robot & RHONN
    x_true = np.zeros((n_steps, 3))
    x_true[0] = [0.0, 0.0, 0.0]
    num_neurons = 3
    num_features = 10  # matches construct_z_vector output
    common_initial_weights = [np.random.uniform(-0.5, 0.5, num_features) for _ in range(num_neurons)]

    # Optimize filter parameters
    print("🔧" + "="*60)
    optimized_params = optimize_filter_parameters(run_optimization, optimization_trials, optimization_maxiter)
    print("="*62)

    # Initialize filters with optimized parameters
    ekf_params = optimized_params['EKF']
    ukf_params = optimized_params['UKF'] 
    pf_params = optimized_params['PF']
    
    ekf_trainer = EKF_RHONN_Trainer(num_neurons, num_features, common_initial_weights,
                                    Q_init=ekf_params[0], R_init=ekf_params[1], 
                                    P_init=ekf_params[2], eta=ekf_params[3])
    ukf_trainer = UKF_RHONN_Trainer(num_neurons, num_features, common_initial_weights,
                                    Q_init=ukf_params[0], R_init=ukf_params[1], 
                                    P_init=ukf_params[2], eta=ukf_params[3], alpha=ukf_params[4])
    pf_trainer = PF_RHONN_Trainer(num_neurons, num_features, n_particles=800,
                                  initial_weights=common_initial_weights,
                                  Q_std=pf_params[:3], R_std=pf_params[3:6], 
                                  ess_threshold=int(800 * pf_params[6]))

    x_hat_ekf = np.zeros_like(x_true); x_hat_ekf[0] = x_true[0]
    x_hat_ukf = np.zeros_like(x_true); x_hat_ukf[0] = x_true[0]
    x_hat_pf  = np.zeros_like(x_true); x_hat_pf[0]  = x_true[0]

    print("\n✅ Starting RHONN identification simulation...")
    print(f"Using {excitation_type} excitation signal for system identification")
    
    for k in range(n_steps - 1):
        t = k * dt
        
        # Generate excitation signal for identification
        u_current = generate_excitation_signal(t, excitation_type)

        x_true[k+1] = plant(x_true[k], u_current, dt, 'laplacian', process_noise_std, terrain_roughness, sensor_bias)

        # Inject measurement noise (simulated sensor)
        meas_noise = np.random.normal(0, [0.02, 0.02, np.deg2rad(1)], size=3)
        x_measured = x_true[k+1] + meas_noise

        # Update filters using noisy measurement as "truth"
        ekf_trainer.update(x_measured, x_hat_ekf[k], x_hat_ekf[k], u_current)
        ukf_trainer.update(x_measured, x_hat_ukf[k], x_hat_ukf[k], u_current)
        pf_trainer.update(x_measured, x_hat_pf[k], x_hat_pf[k], u_current)

        # Predict next state using RHONN
        for est, trainer, x_hist in [(x_hat_ekf, ekf_trainer, x_hat_ekf),
                                     (x_hat_ukf, ukf_trainer, x_hat_ukf)]:
            x_state_for_z = np.copy(x_hist[k])
            x_state_for_z[0] = x_hist[k][0]
            for i in range(3):
                est[k+1, i] = RHONN_predict(x_state_for_z, trainer.weights[i], u_current)

        pf_weights = pf_trainer.get_estimate()
        x_state_for_z = np.copy(x_hat_pf[k])
        x_state_for_z[0] = x_hat_pf[k][0]
        for i in range(3):
            x_hat_pf[k+1, i] = RHONN_predict(x_state_for_z, pf_weights[i], u_current)

        if k % (n_steps // 10) == 0:
            print(f"Progress: {k/n_steps*100:.1f}%")

    # Compute RMSE
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
    print(f"🎲 RANDOM_SEED: {RANDOM_SEED}")
    print("\n📋 PARAMETER SUMMARY:")
    print(f"   EKF: Q={ekf_params[0]:.2e}, R={ekf_params[1]:.2e}, P={ekf_params[2]:.3f}, η={ekf_params[3]:.3f}")
    print(f"   UKF: Q={ukf_params[0]:.2e}, R={ukf_params[1]:.2e}, P={ukf_params[2]:.3f}, η={ukf_params[3]:.3f}, α={ukf_params[4]:.2e}")
    print(f"   PF:  Q_std={pf_params[:3]}, R_std={pf_params[3:6]}, ESS_ratio={pf_params[6]:.2f}")
    print("="*60)

    # Plotting
    if PLOTLY_AVAILABLE:
        # State plots
        states = ['x', 'y', 'theta']
        for i, name in enumerate(states):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t_history, y=x_true[:,i], name='True', line=dict(color='black')))
            fig.add_trace(go.Scatter(x=t_history, y=x_hat_ekf[:,i], name='EKF', line=dict(dash='dash', color='blue')))
            fig.add_trace(go.Scatter(x=t_history, y=x_hat_ukf[:,i], name='UKF', line=dict(dash='dashdot', color='green')))
            fig.add_trace(go.Scatter(x=t_history, y=x_hat_pf[:,i], name='PF', line=dict(dash='dot', color='red')))
            fig.update_layout(title=f'{name.upper()} State Identification', xaxis_title='Time (s)', yaxis_title=name)
            fig.show()

        # Trajectory
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_true[:,0], y=x_true[:,1], mode='lines', name='True Path', line=dict(color='black', width=3)))
        fig.add_trace(go.Scatter(x=x_hat_ekf[:,0], y=x_hat_ekf[:,1], mode='lines', name='EKF Estimate', line=dict(dash='dash', color='blue')))
        fig.add_trace(go.Scatter(x=x_hat_ukf[:,0], y=x_hat_ukf[:,1], mode='lines', name='UKF Estimate', line=dict(dash='dashdot', color='green')))
        fig.add_trace(go.Scatter(x=x_hat_pf[:,0], y=x_hat_pf[:,1], mode='lines', name='PF Estimate', line=dict(dash='dot', color='red')))
        fig.update_layout(title='2D Trajectory Identification Results', xaxis_title='X Position (m)', yaxis_title='Y Position (m)')
        fig.show()

    print("\n✅ RHONN identification simulation complete.")

if __name__ == "__main__":
    # To skip optimization for faster execution, set run_optimization = False in main()
    main()