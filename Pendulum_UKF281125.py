import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random

# ==========================================
# 1. Configuration & Constants
# ==========================================
DT = 0.05
STEPS = 1000
PROCESS_NOISE_STD = 0.05  # Laplacian Scale
SHOT_NOISE_PROB = 0.05    # Probability of outlier
SHOT_NOISE_SCALE = 5.0    # Magnitude of outlier (Large spike)
MEASUREMENT_NOISE_STD = 0.1  # Added measurement noise
NUM_PARTICLES = 500
NUM_NEURONS = 2          # theta, theta_dot
NUM_FEATURES = 8        # High Order Connections (2nd Order)
SCALE_FACTOR = 1.0       # Not used for linear output

# --- NEW: Process Noise Scaling for Gaussian Filters (Q) ---
Q_BASE_COV = 1e-6 
# -----------------------------------------------------------

# ==========================================
# 2. Math & RHONN Utils
# ==========================================
def sigmoid(x):
    # Added protection against overflow
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def construct_z_vector(x_state, u_input):
    """
    x_state: [theta, theta_dot]
    u_input: [torque]
    Standard RHONN High-Order Connections
    """
    th = x_state[0]
    th_dot = x_state[1]
    u = u_input[0]
    
    # Base activations
    s1 = sigmoid(th)
    s2 = sigmoid(th_dot)
    su = sigmoid(u)
    
    # High Order Connections (2nd Order Expansion)
    # [s1, s2, su, s1*s2, s2*su, s1^2, s2^2, 1]
    return np.array([
        s1,
        s2,
        su,
        s1 * s2,
        s2 * su,
        s1**2,
        s2**2,
        1.0
    ])

def rhonn_predict(z, weights):
    # Linear Output Layer (Standard RHONN)
    # y = w^T z
    return np.dot(weights, z)

def rhonn_predict_jacobian(z, weights):
    # Derivative of y = w^T z w.r.t w is z
    return z

# ==========================================
# 3. Plant Dynamics (True System)
# ==========================================
def plant_step(x_k, u_k, dt, noise_std):
    """
    x_k: [theta, theta_dot]
    u_k: [torque]
    """
    g = 9.81
    L = 1.0
    m = 1.0
    b = 0.1 # Damping
    
    theta = x_k[0]
    theta_dot = x_k[1]
    torque = u_k[0]
    
    # Dynamics: theta_ddot = -g/L sin(theta) - b/(mL^2) theta_dot + 1/(mL^2) u
    
    # Add noise to input (torque disturbance) - This is the primary heavy-tailed noise source
    torque_noisy = torque + np.random.laplace(0, noise_std)
    
    # Add Shot Noise (Non-Gaussian Outliers)
    if np.random.random() < SHOT_NOISE_PROB:
        # Add a large spike to the torque
        torque_noisy += np.random.choice([-1, 1]) * SHOT_NOISE_SCALE

    theta_ddot = -(g/L) * np.sin(theta) - (b/(m*L**2)) * theta_dot + (1.0/(m*L**2)) * torque_noisy
    
    # Add secondary small noise to state derivatives for observation uncertainty
    theta_dot_noisy = theta_dot + np.random.laplace(0, noise_std * 0.1)
    theta_ddot_noisy = theta_ddot + np.random.laplace(0, noise_std * 0.1)
    
    # Euler integration
    x_kp1 = np.zeros(2)
    x_kp1[0] = theta + dt * theta_dot_noisy
    x_kp1[1] = theta_dot + dt * theta_ddot_noisy
    
    return x_kp1

# ==========================================
# 4. EKF Trainer (Gaussian Assumption)
# ==========================================
class EKF_RHONN_Trainer:
    def __init__(self, num_neurons, num_features, eta=1.0, initial_weights=None):
        self.num_neurons = num_neurons
        self.eta = eta # Learning rate
        
        # Weights: [neurons, features]
        if initial_weights is not None:
            self.weights = np.array(initial_weights, dtype=float)
        else:
            self.weights = np.random.randn(num_neurons, num_features) * 0.1

        # Covariance Matrices (One per neuron for independence)
        self.P = [np.eye(num_features) * 1.0 for _ in range(num_neurons)]
        self.Q = [np.eye(num_features) * Q_BASE_COV for _ in range(num_neurons)]
        self.R = [0.01 for _ in range(num_neurons)] # Scalar measurement noise

    def update(self, chi_kp1, chi_k, u_k, x_hat_prev):
        # Use Estimated state from k (x_hat_k) for Z vector
        x_state_z = x_hat_prev
        z = construct_z_vector(x_state_z, u_k)
        
        # Target is the INCREMENT (Velocity * dt)
        target_delta = chi_kp1 - chi_k
        
        for i in range(self.num_neurons):
            # 1. Predict P
            P_pred = self.P[i] + self.Q[i]
            
            # 2. Kalman Gain
            # H = z (Jacobian of NN w.r.t weights)
            H = rhonn_predict_jacobian(z, self.weights[i])
            
            # PH = P_pred * H
            PH = P_pred @ H
            
            # M = R + H.T * P * H
            M = self.R[i] + H @ PH
            
            # K = PH / M
            K = PH / (M + 1e-12)
            
            # 3. Update Weights
            prediction = rhonn_predict(z, self.weights[i])
            error = target_delta[i] - prediction
            
            # w = w + eta * K * error
            self.weights[i] = self.weights[i] + self.eta * K * error
            
            # 4. Update P
            # P = P - K * H.T * P
            P_update = P_pred - np.outer(K, PH)
            self.P[i] = 0.5 * (P_update + P_update.T) # Symmetrize

    def get_estimate(self, chi_k, u_k):
        x_state_z = chi_k
        z = construct_z_vector(x_state_z, u_k)
        # Estimate is Previous + Predicted Delta
        deltas = np.array([rhonn_predict(z, self.weights[i]) for i in range(self.num_neurons)])
        return chi_k + deltas

# ==========================================
# 5. PF Trainer (Laplacian Matched Filter)
# ==========================================
class PF_RHONN_Trainer:
    def __init__(self, num_neurons, num_features, n_particles, initial_weights=None):
        self.n_particles = n_particles
        self.num_features = num_features
        self.num_neurons = num_neurons
        
        # Hyperparameters
        self.Q_std = 0.5  # Diffusion
        self.R_var = 0.5  # Likelihood Scaling (Laplacian scale factor)
        self.ess_threshold = n_particles / 2.0
        
        # Particles: [neurons, particles, features]
        self.particles = []
        self.weights_pf = []
        
        for i in range(num_neurons):
            base_w = initial_weights[i] if initial_weights is not None else np.random.randn(num_features)*0.1
            
            # Initial particle spread
            p_i = base_w + np.random.randn(n_particles, num_features) * 0.01
            self.particles.append(p_i)
            self.weights_pf.append(np.ones(n_particles) / n_particles)

    def update(self, chi_kp1, chi_k, u_k, x_hat_prev):
        x_state_z = x_hat_prev
        z = construct_z_vector(x_state_z, u_k)
        
        # Target is the INCREMENT
        target_delta = chi_kp1 - chi_k
        
        for i in range(len(self.particles)):
            # 1. Predict (Random Walk)
            noise = np.random.normal(0, self.Q_std, size=self.particles[i].shape)
            self.particles[i] += noise
            
            # 2. Update (Likelihood)
            # Calculate predictions for all particles
            raw_preds = np.dot(self.particles[i], z)
            preds = raw_preds # Linear output
            
            innov = target_delta[i] - preds
            
            # Use Laplacian Likelihood (exp(-|error|)) instead of Gaussian (exp(-error^2))
            log_likelihood = -np.abs(innov) / self.R_var
            
            # Update weights (in log space to prevent underflow, then exp)
            likelihood = np.exp(log_likelihood)
            self.weights_pf[i] *= likelihood
            
            # Normalize
            w_sum = np.sum(self.weights_pf[i])
            if w_sum < 1e-300:
                # If weights collapse, reset to uniform (resilience to collapse)
                self.weights_pf[i] = np.ones(self.n_particles) / self.n_particles
            else:
                self.weights_pf[i] /= w_sum
                
            # 3. Resample
            ess = 1.0 / np.sum(self.weights_pf[i]**2)
            if ess < self.ess_threshold:
                self.resample(i)

    def resample(self, idx):
        weights = self.weights_pf[idx]
        particles = self.particles[idx]
        N = self.n_particles
        
        # Systematic Resampling (more efficient than multinomial)
        positions = (np.arange(N) + np.random.random()) / N
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        
        self.particles[idx] = particles[indexes]
        self.weights_pf[idx] = np.ones(N) / N

    def get_estimate(self, chi_k, u_k):
        x_state_z = chi_k
        z = construct_z_vector(x_state_z, u_k)
        
        estimates = []
        for i in range(self.num_neurons):
            # Weighted average of particle weights (the neural network weights)
            mean_w = np.average(self.particles[i], weights=self.weights_pf[i], axis=0)
            estimates.append(rhonn_predict(z, mean_w))
            
        return chi_k + np.array(estimates)

# ==========================================
# 6. UKF Trainer (Unscented Kalman Filter)
# ==========================================
class UKF_RHONN_Trainer:
    def __init__(self, num_neurons, num_features, eta=1.0, initial_weights=None, alpha=1e-3, beta=2.0, kappa=0.0):
        self.num_neurons = num_neurons
        self.num_features = num_features
        self.eta = eta
        
        # UKF Parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = self.alpha**2 * (self.num_features + self.kappa) - self.num_features
        
        # Weights: [neurons, features]
        if initial_weights is not None:
            self.weights = np.array(initial_weights, dtype=float)
        else:
            self.weights = np.random.randn(num_neurons, num_features) * 0.1

        # Covariance Matrices
        self.P = [np.eye(num_features) * 1.0 for _ in range(num_neurons)]
        self.Q = [np.eye(num_features) * Q_BASE_COV for _ in range(num_neurons)]
        self.R = [0.01 for _ in range(num_neurons)]

        # Weights for sigma points
        self.n = self.num_features
        self.Wm = np.zeros(2 * self.n + 1)
        self.Wc = np.zeros(2 * self.n + 1)
        
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2 * self.n + 1):
            self.Wm[i] = 1.0 / (2 * (self.n + self.lam))
            self.Wc[i] = self.Wm[i]

    def generate_sigma_points(self, w, P):
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = w
        
        try:
            # Cholesky decomposition for matrix square root
            sqrt_P = np.linalg.cholesky((self.n + self.lam) * P)
        except np.linalg.LinAlgError:
            # Jitter for stability
            P_stable = P + np.eye(self.n) * 1e-6 
            sqrt_P = np.linalg.cholesky((self.n + self.lam) * P_stable)

        for i in range(self.n):
            sigma_points[i + 1] = w + sqrt_P[:, i]
            sigma_points[i + 1 + self.n] = w - sqrt_P[:, i]
            
        return sigma_points

    def update(self, chi_kp1, chi_k, u_k, x_hat_prev):
        x_state_z = x_hat_prev
        z = construct_z_vector(x_state_z, u_k)
        
        # Target is the INCREMENT
        target_delta = chi_kp1 - chi_k
        
        for i in range(self.num_neurons):
            # 1. Prediction Step (Time Update)
            w_pred = self.weights[i]
            P_pred = self.P[i] + self.Q[i]
            
            # 2. Sigma Points
            sigmas = self.generate_sigma_points(w_pred, P_pred)
            
            # 3. Measurement Prediction
            # y = rhonn_predict(z, w)
            y_sigmas = np.array([rhonn_predict(z, s) for s in sigmas])
            
            # Predicted measurement mean
            y_pred = np.dot(self.Wm, y_sigmas)
            
            # Predicted measurement covariance
            Py = self.R[i]
            for j in range(2 * self.n + 1):
                Py += self.Wc[j] * (y_sigmas[j] - y_pred)**2
                
            # Cross covariance
            Pxy = np.zeros(self.n)
            for j in range(2 * self.n + 1):
                Pxy += self.Wc[j] * (sigmas[j] - w_pred) * (y_sigmas[j] - y_pred)
                
            # 4. Update Step (Measurement Update)
            K = Pxy / (Py + 1e-12) # Kalman Gain
            
            innovation = target_delta[i] - y_pred
            
            # w = w + eta * K * innovation
            self.weights[i] = w_pred + self.eta * K * innovation
            
            # P = P - K * Py * K.T
            self.P[i] = P_pred - np.outer(K, K) * Py
            self.P[i] = 0.5 * (self.P[i] + self.P[i].T) # Symmetrize

    def get_estimate(self, chi_k, u_k):
        x_state_z = chi_k
        z = construct_z_vector(x_state_z, u_k)
        deltas = np.array([rhonn_predict(z, self.weights[i]) for i in range(self.num_neurons)])
        return chi_k + deltas

# ==========================================
# 7. Main Simulation Loop
# ==========================================
def run_simulation():
    # np.random.seed(42)
    # random.seed(42)
    
    # History containers
    h_true = np.zeros((STEPS, NUM_NEURONS))
    h_ekf = np.zeros((STEPS, NUM_NEURONS))
    h_ukf = np.zeros((STEPS, NUM_NEURONS))
    h_pf = np.zeros((STEPS, NUM_NEURONS))
    h_measured = np.zeros((STEPS, NUM_NEURONS))
    
    # Weights History
    h_ekf_w = np.zeros((STEPS, NUM_NEURONS, NUM_FEATURES))
    h_ukf_w = np.zeros((STEPS, NUM_NEURONS, NUM_FEATURES))
    h_pf_w = np.zeros((STEPS, NUM_NEURONS, NUM_FEATURES))
    
    # Initial State [theta, theta_dot]
    x_true = np.array([np.pi/2, 0.0]) # Start at 90 degrees
    x_ekf = np.array([np.pi/2, 0.0])
    x_ukf = np.array([np.pi/2, 0.0])
    x_pf = np.array([np.pi/2, 0.0])
    
    h_true[0] = x_true
    h_ekf[0] = x_ekf
    h_ukf[0] = x_ukf
    h_pf[0] = x_pf
    h_measured[0] = x_true + np.random.normal(0, MEASUREMENT_NOISE_STD, size=2)
    
    # Initial Weights (Shared)
    init_w = np.random.uniform(-0.1, 0.1, (NUM_NEURONS, NUM_FEATURES))
    
    # Initialize Trainers
    ekf = EKF_RHONN_Trainer(NUM_NEURONS, NUM_FEATURES, eta=1.0, initial_weights=init_w)
    ukf = UKF_RHONN_Trainer(NUM_NEURONS, NUM_FEATURES, eta=1.0, initial_weights=init_w)
    pf = PF_RHONN_Trainer(NUM_NEURONS, NUM_FEATURES, NUM_PARTICLES, initial_weights=init_w)
    
    print(f"Starting Pendulum Simulation: {STEPS} steps")
    print(f"Process Noise: Laplacian (Scale={PROCESS_NOISE_STD}) + Shot Noise (Prob={SHOT_NOISE_PROB}, Scale={SHOT_NOISE_SCALE})")
    print(f"Measurement Noise: Gaussian (Std={MEASUREMENT_NOISE_STD})")
    print(f"EKF/UKF Assumption: Gaussian (Q_BASE_COV={Q_BASE_COV})")
    print(f"PF Assumption: Laplacian Likelihood (Matched Filter)")
    
    for k in range(STEPS - 1):
        # 1. Controls (Torque)
        # No input to trace normal behavior (Free response)
        torque = 0.0
        u = np.array([torque])
        
        # 2. Plant Step (Gives the TRUE next state)
        x_next_true = plant_step(x_true, u, DT, PROCESS_NOISE_STD)
        
        # CRITICAL FIX: Add measurement noise to create divergence between methods
        # Each filter now sees a noisy measurement, not the true state
        measurement_noise = np.random.normal(0, MEASUREMENT_NOISE_STD, size=2)
        x_measured = x_next_true + measurement_noise
        
        # 3. EKF Update & Predict
        # CRITICAL FIX: Use own previous estimate for Z vector calculation
        ekf.update(x_measured, x_ekf, u, x_ekf)
        x_next_ekf = ekf.get_estimate(x_ekf, u)
        
        # 4. UKF Update & Predict
        ukf.update(x_measured, x_ukf, u, x_ukf)
        x_next_ukf = ukf.get_estimate(x_ukf, u)
        
        # 5. PF Update & Predict
        pf.update(x_measured, x_pf, u, x_pf)
        x_next_pf = pf.get_estimate(x_pf, u)
        
        # 6. Advance
        x_true = x_next_true
        x_ekf = x_next_ekf
        x_ukf = x_next_ukf
        x_pf = x_next_pf
        
        h_true[k+1] = x_true
        h_ekf[k+1] = x_ekf
        h_ukf[k+1] = x_ukf
        h_pf[k+1] = x_pf
        h_measured[k+1] = x_measured
        
        # Store Weights
        h_ekf_w[k+1] = ekf.weights
        h_ukf_w[k+1] = ukf.weights
        for i in range(NUM_NEURONS):
            # Store the weighted average weight vector for the PF
            h_pf_w[k+1, i] = np.average(pf.particles[i], weights=pf.weights_pf[i], axis=0)

    # ==========================================
    # 8. Visualization
    # ==========================================
    # Create subplots
    fig = make_subplots(rows=4, cols=1, 
                        subplot_titles=("Pendulum Angle (Theta)", "Pendulum Velocity (Theta_dot)", "Mean Squared Error (Full State)", "Weights Norm Evolution (Neuron 1)"),
                        vertical_spacing=0.08)

    # 1. Angle
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_true[:, 0], mode='lines', name='True Angle', line=dict(color='green', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_measured[:, 0], mode='markers', name='Measured', marker=dict(color='gray', size=2, opacity=0.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_ekf[:, 0], mode='lines', name='EKF Angle', line=dict(color='red', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_ukf[:, 0], mode='lines', name='UKF Angle', line=dict(color='orange', width=1, dash='dashdot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_pf[:, 0], mode='lines', name='PF Angle', line=dict(color='blue', width=1.5, dash='dot')), row=1, col=1)

    # 2. Velocity
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_true[:, 1], mode='lines', name='True Vel', line=dict(color='green', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_measured[:, 1], mode='markers', name='Measured', marker=dict(color='gray', size=2, opacity=0.3), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_ekf[:, 1], mode='lines', name='EKF Vel', line=dict(color='red', width=1, dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_ukf[:, 1], mode='lines', name='UKF Vel', line=dict(color='orange', width=1, dash='dashdot')), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_pf[:, 1], mode='lines', name='PF Vel', line=dict(color='blue', width=1.5, dash='dot')), row=2, col=1)

    # 3. MSE
    err_ekf = np.mean((h_true - h_ekf)**2, axis=1)
    err_ukf = np.mean((h_true - h_ukf)**2, axis=1)
    err_pf = np.mean((h_true - h_pf)**2, axis=1)
    
    # Calculate the mean of the MSE after an initial transient period (e.g., first 100 steps)
    transient_steps = 100
    mean_ekf_mse = np.mean(err_ekf[transient_steps:])
    mean_ukf_mse = np.mean(err_ukf[transient_steps:])
    mean_pf_mse = np.mean(err_pf[transient_steps:])
    
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=err_ekf, mode='lines', name=f'EKF MSE (Mean: {mean_ekf_mse:.5f})', line=dict(color='red', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=err_ukf, mode='lines', name=f'UKF MSE (Mean: {mean_ukf_mse:.5f})', line=dict(color='orange', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=err_pf, mode='lines', name=f'PF MSE (Mean: {mean_pf_mse:.5f})', line=dict(color='blue', width=1)), row=3, col=1)

    # 4. Weights (Neuron 0)
    w_norm_ekf = np.linalg.norm(h_ekf_w[:, 0, :], axis=1)
    w_norm_ukf = np.linalg.norm(h_ukf_w[:, 0, :], axis=1)
    w_norm_pf = np.linalg.norm(h_pf_w[:, 0, :], axis=1)

    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=w_norm_ekf, mode='lines', name='EKF |W|', line=dict(color='red', width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=w_norm_ukf, mode='lines', name='UKF |W|', line=dict(color='orange', width=1)), row=4, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=w_norm_pf, mode='lines', name='PF |W|', line=dict(color='blue', width=1)), row=4, col=1)

    # Update layout
    fig.update_layout(height=1200, width=1000, title_text="Pendulum System Identification: EKF vs UKF vs PF (Non-Gaussian Noise)")
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_yaxes(title_text="Angle (rad)", row=1, col=1)
    fig.update_yaxes(title_text="Vel (rad/s)", row=2, col=1)
    fig.update_yaxes(title_text="MSE", type="log", row=3, col=1)
    fig.update_yaxes(title_text="|W|", row=4, col=1)

    fig.show()
    
    print("\n=== Final Statistics ===")
    print(f"EKF Mean MSE: {mean_ekf_mse:.6f}")
    print(f"UKF Mean MSE: {mean_ukf_mse:.6f}")
    print(f"PF Mean MSE: {mean_pf_mse:.6f}")

if __name__ == "__main__":
    run_simulation()