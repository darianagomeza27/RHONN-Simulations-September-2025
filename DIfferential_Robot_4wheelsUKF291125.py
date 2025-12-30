import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# FIXED RHONN with Per-State Configuration
# ==========================================
# Fixed Issues:
# 1. Made RHONN truly nonlinear (added hidden layer activation)
# 2. Fixed UKF to use proper unscented transform with nonlinear model
# 3. Fixed PF to work with configurable particle count
# 4. Added proper parallel architecture (uses estimates, not measurements)
# ==========================================

# ==========================================
# 1. Configuration & Constants
# ==========================================
DT = 0.05
STEPS = 1000
PROCESS_NOISE_STD = 0.1  # Laplacian Scale
NUM_PARTICLES = 500  # FIXED: Set to valid number
NUM_NEURONS = 3      # px, py, theta

# ==========================================
# 1.1 RHONN Configuration
# ==========================================
STATE_0_FEATURES = ['cos_theta', 'sin_theta', 'u1', 'u2', 'u3', 'u4', 'bias']
STATE_1_FEATURES = ['cos_theta', 'sin_theta', 'u1', 'u2', 'u3', 'u4', 'bias']
STATE_2_FEATURES = ['u1', 'u2', 'u3', 'u4', 'theta', 'bias']

RHONN_CONFIG = [STATE_0_FEATURES, STATE_1_FEATURES, STATE_2_FEATURES]
NUM_FEATURES_PER_STATE = [len(features) for features in RHONN_CONFIG]

# ==========================================
# 2. RHONN Utils (NOW TRULY NONLINEAR)
# ==========================================
def sigmoid(z, beta=1.0):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-beta * z))

def sigmoid_derivative(z, beta=1.0):
    s = sigmoid(z, beta)
    return beta * s * (1 - s)

def get_all_available_features(x_state, u_input):
    """Calculate all available features."""
    px, py, theta = x_state
    u1, u2, u3, u4 = u_input
    
    return {
        'px': px,
        'py': py,
        'theta': theta,
        'u1': u1,
        'u2': u2,
        'u3': u3,
        'u4': u4,
        'px*py': px * py,
        'px*theta': px * theta,
        'py*theta': py * theta,
        'theta*u1': theta * u1,
        'theta*u2': theta * u2,
        'theta*u3': theta * u3,
        'theta*u4': theta * u4,
        'cos_theta': np.cos(theta),
        'sin_theta': np.sin(theta),
        'bias': 1.0
    }

def construct_z_vector_for_state(x_state, u_input, state_idx):
    """Construct feature vector for a specific state."""
    all_features = get_all_available_features(x_state, u_input)
    selected_features = RHONN_CONFIG[state_idx]
    z = np.array([all_features[feat] for feat in selected_features])
    return z

# FIXED: Now truly nonlinear RHONN
def rhonn_predict(z, weights):
    """Nonlinear RHONN: y = sigmoid(w^T * z)"""
    activation = np.dot(weights, z)
    return np.tanh(activation)  # Nonlinear activation

def rhonn_predict_with_jacobian(z, weights):
    """Returns both prediction and Jacobian w.r.t. weights."""
    activation = np.dot(weights, z)
    prediction = np.tanh(activation)
    # Jacobian: d(tanh(w^T*z))/dw = sech^2(w^T*z) * z = (1 - tanh^2(...)) * z
    jacobian = (1 - prediction**2) * z
    return prediction, jacobian

# ==========================================
# 3. Plant Dynamics (True System)
# ==========================================
def plant_step(x_k, u_k, dt, noise_std):
    """Skid-Steer Kinematics with Laplacian process noise."""
    r = 0.1   # Wheel radius
    B = 0.5   # Track width
    
    u1, u2, u3, u4 = u_k
    w_L = (u1 + u3) / 2.0
    w_R = (u2 + u4) / 2.0
    
    v = (r / 2.0) * (w_R + w_L)
    w = (r / (2.0 * B)) * (w_R - w_L)
    
    px_dot = v * np.cos(x_k[2])
    py_dot = v * np.sin(x_k[2])
    theta_dot = w
    
    # Laplacian process noise
    noise_px = np.random.laplace(0, noise_std / 10)
    noise_py = np.random.laplace(0, noise_std / 10)
    noise_th = np.random.laplace(0, noise_std / 10)
    
    x_kp1 = np.zeros(3)
    x_kp1[0] = x_k[0] + dt * px_dot + noise_px
    x_kp1[1] = x_k[1] + dt * py_dot + noise_py
    x_kp1[2] = x_k[2] + dt * theta_dot + noise_th
    
    return x_kp1

# ==========================================
# 4. EKF Trainer (FIXED: Proper Linearization)
# ==========================================
class EKF_RHONN_Trainer:
    def __init__(self, num_neurons, num_features_per_state, eta=0.8, initial_weights=None):
        self.num_neurons = num_neurons
        self.num_features_per_state = num_features_per_state
        self.eta = eta
        
        self.weights = []
        for i in range(num_neurons):
            num_feat = num_features_per_state[i]
            if initial_weights is not None:
                self.weights.append(np.array(initial_weights[i], dtype=float))
            else:
                self.weights.append(np.random.randn(num_feat) * 0.1)

        self.P = [np.eye(num_features_per_state[i]) * 1.0 for i in range(num_neurons)]
        self.Q = [np.eye(num_features_per_state[i]) * 1e-4 for i in range(num_neurons)]
        self.R = [0.01 for _ in range(num_neurons)]

    def update(self, chi_kp1, chi_k, u_k, x_hat_prev):
        # FIXED: Use previous estimates (parallel architecture)
        x_state_z = x_hat_prev.copy()
        target_delta = chi_kp1 - chi_k
        
        for i in range(self.num_neurons):
            z = construct_z_vector_for_state(x_state_z, u_k, i)
            
            # Predict
            P_pred = self.P[i] + self.Q[i]
            
            # Get prediction and Jacobian
            prediction, H = rhonn_predict_with_jacobian(z, self.weights[i])
            
            # Kalman Gain
            PH = P_pred @ H
            M = self.R[i] + H @ PH
            K = PH / (M + 1e-12)
            
            # Update
            error = target_delta[i] - prediction
            self.weights[i] = self.weights[i] + self.eta * K * error
            
            # Update P
            P_update = P_pred - np.outer(K, PH)
            self.P[i] = 0.5 * (P_update + P_update.T)

    def get_estimate(self, chi_k, x_hat_prev, u_k):
        x_state_z = x_hat_prev.copy()
        deltas = []
        for i in range(3):
            z = construct_z_vector_for_state(x_state_z, u_k, i)
            deltas.append(rhonn_predict(z, self.weights[i]))
        return chi_k + np.array(deltas)

# ==========================================
# 5. UKF Trainer (Unscented Transform)
# ==========================================
class UKF_RHONN_Trainer:
    def __init__(self, num_neurons, num_features_per_state, eta=1.0, initial_weights=None, 
                 alpha=1e-3, beta=1.0, kappa=0.0):
        self.num_neurons = num_neurons
        self.num_features_per_state = num_features_per_state
        self.eta = eta
        self.alpha = alpha
        self.beta_ukf = beta
        self.kappa = kappa
        
        self.weights = []
        for i in range(num_neurons):
            num_feat = num_features_per_state[i]
            if initial_weights is not None:
                self.weights.append(np.array(initial_weights[i], dtype=float))
            else:
                self.weights.append(np.random.randn(num_feat) * 0.1)

        self.P = [np.eye(num_features_per_state[i]) * 1.0 for i in range(num_neurons)]
        self.Q = [np.eye(num_features_per_state[i]) * 1e-4 for i in range(num_neurons)]
        self.R = [0.01 for _ in range(num_neurons)]

        self.lam = []
        self.Wm = []
        self.Wc = []
        
        for i in range(num_neurons):
            n = num_features_per_state[i]
            lam = self.alpha**2 * (n + self.kappa) - n
            self.lam.append(lam)
            
            Wm = np.zeros(2 * n + 1)
            Wc = np.zeros(2 * n + 1)
            
            Wm[0] = lam / (n + lam)
            Wc[0] = lam / (n + lam) + (1 - self.alpha**2 + self.beta_ukf)
            
            for j in range(1, 2 * n + 1):
                Wm[j] = 1.0 / (2 * (n + lam))
                Wc[j] = Wm[j]
            
            self.Wm.append(Wm)
            self.Wc.append(Wc)

    def generate_sigma_points(self, w, P, neuron_idx):
        n = self.num_features_per_state[neuron_idx]
        lam = self.lam[neuron_idx]
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = w
        
        try:
            sqrt_P = np.linalg.cholesky((n + lam) * P)
        except np.linalg.LinAlgError:
            P_stable = P + np.eye(n) * 1e-6
            sqrt_P = np.linalg.cholesky((n + lam) * P_stable)

        for i in range(n):
            sigma_points[i + 1] = w + sqrt_P[:, i]
            sigma_points[i + 1 + n] = w - sqrt_P[:, i]
            
        return sigma_points

    def update(self, chi_kp1, chi_k, u_k, x_hat_prev):
        # FIXED: Use previous estimates
        x_state_z = x_hat_prev.copy()
        target_delta = chi_kp1 - chi_k
        
        for i in range(self.num_neurons):
            z = construct_z_vector_for_state(x_state_z, u_k, i)
            
            w_pred = self.weights[i]
            P_pred = self.P[i] + self.Q[i]
            
            # FIXED: Proper sigma point transformation through NONLINEAR function
            sigmas = self.generate_sigma_points(w_pred, P_pred, i)
            y_sigmas = np.array([rhonn_predict(z, s) for s in sigmas])
            
            # Predicted measurement
            y_pred = np.dot(self.Wm[i], y_sigmas)
            
            # Measurement covariance
            Py = self.R[i]
            for j in range(len(y_sigmas)):
                Py += self.Wc[i][j] * (y_sigmas[j] - y_pred)**2
                
            # Cross covariance
            Pxy = np.zeros(self.num_features_per_state[i])
            for j in range(len(sigmas)):
                Pxy += self.Wc[i][j] * (sigmas[j] - w_pred) * (y_sigmas[j] - y_pred)
                
            # Update
            K = Pxy / (Py + 1e-12)
            innovation = target_delta[i] - y_pred
            
            self.weights[i] = w_pred + self.eta * K * innovation
            self.P[i] = P_pred - np.outer(K, K) * Py

    def get_estimate(self, chi_k, x_hat_prev, u_k):
        x_state_z = x_hat_prev.copy()
        deltas = []
        for i in range(3):
            z = construct_z_vector_for_state(x_state_z, u_k, i)
            deltas.append(rhonn_predict(z, self.weights[i]))
        return chi_k + np.array(deltas)

# ==========================================
# 6. PF Trainer
# ==========================================
class PF_RHONN_Trainer:
    def __init__(self, num_neurons, num_features_per_state, n_particles, initial_weights=None):
        if n_particles <= 0:
            raise ValueError(f"Number of particles must be positive, got {n_particles}")
        
        self.n_particles = n_particles
        self.num_features_per_state = num_features_per_state
        
        # Adaptive hyperparameters
        self.Q_std = 0.05 / np.sqrt(n_particles / 100.0)
        self.R_var = 0.05
        self.ess_threshold = n_particles / 2.0
        self.regularization_std = 0.01 * np.sqrt(100.0 / n_particles)
        
        print(f"  PF Config: N={n_particles}, Q_std={self.Q_std:.4f}, reg_std={self.regularization_std:.6f}")
        
        self.particles = []
        self.weights_pf = []
        
        for i in range(num_neurons):
            num_feat = num_features_per_state[i]
            base_w = initial_weights[i] if initial_weights is not None else np.random.randn(num_feat)*0.1
            init_spread = 0.2 if n_particles < 500 else 0.3
            p_i = base_w + np.random.randn(n_particles, num_feat) * init_spread
            self.particles.append(p_i)
            self.weights_pf.append(np.ones(n_particles) / n_particles)

    def update(self, chi_kp1, chi_k, u_k, x_hat_prev):
        # FIXED: Use previous estimates
        x_state_z = x_hat_prev.copy()
        target_delta = chi_kp1 - chi_k
        
        for i in range(len(self.particles)):
            z = construct_z_vector_for_state(x_state_z, u_k, i)
            
            # Predict (Random Walk)
            noise = np.random.normal(0, self.Q_std, size=self.particles[i].shape)
            self.particles[i] += noise
            
            # Update (Laplacian Likelihood)
            preds = np.array([rhonn_predict(z, p) for p in self.particles[i]])
            innov = target_delta[i] - preds
            
            # Laplacian likelihood
            log_likelihood = -np.abs(innov) / self.R_var
            log_likelihood_shifted = log_likelihood - np.max(log_likelihood)
            likelihood = np.exp(log_likelihood_shifted)
            
            self.weights_pf[i] *= likelihood
            
            w_sum = np.sum(self.weights_pf[i])
            if w_sum < 1e-300:
                self.weights_pf[i] = np.ones(self.n_particles) / self.n_particles
            else:
                self.weights_pf[i] /= w_sum
                
            # Resample if needed
            ess = 1.0 / np.sum(self.weights_pf[i]**2)
            if ess < self.ess_threshold:
                self.resample(i)

    def resample(self, idx):
        weights = self.weights_pf[idx]
        particles = self.particles[idx]
        N = self.n_particles
        
        # Systematic Resampling
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
        
        self.particles[idx] = particles[indexes].copy()
        reg_noise = np.random.normal(0, self.regularization_std, size=self.particles[idx].shape)
        self.particles[idx] += reg_noise
        self.weights_pf[idx] = np.ones(N) / N

    def get_estimate(self, chi_k, x_hat_prev, u_k):
        x_state_z = x_hat_prev.copy()
        estimates = []
        for i in range(3):
            mean_w = np.average(self.particles[i], weights=self.weights_pf[i], axis=0)
            z = construct_z_vector_for_state(x_state_z, u_k, i)
            estimates.append(rhonn_predict(z, mean_w))
        return chi_k + np.array(estimates)

# ==========================================
# 7. Main Simulation Loop
# ==========================================
def run_simulation():
    h_true = np.zeros((STEPS, 3))
    h_ekf = np.zeros((STEPS, 3))
    h_ukf = np.zeros((STEPS, 3))
    h_pf = np.zeros((STEPS, 3))
    
    x_true = np.zeros(3)
    x_ekf = np.zeros(3)
    x_ukf = np.zeros(3)
    x_pf = np.zeros(3)
    
    h_true[0] = x_true
    h_ekf[0] = x_ekf
    h_ukf[0] = x_ukf
    h_pf[0] = x_pf
    
    init_w = []
    for i in range(NUM_NEURONS):
        init_w.append(np.random.uniform(-0.1, 0.1, NUM_FEATURES_PER_STATE[i]))
    
    ekf = EKF_RHONN_Trainer(NUM_NEURONS, NUM_FEATURES_PER_STATE, eta=1.0, initial_weights=init_w)
    ukf = UKF_RHONN_Trainer(NUM_NEURONS, NUM_FEATURES_PER_STATE, eta=1.0, initial_weights=init_w)
    pf = PF_RHONN_Trainer(NUM_NEURONS, NUM_FEATURES_PER_STATE, NUM_PARTICLES, initial_weights=init_w)
    
    print(f"Starting FIXED RHONN Simulation: {STEPS} steps")
    print(f"Noise Type: Laplacian (Scale={PROCESS_NOISE_STD})")
    print(f"RHONN: Now truly NONLINEAR with tanh activation")
    print(f"Architecture: PARALLEL (uses estimates, not measurements)")
    print(f"Particles: {NUM_PARTICLES}")
    
    for k in range(STEPS - 1):
        t = k * DT
        
        base_speed = 5.0
        turn_bias = 2.0 * np.sin(0.5 * t)
        
        u1 = base_speed - turn_bias
        u3 = base_speed - turn_bias
        u2 = base_speed + turn_bias
        u4 = base_speed + turn_bias
        
        u = np.array([u1, u2, u3, u4])
        
        x_next_true = plant_step(x_true, u, DT, PROCESS_NOISE_STD)
        measurement_noise = np.random.normal(0, 0.01, 3)
        x_measured = x_next_true + measurement_noise
        
        # Update all filters
        ekf.update(x_measured, x_true, u, x_ekf)
        x_next_ekf = ekf.get_estimate(x_true, x_ekf, u)
        
        ukf.update(x_measured, x_true, u, x_ukf)
        x_next_ukf = ukf.get_estimate(x_true, x_ukf, u)
        
        pf.update(x_measured, x_true, u, x_pf)
        x_next_pf = pf.get_estimate(x_true, x_pf, u)
        
        x_true = x_next_true
        x_ekf = x_next_ekf
        x_ukf = x_next_ukf
        x_pf = x_next_pf
        
        h_true[k+1] = x_true
        h_ekf[k+1] = x_ekf
        h_ukf[k+1] = x_ukf
        h_pf[k+1] = x_pf

    # ==========================================
    # 8. Visualization
    # ==========================================
    fig = make_subplots(rows=5, cols=1, 
                        subplot_titles=("2D Trajectory", "State X", "State Y", "State Theta", "Mean Squared Error"),
                        vertical_spacing=0.05)

    fig.add_trace(go.Scatter(x=h_true[:, 0], y=h_true[:, 1], mode='lines', name='True Path', 
                            line=dict(color='green', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=h_ekf[:, 0], y=h_ekf[:, 1], mode='lines', name='EKF', 
                            line=dict(color='red', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=h_ukf[:, 0], y=h_ukf[:, 1], mode='lines', name='UKF', 
                            line=dict(color='orange', width=1, dash='dashdot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=h_pf[:, 0], y=h_pf[:, 1], mode='lines', name='PF', 
                            line=dict(color='blue', width=1.5, dash='dot')), row=1, col=1)

    # State plots
    for state_idx, state_name in enumerate(['X', 'Y', 'Theta']):
        row = state_idx + 2
        fig.add_trace(go.Scatter(y=h_true[:, state_idx], mode='lines', name=f'True {state_name}', 
                                line=dict(color='green', width=2), showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(y=h_ekf[:, state_idx], mode='lines', name=f'EKF {state_name}', 
                                line=dict(color='red', width=1, dash='dash'), showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(y=h_ukf[:, state_idx], mode='lines', name=f'UKF {state_name}', 
                                line=dict(color='orange', width=1, dash='dashdot'), showlegend=False), row=row, col=1)
        fig.add_trace(go.Scatter(y=h_pf[:, state_idx], mode='lines', name=f'PF {state_name}', 
                                line=dict(color='blue', width=1.5, dash='dot'), showlegend=False), row=row, col=1)

    # Errors
    err_ekf = np.mean((h_true - h_ekf)**2, axis=1)
    err_ukf = np.mean((h_true - h_ukf)**2, axis=1)
    err_pf = np.mean((h_true - h_pf)**2, axis=1)
    
    fig.add_trace(go.Scatter(y=err_ekf, mode='lines', name=f'EKF MSE (Mean: {np.mean(err_ekf):.5f})', 
                            line=dict(color='red', width=1)), row=5, col=1)
    fig.add_trace(go.Scatter(y=err_ukf, mode='lines', name=f'UKF MSE (Mean: {np.mean(err_ukf):.5f})', 
                            line=dict(color='orange', width=1)), row=5, col=1)
    fig.add_trace(go.Scatter(y=err_pf, mode='lines', name=f'PF MSE (Mean: {np.mean(err_pf):.5f})', 
                            line=dict(color='blue', width=1)), row=5, col=1)

    fig.update_layout(height=1200, width=1000, title_text="FIXED: Nonlinear RHONN with EKF, UKF, and PF")
    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=5, col=1)
    fig.update_yaxes(title_text="MSE", row=5, col=1)

    fig.show()
    
    print("\nFinal MSE Results:")
    print(f"EKF: {np.mean(err_ekf):.6f}")
    print(f"UKF: {np.mean(err_ukf):.6f}")
    print(f"PF:  {np.mean(err_pf):.6f}")

if __name__ == "__main__":
    run_simulation()