import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. Configuration & Constants
# ==========================================
DT = 0.05
STEPS = 1000
PROCESS_NOISE_STD = 0.05  # Laplacian Scale
SHOT_NOISE_PROB = 0.05    # Probability of outlier
SHOT_NOISE_SCALE = 5.0    # Magnitude of outlier (Large spike)
NUM_PARTICLES = 800
NUM_NEURONS = 2          # theta, theta_dot
NUM_FEATURES = 10        # High Order Connections (2nd Order)
SCALE_FACTOR = 1.0       # Not used for linear output

# ==========================================
# 2. Math & RHONN Utils
# ==========================================
def sigmoid(x):
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
    # [s1, s2, su, s1*s2, s1*su, s2*su, s1^2, s2^2, su^2, 1]
    return np.array([
        s1,
        s2,
        su,
        s1 * s2,
        s1 * su,
        s2 * su,
        s1**2,
        s2**2,
        su**2,
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
    
    # Add noise to input (torque disturbance)
    torque_noisy = torque + np.random.laplace(0, noise_std)
    
    # Add Shot Noise (Non-Gaussian Outliers)
    if np.random.random() < SHOT_NOISE_PROB:
        # Add a large spike to the torque
        torque_noisy += np.random.choice([-1, 1]) * SHOT_NOISE_SCALE

    theta_ddot = -(g/L) * np.sin(theta) - (b/(m*L**2)) * theta_dot + (1.0/(m*L**2)) * torque_noisy
    
    # Add noise to state derivatives
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
        self.Q = [np.eye(num_features) * 1e-4 for _ in range(num_neurons)]
        self.R = [0.01 for _ in range(num_neurons)] # Scalar measurement noise

    def update(self, chi_kp1, chi_k, u_k, x_hat_prev):
        # Parallel: Use Estimated state from k
        x_state_z = x_hat_prev
        z = construct_z_vector(x_state_z, u_k)
        
        # Target is the INCREMENT (Velocity * dt)
        target_delta = chi_kp1 - chi_k
        
        for i in range(self.num_neurons):
            # 1. Predict P
            P_pred = self.P[i] + self.Q[i]
            
            # 2. Kalman Gain
            # Linearized H = Jacobian of NN w.r.t weights
            H = rhonn_predict_jacobian(z, self.weights[i])
            
            # M = R + H.T * P * H
            PH = P_pred @ H
            M = self.R[i] + H @ PH
            
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
        self.Q_std = 0.05  # Diffusion
        self.R_var = 0.05  # Likelihood Scaling
        self.ess_threshold = n_particles / 2.0
        
        # Particles: [neurons, particles, features]
        self.particles = []
        self.weights_pf = []
        
        for i in range(num_neurons):
            base_w = initial_weights[i] if initial_weights is not None else np.random.randn(num_features)*0.1
            # Initialize particles around base weights
            p_i = base_w + np.random.randn(n_particles, num_features) * 0.1
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
            
            # *** ACADEMIC ADVANTAGE ***
            # Use Laplacian Likelihood (exp(-|error|)) instead of Gaussian (exp(-error^2))
            log_likelihood = -np.abs(innov) / self.R_var
            
            # Update weights (in log space to prevent underflow, then exp)
            likelihood = np.exp(log_likelihood)
            self.weights_pf[i] *= likelihood
            
            # Normalize
            w_sum = np.sum(self.weights_pf[i])
            if w_sum < 1e-300:
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
        self.Q = [np.eye(num_features) * 1e-4 for _ in range(num_neurons)]
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
            sqrt_P = np.linalg.cholesky((self.n + self.lam) * P)
        except np.linalg.LinAlgError:
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
            K = Pxy / (Py + 1e-12)
            
            innovation = target_delta[i] - y_pred
            
            self.weights[i] = w_pred + self.eta * K * innovation
            self.P[i] = P_pred - np.outer(K, K) * Py

    def get_estimate(self, chi_k, u_k):
        x_state_z = chi_k
        z = construct_z_vector(x_state_z, u_k)
        deltas = np.array([rhonn_predict(z, self.weights[i]) for i in range(self.num_neurons)])
        return chi_k + deltas

# ==========================================
# 7. Main Simulation Loop
# ==========================================
def run_simulation():
    # History containers
    h_true = np.zeros((STEPS, NUM_NEURONS))
    h_ekf = np.zeros((STEPS, NUM_NEURONS))
    h_ukf = np.zeros((STEPS, NUM_NEURONS))
    h_pf = np.zeros((STEPS, NUM_NEURONS))
    
    # Initial State [theta, theta_dot]
    x_true = np.array([np.pi/2, 0.0]) # Start at 90 degrees
    x_ekf = np.array([np.pi/2, 0.0])
    x_ukf = np.array([np.pi/2, 0.0])
    x_pf = np.array([np.pi/2, 0.0])
    
    h_true[0] = x_true
    h_ekf[0] = x_ekf
    h_ukf[0] = x_ukf
    h_pf[0] = x_pf
    
    # Initial Weights (Shared)
    init_w = np.random.uniform(-0.1, 0.1, (NUM_NEURONS, NUM_FEATURES))
    
    # Initialize Trainers
    ekf = EKF_RHONN_Trainer(NUM_NEURONS, NUM_FEATURES, eta=1.0, initial_weights=init_w)
    ukf = UKF_RHONN_Trainer(NUM_NEURONS, NUM_FEATURES, eta=1.0, initial_weights=init_w)
    pf = PF_RHONN_Trainer(NUM_NEURONS, NUM_FEATURES, NUM_PARTICLES, initial_weights=init_w)
    
    print(f"Starting Pendulum Simulation: {STEPS} steps")
    print(f"Noise Type: Laplacian (Scale={PROCESS_NOISE_STD}) + Shot Noise (Prob={SHOT_NOISE_PROB}, Scale={SHOT_NOISE_SCALE})")
    print(f"EKF Assumption: Gaussian (Standard parameters)")
    print(f"UKF Assumption: Gaussian (Unscented Transform)")
    print(f"PF Assumption: Laplacian (Matched Likelihood)")
    
    for k in range(STEPS - 1):
        t = k * DT
        
        # 1. Controls (Torque)
        # Sinusoidal input to excite dynamics
        torque = 2.0 * np.sin(2.0 * t)
        u = np.array([torque])
        
        # 2. Plant Step
        x_next_true = plant_step(x_true, u, DT, PROCESS_NOISE_STD)
        
        # 3. EKF Update & Predict
        ekf.update(x_next_true, x_true, u, x_ekf)
        x_next_ekf = ekf.get_estimate(x_ekf, u)
        
        # 4. UKF Update & Predict
        ukf.update(x_next_true, x_true, u, x_ukf)
        x_next_ukf = ukf.get_estimate(x_ukf, u)
        
        # 5. PF Update & Predict
        pf.update(x_next_true, x_true, u, x_pf)
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

    # ==========================================
    # 8. Visualization
    # ==========================================
    # Create subplots
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Pendulum Angle (Theta): EKF vs UKF vs PF (with Shot Noise)", "Mean Squared Error over Time"),
                        vertical_spacing=0.15)

    # Plotting Trajectory (Theta)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_true[:, 0], mode='lines', name='True Angle', line=dict(color='green', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_ekf[:, 0], mode='lines', name='EKF', line=dict(color='red', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_ukf[:, 0], mode='lines', name='UKF', line=dict(color='orange', width=1, dash='dashdot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=h_pf[:, 0], mode='lines', name='PF', line=dict(color='blue', width=1.5, dash='dot')), row=1, col=1)

    # Plotting Errors (MSE of full state)
    err_ekf = np.mean((h_true - h_ekf)**2, axis=1)
    err_ukf = np.mean((h_true - h_ukf)**2, axis=1)
    err_pf = np.mean((h_true - h_pf)**2, axis=1)
    
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=err_ekf, mode='lines', name=f'EKF MSE (Mean: {np.mean(err_ekf):.5f})', line=dict(color='red', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=err_ukf, mode='lines', name=f'UKF MSE (Mean: {np.mean(err_ukf):.5f})', line=dict(color='orange', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.arange(STEPS)*DT, y=err_pf, mode='lines', name=f'PF MSE (Mean: {np.mean(err_pf):.5f})', line=dict(color='blue', width=1)), row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=1000, title_text="Pendulum System Identification Results")
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Angle (rad)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="MSE", row=2, col=1)

    fig.show()

if __name__ == "__main__":
    run_simulation()