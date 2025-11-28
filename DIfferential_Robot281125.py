import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. Configuration & Constants
# ==========================================
DT = 0.05
STEPS = 1000
PROCESS_NOISE_STD = 0.1  # Laplacian Scale
NUM_PARTICLES = 200
NUM_NEURONS = 3          # px, py, theta
NUM_FEATURES = 19        # Dimension of RHONN input z

# ==========================================
# 2. Math & RHONN Utils
# ==========================================
def sigmoid(z, beta=1.0):
    # Clip to prevent overflow
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-beta * z))

def construct_z_vector(x_state, u_input):
    """
    Replicates logic/rhonn.ts
    x_state: [px, py, theta]
    u_input: [v, w]
    """
    s_px = sigmoid(x_state[0])
    s_py = sigmoid(x_state[1])
    s_th = sigmoid(x_state[2])
    s_v  = sigmoid(u_input[0])
    s_w  = sigmoid(u_input[1])
    
    # 19 Features
    return np.array([
        s_px, s_py, s_th, s_v, s_w,                           # Linear
        s_px*s_py, s_px*s_th, s_py*s_th,                      # State Cross
        s_px*s_v, s_py*s_v, s_th*s_v, s_th*s_w, s_v*s_w,      # State-Input Cross
        s_px**2, s_py**2, s_th**2, s_v**2, s_w**2,            # Quadratic
        1.0                                                   # Bias
    ])

def rhonn_predict(z, weights):
    return np.dot(weights, z)

# ==========================================
# 3. Plant Dynamics (True System)
# ==========================================
def plant_step(x_k, u_k, dt, noise_std):
    """
    x_k: [px, py, theta]
    u_k: [v, w]
    Generates LAPLACIAN noise (Heavy-tailed)
    """
    # Noise on inputs (Traction loss)
    delta_v = np.random.laplace(0, noise_std * 2)
    delta_w = np.random.laplace(0, noise_std)
    
    # Noise on state
    noise_px = np.random.laplace(0, noise_std / 10)
    noise_py = np.random.laplace(0, noise_std / 10)
    noise_th = np.random.laplace(0, noise_std / 10)
    
    v_actual = u_k[0] + delta_v
    w_actual = u_k[1] + delta_w
    
    px_dot = v_actual * np.cos(x_k[2])
    py_dot = v_actual * np.sin(x_k[2])
    theta_dot = w_actual
    
    x_kp1 = np.zeros(3)
    x_kp1[0] = x_k[0] + dt * px_dot + noise_px
    x_kp1[1] = x_k[1] + dt * py_dot + noise_py
    x_kp1[2] = x_k[2] + dt * theta_dot + noise_th
    
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
        # Series-parallel: Use Measured px, py from k, but estimated theta
        x_state_z = np.array([chi_k[0], chi_k[1], x_hat_prev[2]])
        z = construct_z_vector(x_state_z, u_k)
        
        for i in range(self.num_neurons):
            # 1. Predict P
            P_pred = self.P[i] + self.Q[i]
            
            # 2. Kalman Gain
            # H is just z in this formulation (linear in weights)
            # M = R + z.T * P * z
            Pz = P_pred @ z
            M = self.R[i] + z @ Pz
            
            K = Pz / (M + 1e-12)
            
            # 3. Update Weights
            prediction = np.dot(self.weights[i], z)
            error = chi_kp1[i] - prediction
            
            # w = w + eta * K * error
            self.weights[i] = self.weights[i] + self.eta * K * error
            
            # 4. Update P
            # P = P - K * z.T * P
            # Outer product of K and Pz (since Pz = z.T * P due to symmetry)
            P_update = P_pred - np.outer(K, Pz)
            self.P[i] = 0.5 * (P_update + P_update.T) # Symmetrize

    def get_estimate(self, chi_k, x_hat_prev, u_k):
        x_state_z = np.array([chi_k[0], chi_k[1], x_hat_prev[2]])
        z = construct_z_vector(x_state_z, u_k)
        return np.array([rhonn_predict(z, self.weights[i]) for i in range(3)])

# ==========================================
# 5. PF Trainer (Laplacian Matched Filter)
# ==========================================
class PF_RHONN_Trainer:
    def __init__(self, num_neurons, num_features, n_particles, initial_weights=None):
        self.n_particles = n_particles
        self.num_features = num_features
        
        # Hyperparameters (Tuned in Web App)
        self.Q_std = 0.02  # Diffusion
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
        x_state_z = np.array([chi_k[0], chi_k[1], x_hat_prev[2]])
        z = construct_z_vector(x_state_z, u_k)
        
        for i in range(len(self.particles)):
            # 1. Predict (Random Walk)
            noise = np.random.normal(0, self.Q_std, size=self.particles[i].shape)
            self.particles[i] += noise
            
            # 2. Update (Likelihood)
            # Calculate predictions for all particles
            preds = np.dot(self.particles[i], z)
            innov = chi_kp1[i] - preds
            
            # *** ACADEMIC ADVANTAGE ***
            # Use Laplacian Likelihood (exp(-|error|)) instead of Gaussian (exp(-error^2))
            log_likelihood = -np.abs(innov) / self.R_var
            
            # Update weights (in log space to prevent underflow, then exp)
            # Simplified here to match TS logic:
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

    def get_estimate(self, chi_k, x_hat_prev, u_k):
        x_state_z = np.array([chi_k[0], chi_k[1], x_hat_prev[2]])
        z = construct_z_vector(x_state_z, u_k)
        
        estimates = []
        for i in range(3):
            # Weighted average of particle weights (the neural network weights)
            mean_w = np.average(self.particles[i], weights=self.weights_pf[i], axis=0)
            estimates.append(rhonn_predict(z, mean_w))
            
        return np.array(estimates)

# ==========================================
# 6. Main Simulation Loop
# ==========================================
def run_simulation():
    # History containers
    h_true = np.zeros((STEPS, 3))
    h_ekf = np.zeros((STEPS, 3))
    h_pf = np.zeros((STEPS, 3))
    
    # Initial State
    x_true = np.zeros(3)
    x_ekf = np.zeros(3)
    x_pf = np.zeros(3)
    
    h_true[0] = x_true
    h_ekf[0] = x_ekf
    h_pf[0] = x_pf
    
    # Initial Weights (Shared)
    init_w = np.random.uniform(-0.1, 0.1, (NUM_NEURONS, NUM_FEATURES))
    
    # Initialize Trainers
    ekf = EKF_RHONN_Trainer(NUM_NEURONS, NUM_FEATURES, eta=1.0, initial_weights=init_w)
    pf = PF_RHONN_Trainer(NUM_NEURONS, NUM_FEATURES, NUM_PARTICLES, initial_weights=init_w)
    
    print(f"Starting Simulation: {STEPS} steps")
    print(f"Noise Type: Laplacian (Scale={PROCESS_NOISE_STD})")
    print(f"EKF Assumption: Gaussian (Standard parameters)")
    print(f"PF Assumption: Laplacian (Matched Likelihood)")
    
    for k in range(STEPS - 1):
        t = k * DT
        
        # 1. Controls (Varying Circular)
        v = 0.5 + 0.3 * np.sin(0.5 * t)
        w = 0.3 + 0.2 * np.cos(0.3 * t)
        u = np.array([v, w])
        
        # 2. Plant Step
        x_next_true = plant_step(x_true, u, DT, PROCESS_NOISE_STD)
        
        # 3. EKF Update & Predict
        ekf.update(x_next_true, x_true, u, x_ekf)
        x_next_ekf = ekf.get_estimate(x_true, x_ekf, u)
        
        # 4. PF Update & Predict
        pf.update(x_next_true, x_true, u, x_pf)
        x_next_pf = pf.get_estimate(x_true, x_pf, u)
        
        # 5. Advance
        x_true = x_next_true
        x_ekf = x_next_ekf
        x_pf = x_next_pf
        
        h_true[k+1] = x_true
        h_ekf[k+1] = x_ekf
        h_pf[k+1] = x_pf

    # ==========================================
    # 7. Visualization
    # ==========================================
    # Create subplots
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("2D Trajectory: EKF vs PF in Laplacian Noise", "Mean Squared Error over Time"),
                        vertical_spacing=0.15)

    # Plotting Trajectory
    fig.add_trace(go.Scatter(x=h_true[:, 0], y=h_true[:, 1], mode='lines', name='True Path', line=dict(color='green', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=h_ekf[:, 0], y=h_ekf[:, 1], mode='lines', name='EKF (Gaussian Assum.)', line=dict(color='red', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=h_pf[:, 0], y=h_pf[:, 1], mode='lines', name='PF (Laplacian Match)', line=dict(color='blue', width=1.5, dash='dot')), row=1, col=1)

    # Plotting Errors
    err_ekf = np.mean((h_true - h_ekf)**2, axis=1)
    err_pf = np.mean((h_true - h_pf)**2, axis=1)
    
    fig.add_trace(go.Scatter(y=err_ekf, mode='lines', name=f'EKF MSE (Mean: {np.mean(err_ekf):.5f})', line=dict(color='red', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(y=err_pf, mode='lines', name=f'PF MSE (Mean: {np.mean(err_pf):.5f})', line=dict(color='blue', width=1)), row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=1000, title_text="Differential Robot Simulation Results")
    fig.update_xaxes(title_text="X Position (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y Position (m)", row=1, col=1)
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_yaxes(title_text="MSE", row=2, col=1)

    fig.show()

if __name__ == "__main__":
    run_simulation()