
# Neural Identifier Training - Differential Drive Robot
# Methods: EKF, UKF, Particle Filter
# Con características específicas por neurona

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1) True nonlinear system (Differential Drive Robot)
# ============================================================
def plant_dynamics(state, u):
    """
    Dynamics of a differential drive robot.
    state = [x, y, θ, v, ω] 
        x, y: position (m)
        θ: orientation (rad)
        v: linear velocity (m/s)
        ω: angular velocity (rad/s)
    u = [v_cmd, ω_cmd]: velocity commands
    """
    # Robot Parameters
    m = 2.0          # Mass (kg)
    I = 0.5          # Moment of inertia (kg⋅m²)
    b_v = 0.5        # Linear drag coefficient
    b_ω = 0.3        # Angular drag coefficient
    tau_v = 0.2      # Time constant for linear velocity
    tau_ω = 0.15     # Time constant for angular velocity
    
    x, y, θ, v, ω = state
    v_cmd, ω_cmd = u
    
    # Kinematic equations (position and orientation)
    x_dot = v * np.cos(θ)
    y_dot = v * np.sin(θ)
    θ_dot = ω
    
    # Dynamic equations (velocities with first-order dynamics)
    # Modelo simplificado: τ*dv/dt + v = v_cmd
    v_dot = (v_cmd - v) / tau_v - (b_v / m) * v * np.abs(v)  # Con fricción no lineal
    ω_dot = (ω_cmd - ω) / tau_ω - (b_ω / I) * ω * np.abs(ω)  # Con fricción no lineal
    
    return np.array([x_dot, y_dot, θ_dot, v_dot, ω_dot])

def plant(x_k, u_k, dt=0.01, process_noise_std=1e-3):
    """
    RK4 integration step for better accuracy.
    """
    # RK4 para mayor precisión
    k1 = plant_dynamics(x_k, u_k)
    k2 = plant_dynamics(x_k + 0.5*dt*k1, u_k)
    k3 = plant_dynamics(x_k + 0.5*dt*k2, u_k)
    k4 = plant_dynamics(x_k + dt*k3, u_k)
    
    x_kp1 = x_k + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Add small process noise (Laplacian for heavier tails)
    noise = np.random.laplace(0, process_noise_std, size=5)
    return x_kp1 + noise

# ============================================================
# 2) RHONN structure - CARACTERÍSTICAS POR NEURONA
# ============================================================
def sigmoidal(z, beta=0.5):
    """Sigmoid S(z)."""
    z = np.clip(z, -50, 50) 
    return 1.0 / (1.0 + np.exp(-beta * z))

def construct_z_vector(x_est, u_input, neuron_index):
    """
    Features for Differential Drive Robot - ESPECÍFICAS PARA CADA NEURONA.
    
    x_est = [x, y, θ, v, ω]
    u_input = [v_cmd, ω_cmd]
    neuron_index: índice de la neurona (0=x, 1=y, 2=θ, 3=v, 4=ω)
    """
    x, y, θ, v, ω = x_est
    v_cmd, ω_cmd = u_input
    
    # Términos básicos sigmoidales
    s_x = sigmoidal(x)
    s_y = sigmoidal(y)
    s_θ = sigmoidal(θ)
    s_v = sigmoidal(v)
    s_ω = sigmoidal(ω)
    
    # Términos trigonométricos (importantes para la cinemática)
    cos_θ = np.cos(θ)
    sin_θ = np.sin(θ)
    
    # Comandos escalados
    s_v_cmd = sigmoidal(v_cmd)
    s_ω_cmd = sigmoidal(ω_cmd)
    
    # ========== CARACTERÍSTICAS ESPECÍFICAS POR NEURONA ==========
    
    if neuron_index == 0:  # Neurona para x (posición horizontal)
        # dx/dt = v*cos(θ)
        return np.array([
            s_v,                       # Velocidad lineal
            # cos_θ,                     # Componente direccional
            # s_v * cos_θ,              # Término cinemático principal
            s_θ,                       # Orientación
            s_v * s_θ,                # Interacción velocidad-orientación
            s_v**2,                    # Término cuadrático
            # 1.0                        # Bias
        ])
    
    elif neuron_index == 1:  # Neurona para y (posición vertical)
        # dy/dt = v*sin(θ)
        return np.array([
            # s_v,                       # Velocidad lineal
            # sin_θ,                     # Componente direccional
            # s_v * sin_θ,              # Término cinemático principal
            s_θ,                       # Orientación
            s_v * s_θ,                # Interacción velocidad-orientación
            s_v**2,                    # Término cuadrático
            # 1.0                        # Bias
        ])
    
    elif neuron_index == 2:  # Neurona para θ (orientación)
        # dθ/dt = ω
        return np.array([
            s_ω,                       # Velocidad angular (término principal)
            s_ω**2,                    # Término cuadrático
            # s_ω**3,                    # Término cúbico (no linealidad)
            s_v * s_ω,                # Acoplamiento con velocidad lineal
            s_θ,                       # Orientación actual
            # s_ω_cmd * 0.2,            # Comando de velocidad angular
            # 1.0                        # Bias
        ])
    
    elif neuron_index == 3:  # Neurona para v (velocidad lineal)
        # dv/dt = (v_cmd - v)/tau - friction
        return np.array([
            s_v,                       # Estado actual
            s_v**2,                    # Fricción cuadrática
            s_v**3,                    # Fricción cúbica
            # v_cmd * 0.1,              # Comando (lineal, no saturado)
            s_v_cmd,                   # Comando (sigmoidal)
            # s_v * s_v_cmd,            # Interacción estado-comando
            s_ω,                       # Acoplamiento con velocidad angular
            # s_v * np.abs(v),          # Término de fricción absoluta
            # 1.0                        # Bias
        ])
    
    elif neuron_index == 4:  # Neurona para ω (velocidad angular)
        # dω/dt = (ω_cmd - ω)/tau - friction
        return np.array([
            s_ω,                       # Estado actual
            # s_ω**2,                    # Fricción cuadrática
            s_ω**3,                    # Fricción cúbica
            # ω_cmd * 0.1,              # Comando (lineal, no saturado)
            # s_ω_cmd,                   # Comando (sigmoidal)
            s_ω * s_ω_cmd,            # Interacción estado-comando
            s_v,                       # Acoplamiento con velocidad lineal
            # s_ω * np.abs(ω),          # Término de fricción absoluta
            # 1.0                        # Bias
        ])
    
    else:
        raise ValueError(f"Índice de neurona inválido: {neuron_index}")

# Función auxiliar para obtener el tamaño de características de cada neurona
def get_z_size(neuron_index):
    """Retorna el número de características para una neurona dada."""
    if neuron_index == 0:  # x
        return 4
    elif neuron_index == 1:  # y
        return 3
    elif neuron_index == 2:  # θ
        return 4
    elif neuron_index == 3:  # v
        return 5
    elif neuron_index == 4:  # ω
        return 4
    else:
        raise ValueError(f"Índice de neurona inválido: {neuron_index}")

# ============================================================
# 3) Trainers (EKF, UKF, PF) - ADAPTADOS PARA MÚLTIPLES TAMAÑOS
# ============================================================

class Generic_RHONN_Trainer:
    """ Base class to handle the loop logic easily """
    def get_prediction(self, weights, x_k, u_k, neuron_idx):
        z = construct_z_vector(x_k, u_k, neuron_idx)
        return np.dot(weights, z)

class EKF_Trainer(Generic_RHONN_Trainer):
    def __init__(self, n_neurons, eta=1.0, P0=1.0, Q=1e-3, R=1e-5):
        self.n_neurons = n_neurons
        self.weights = [np.random.randn(get_z_size(i))*0.1 for i in range(n_neurons)]
        self.P = [np.eye(get_z_size(i))*P0 for i in range(n_neurons)]
        self.Q_matrices = [np.eye(get_z_size(i))*Q for i in range(n_neurons)]
        self.R = R
        self.eta = eta

    def update(self, x_kp1, x_k, u_k):
        for i in range(self.n_neurons):
            z = construct_z_vector(x_k, u_k, i)
            H = z.reshape(-1, 1)
            
            # Predict P
            P_pred = self.P[i] + self.Q_matrices[i]
            
            # Kalman Gain
            S = self.R + (H.T @ P_pred @ H)[0,0]
            K = (P_pred @ H).flatten() / S
            
            # Error
            y_pred = np.dot(self.weights[i], z)
            err = x_kp1[i] - y_pred
            
            # Update Weights
            self.weights[i] += self.eta * K * err
            
            # Update Covariance (Joseph form)
            I_KH = np.eye(len(z)) - np.outer(K, z)
            self.P[i] = I_KH @ P_pred @ I_KH.T + np.outer(K, K)*self.R

class UKF_Trainer(Generic_RHONN_Trainer):
    def __init__(self, n_neurons, eta=1.0, alpha=1e-2):
        self.n_neurons = n_neurons
        self.weights = [np.random.randn(get_z_size(i))*0.1 for i in range(n_neurons)]
        self.P = [np.eye(get_z_size(i)) for i in range(n_neurons)]
        self.Q_matrices = [np.eye(get_z_size(i))*1e-4 for i in range(n_neurons)]
        self.R = 1e-5
        self.eta = eta
        self.alpha = alpha

    def update(self, x_kp1, x_k, u_k):
        for i in range(self.n_neurons):
            z = construct_z_vector(x_k, u_k, i)
            n = get_z_size(i)
            
            # Sigma params
            lambda_ = self.alpha**2 * n - n
            Wm = np.full(2*n+1, 1/(2*(n+lambda_)))
            Wc = np.copy(Wm)
            Wm[0] = lambda_/(n+lambda_)
            Wc[0] = Wm[0] + (3 - self.alpha**2)
            
            # Generate Sigmas
            try:
                L = np.linalg.cholesky((n + lambda_) * self.P[i])
            except:
                L = np.eye(n) * 0.1
                
            sigmas = np.zeros((2*n+1, n))
            sigmas[0] = self.weights[i]
            for k in range(n):
                sigmas[k+1] = self.weights[i] + L[:,k]
                sigmas[n+k+1] = self.weights[i] - L[:,k]
            
            # Transform
            Y_sigmas = np.dot(sigmas, z)
            y_mean = np.sum(Wm * Y_sigmas)
            
            # Covariances
            Py = np.sum(Wc * (Y_sigmas - y_mean)**2) + self.R
            Pxy = np.zeros(n)
            for k in range(2*n+1):
                Pxy += Wc[k] * (sigmas[k] - self.weights[i]) * (Y_sigmas[k] - y_mean)
                
            # Update
            K = Pxy / Py
            err = x_kp1[i] - y_mean
            self.weights[i] += self.eta * K * err
            self.P[i] -= np.outer(K, K) * Py
            
            # Regularize P
            self.P[i] += np.eye(n)*1e-6

class PF_Trainer(Generic_RHONN_Trainer):
    def __init__(self, n_neurons, n_particles=800):
        self.n_neurons = n_neurons
        self.n_particles = n_particles
        self.particles = [np.random.randn(n_particles, get_z_size(i))*0.2 
                         for i in range(n_neurons)]
        self.weights_pf = [np.ones(n_particles)/n_particles for _ in range(n_neurons)]
        self.Q_std = 0.2
        self.R_std = 0.1

    def update(self, x_kp1, x_k, u_k):
        for i in range(self.n_neurons):
            z = construct_z_vector(x_k, u_k, i)
            n_weights = get_z_size(i)
            
            # 1. Drift
            self.particles[i] += np.random.randn(self.n_particles, n_weights) * self.Q_std
            
            # 2. Weight
            preds = self.particles[i] @ z
            err = x_kp1[i] - preds
            likelihood = np.exp(-0.5 * (err/self.R_std)**2)
            self.weights_pf[i] *= (likelihood + 1e-300)
            self.weights_pf[i] /= np.sum(self.weights_pf[i])
            
            # 3. Resample
            eff_N = 1.0 / np.sum(self.weights_pf[i]**2)
            if eff_N < self.n_particles/2:
                indices = np.random.choice(self.n_particles, self.n_particles, 
                                         p=self.weights_pf[i])
                self.particles[i] = self.particles[i][indices]
                self.weights_pf[i].fill(1.0/self.n_particles)
                
    def get_estimates(self):
        return [np.average(self.particles[i], axis=0, weights=self.weights_pf[i]) 
                for i in range(self.n_neurons)]

# ============================================================
# 4) Parameter Optimization (Optional)
# ============================================================
def run_simulation_with_params(params, filter_type='EKF', n_steps=500, verbose=False):
    """
    Run a simulation with given parameters and return MSE.
    Used for parameter optimization.
    """
    dt = 0.01
    n_states = 5
    
    # Unpack parameters based on filter type
    try:
        if filter_type == 'EKF':
            eta = params[0]
            P0 = 10 ** params[1]
            Q = 10 ** params[2]
            R = 10 ** params[3]
            trainer = EKF_Trainer(n_states, eta=eta, P0=P0, Q=Q, R=R)
            
        elif filter_type == 'UKF':
            eta = params[0]
            alpha = 10 ** params[1]
            trainer = UKF_Trainer(n_states, eta=eta, alpha=alpha)
            
        elif filter_type == 'PF':
            n_particles = int(params[0] * 100)
            trainer = PF_Trainer(n_states, n_particles=n_particles)
            
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Initialize arrays
        x_true = np.zeros((n_steps, 5))
        x_est = np.zeros((n_steps, 5))
        x_true[0] = [0.0, 0.0, 0.0, 0.0, 0.0]
        x_est[0] = x_true[0]
        
        # Generate excitation input
        t = np.linspace(0, (n_steps-1)*dt, n_steps)
        u_hist = np.zeros((n_steps, 2))
        for k in range(n_steps):
            v_cmd = 1.0 + 0.5 * np.sin(0.5 * t[k])
            ω_cmd = 0.8 * np.sin(1.0 * t[k])
            if 100 < k < 120:
                v_cmd = 2.0
                ω_cmd = 1.5
            elif 250 < k < 270:
                v_cmd = 0.5
                ω_cmd = -1.2
            u_hist[k] = [v_cmd, ω_cmd]
        
        # Run simulation
        for k in range(n_steps - 1):
            x_true[k+1] = plant(x_true[k], u_hist[k], dt)
            trainer.update(x_true[k+1], x_true[k], u_hist[k])
            
            if filter_type == 'PF':
                w_est = trainer.get_estimates()
                for i in range(5):
                    z = construct_z_vector(x_true[k], u_hist[k], i)
                    x_est[k+1, i] = np.dot(w_est[i], z)
            else:
                for i in range(5):
                    z = construct_z_vector(x_true[k], u_hist[k], i)
                    x_est[k+1, i] = np.dot(trainer.weights[i], z)
        
        # Calculate total MSE
        total_mse = 0.0
        for i in range(5):
            total_mse += np.mean((x_true[:, i] - x_est[:, i])**2)
        
        if verbose:
            print(f"  MSE: {total_mse:.6e} | Params: {params}")
        
        return total_mse
    
    except Exception as e:
        if verbose:
            print(f"  Error with params {params}: {e}")
        return 1e10

def optimize_filter_parameters(filter_type='EKF', method='differential_evolution', 
                               n_steps=500, maxiter=50, verbose=True):
    """
    Optimize filter parameters using scipy optimization.
    
    Parameters:
    -----------
    filter_type : str
        Type of filter to optimize: 'EKF', 'UKF', or 'PF'
    method : str
        Optimization method: 'differential_evolution' or 'nelder-mead'
    n_steps : int
        Number of simulation steps for evaluation
    maxiter : int
        Maximum iterations for optimization
    verbose : bool
        Print optimization progress
    
    Returns:
    --------
    dict : Optimized parameters and final MSE
    """
    print(f"\n{'='*70}")
    print(f"🔧 OPTIMIZING {filter_type} PARAMETERS (Robot Diferencial)")
    print(f"{'='*70}")
    print(f"Method: {method}")
    print(f"Simulation length: {n_steps} steps")
    print(f"Max iterations: {maxiter}\n")
    
    # Define parameter bounds and initial guesses
    if filter_type == 'EKF':
        bounds = [(0.1, 2.0), (-2, 1), (-6, -2), (-4, -1)]  # eta, log10(P0), log10(Q), log10(R)
        x0 = [1.0, 0.0, -4, -2]
        param_names = ['eta', 'log10(P0)', 'log10(Q)', 'log10(R)']
        
    elif filter_type == 'UKF':
        bounds = [(0.1, 2.0), (-3, 0)]  # eta, log10(alpha)
        x0 = [0.9, -2]
        param_names = ['eta', 'log10(alpha)']
        
    elif filter_type == 'PF':
        bounds = [(5, 15)]  # n_particles_ratio (multiplied by 100)
        x0 = [8]
        param_names = ['n_particles_ratio']
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Objective function
    objective = lambda params: run_simulation_with_params(params, filter_type, n_steps, verbose=False)
    
    # Optimize
    if method == 'differential_evolution':
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=maxiter,
            popsize=15,
            seed=42,
            atol=1e-6,
            tol=1e-6,
            workers=1,
            updating='deferred',
            disp=verbose
        )
    elif method == 'nelder-mead':
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'disp': verbose, 'xatol': 1e-6, 'fatol': 1e-6}
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    # Extract and display results
    optimized_params = result.x
    final_mse = result.fun
    
    print(f"\n{'='*70}")
    print(f"✅ OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Final MSE: {final_mse:.6e}")
    print(f"\nOptimized Parameters:")
    
    # Display optimized parameters
    for name, value in zip(param_names, optimized_params):
        print(f"  {name}: {value:.6f}")
    
    # Return dictionary with optimized parameters
    if filter_type == 'EKF':
        return {
            'eta': optimized_params[0],
            'P0': 10 ** optimized_params[1],
            'Q': 10 ** optimized_params[2],
            'R': 10 ** optimized_params[3],
            'mse': final_mse
        }
    elif filter_type == 'UKF':
        return {
            'eta': optimized_params[0],
            'alpha': 10 ** optimized_params[1],
            'mse': final_mse
        }
    elif filter_type == 'PF':
        return {
            'n_particles': int(optimized_params[0] * 100),
            'mse': final_mse
        }

# ============================================================
# 5) Simulation Main Loop
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # OPTIONAL: Parameter Optimization
    # ============================================================
    # Uncomment to optimize parameters for a specific filter:
    # optimized_ekf = optimize_filter_parameters('EKF', method='differential_evolution', n_steps=500, maxiter=30)
    # optimized_ukf = optimize_filter_parameters('UKF', method='differential_evolution', n_steps=500, maxiter=30)
    # optimized_pf = optimize_filter_parameters('PF', method='differential_evolution', n_steps=500, maxiter=30)
    
    # ============================================================
    # Main Simulation
    # ============================================================
    n_steps = 1500
    dt = 0.01
    t = np.linspace(0, (n_steps-1)*dt, n_steps)
    
    n_states = 5  # [x, y, θ, v, ω]
    
    # Init Trainers (use optimized parameters if available)
    ekf = EKF_Trainer(n_states, eta=1.0)  # Use optimized_ekf['eta'], etc. if optimized
    ukf = UKF_Trainer(n_states, eta=1.0, Q=1e-4, R=1e-5)  # Use optimized_ukf['eta'], etc. if optimized
    pf = PF_Trainer(n_states, n_particles=1000)  # Use optimized_pf['n_particles'] if optimized
    
    # Arrays
    x_true = np.zeros((n_steps, 5))
    x_est_ekf = np.zeros((n_steps, 5))
    x_est_ukf = np.zeros((n_steps, 5))
    x_est_pf = np.zeros((n_steps, 5))
    
    # Initial Conditions (robot at origin, facing right)
    x_true[0] = [0.0, 0.0, 0.0, 0.0, 0.0]
    x_est_ekf[0] = x_true[0]
    x_est_ukf[0] = x_true[0]
    x_est_pf[0]  = x_true[0]
    
    # Excitation Input (comandos de velocidad variados)
    u_hist = np.zeros((n_steps, 2))
    for k in range(n_steps):
        # Trayectoria tipo "figure-8" o lemniscata
        v_cmd = 1.0 + 0.5 * np.sin(0.5 * t[k])           # Velocidad lineal variable
        ω_cmd = 0.8 * np.sin(1.0 * t[k])                 # Velocidad angular variable
        
        # Agregar algunos cambios bruscos para excitar la dinámica
        if 300 < k < 320:
            v_cmd = 2.0
            ω_cmd = 1.5
        elif 700 < k < 720:
            v_cmd = 0.5
            ω_cmd = -1.2
        elif 1100 < k < 1120:
            v_cmd = -0.8
            ω_cmd = 0.0
            
        u_hist[k] = [v_cmd, ω_cmd]

    print("Simulating Differential Drive Robot with neuron-specific features...")
    print("\nEstructura de características por neurona:")
    state_names = ['x (pos)', 'y (pos)', 'θ (orient)', 'v (lin vel)', 'ω (ang vel)']
    for i in range(n_states):
        print(f"  Neurona {i} ({state_names[i]}): {get_z_size(i)} características")
    
    for k in range(n_steps - 1):
        # 1. Physics Step
        x_true[k+1] = plant(x_true[k], u_hist[k], dt) + x_true[k+1] + np.random.randn(x_true[k+1].shape[0]) * 0.05
        
        # 2. Identify (Series-Parallel: usa estado real)
        # EKF
        ekf.update(x_true[k+1], x_true[k], u_hist[k])
        for i in range(5): 
            z = construct_z_vector(x_true[k], u_hist[k], i)
            x_est_ekf[k+1, i] = np.dot(ekf.weights[i], z)
            
        # UKF
        ukf.update(x_true[k+1], x_true[k], u_hist[k])
        for i in range(5): 
            z = construct_z_vector(x_true[k], u_hist[k], i)
            x_est_ukf[k+1, i] = np.dot(ukf.weights[i], z)
            
        # PF
        pf.update(x_true[k+1], x_true[k], u_hist[k])
        w_pf = pf.get_estimates()
        for i in range(5): 
            z = construct_z_vector(x_true[k], u_hist[k], i)
            x_est_pf[k+1, i] = np.dot(w_pf[i], z)
            
        if k % 200 == 0: print(f"Step {k}/{n_steps-1}")

    # ============================================================
    # 6) Visualización - Formato Tesis
    # ============================================================
    
    print("\nGenerando visualizaciones...")
    
    # Configuración de formato para tesis
    thesis_config = {
        'font_family': 'Computer Modern, serif',
        'font_size': 14,
        'title_font_size': 16,
        'legend_font_size': 12,
        'line_width_true': 2.5,
        'line_width_est': 2.0,
        'plot_width': 1000,
        'plot_height': 500,
        'grid_color': 'rgba(200, 200, 200, 0.3)',
        'grid_width': 0.5
    }
    
    # --- Calcular MSE ---
    mse_x_ekf = np.mean((x_true[:, 0] - x_est_ekf[:, 0])**2)
    mse_y_ekf = np.mean((x_true[:, 1] - x_est_ekf[:, 1])**2)
    mse_θ_ekf = np.mean((x_true[:, 2] - x_est_ekf[:, 2])**2)
    mse_v_ekf = np.mean((x_true[:, 3] - x_est_ekf[:, 3])**2)
    mse_ω_ekf = np.mean((x_true[:, 4] - x_est_ekf[:, 4])**2)
    
    mse_x_ukf = np.mean((x_true[:, 0] - x_est_ukf[:, 0])**2)
    mse_y_ukf = np.mean((x_true[:, 1] - x_est_ukf[:, 1])**2)
    mse_θ_ukf = np.mean((x_true[:, 2] - x_est_ukf[:, 2])**2)
    mse_v_ukf = np.mean((x_true[:, 3] - x_est_ukf[:, 3])**2)
    mse_ω_ukf = np.mean((x_true[:, 4] - x_est_ukf[:, 4])**2)
    
    mse_x_pf = np.mean((x_true[:, 0] - x_est_pf[:, 0])**2)
    mse_y_pf = np.mean((x_true[:, 1] - x_est_pf[:, 1])**2)
    mse_θ_pf = np.mean((x_true[:, 2] - x_est_pf[:, 2])**2)
    mse_v_pf = np.mean((x_true[:, 3] - x_est_pf[:, 3])**2)
    mse_ω_pf = np.mean((x_true[:, 4] - x_est_pf[:, 4])**2)
    
    mse_total_ekf = mse_x_ekf + mse_y_ekf + mse_θ_ekf + mse_v_ekf + mse_ω_ekf
    mse_total_ukf = mse_x_ukf + mse_y_ukf + mse_θ_ukf + mse_v_ukf + mse_ω_ukf
    mse_total_pf = mse_x_pf + mse_y_pf + mse_θ_pf + mse_v_pf + mse_ω_pf
    
    # Reporte
    print("\n" + "="*70)
    print("🏆 MEJOR FILTRO: ", end="")
    mse_dict = {'EKF': mse_total_ekf, 'UKF': mse_total_ukf, 'PF': mse_total_pf}
    best_filter = min(mse_dict, key=mse_dict.get)
    print(f"{best_filter} (MSE total: {mse_dict[best_filter]:.6f})")
    print("="*70)
    
    print("\n--- Comparación de Desempeño (MSE) - Robot Diferencial ---")
    print(f"EKF MSE x:   {mse_x_ekf:.6f} | MSE θ:   {mse_θ_ekf:.6f} | MSE v:   {mse_v_ekf:.6f}")
    print(f"EKF MSE y:   {mse_y_ekf:.6f} | MSE ω:   {mse_ω_ekf:.6f}")
    print(f"UKF MSE x:   {mse_x_ukf:.6f} | MSE θ:   {mse_θ_ukf:.6f} | MSE v:   {mse_v_ukf:.6f}")
    print(f"UKF MSE y:   {mse_y_ukf:.6f} | MSE ω:   {mse_ω_ukf:.6f}")
    print(f"PF  MSE x:   {mse_x_pf:.6f} | MSE θ:   {mse_θ_pf:.6f} | MSE v:   {mse_v_pf:.6f}")
    print(f"PF  MSE y:   {mse_y_pf:.6f} | MSE ω:   {mse_ω_pf:.6f}")
    
    # --- Trayectoria 2D (Vista superior) ---
    fig_traj = go.Figure()
    
    # Trayectoria real
    fig_traj.add_trace(go.Scatter(
        x=x_true[:, 0], y=x_true[:, 1],
        mode='lines',
        name='Trayectoria Real',
        line=dict(color='#000000', width=thesis_config['line_width_true']),
    ))
    
    # Trayectorias estimadas
    fig_traj.add_trace(go.Scatter(
        x=x_est_ekf[:, 0], y=x_est_ekf[:, 1],
        mode='lines',
        name='EKF-RHONN',
        line=dict(color='#1f77b4', width=thesis_config['line_width_est'], dash='dash'),
    ))
    
    fig_traj.add_trace(go.Scatter(
        x=x_est_ukf[:, 0], y=x_est_ukf[:, 1],
        mode='lines',
        name='UKF-RHONN',
        line=dict(color='#2ca02c', width=thesis_config['line_width_est'], dash='dot'),
    ))
    
    fig_traj.add_trace(go.Scatter(
        x=x_est_pf[:, 0], y=x_est_pf[:, 1],
        mode='lines',
        name='PF-RHONN',
        line=dict(color='#d62728', width=thesis_config['line_width_est'], dash='dashdot'),
    ))
    
    # Puntos de inicio y final
    fig_traj.add_trace(go.Scatter(
        x=[x_true[0, 0]], y=[x_true[0, 1]],
        mode='markers',
        name='Inicio',
        marker=dict(size=15, color='green', symbol='circle'),
    ))
    
    fig_traj.add_trace(go.Scatter(
        x=[x_true[-1, 0]], y=[x_true[-1, 1]],
        mode='markers',
        name='Final',
        marker=dict(size=15, color='red', symbol='square'),
    ))
    
    fig_traj.update_layout(
        title={
            'text': 'Trayectoria del Robot Diferencial (Vista Superior)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': thesis_config['title_font_size'], 'family': thesis_config['font_family']}
        },
        xaxis_title='Posición x (m)',
        yaxis_title='Posición y (m)',
        xaxis=dict(
            showgrid=True,
            gridcolor=thesis_config['grid_color'],
            gridwidth=thesis_config['grid_width'],
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1.5,
            ticklen=5,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=thesis_config['grid_color'],
            gridwidth=thesis_config['grid_width'],
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1.5,
            ticklen=5
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=thesis_config['legend_font_size'], family=thesis_config['font_family'])
        ),
        font=dict(size=thesis_config['font_size'], family=thesis_config['font_family']),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=thesis_config['plot_width'],
        height=thesis_config['plot_width'],  # Cuadrado
        margin=dict(l=80, r=40, t=80, b=60)
    )
    
    # Guardar figura
    fig_traj.write_image("01_trayectoria_xy_serie.png", width=thesis_config['plot_width'], height=thesis_config['plot_width'])
    fig_traj.write_html("01_trayectoria_xy_serie.html")
    fig_traj.show()
    
    # --- Gráficas por Estado ---
    states_info = [
        {'idx': 0, 'var': 'x', 'desc': 'Posición Horizontal', 'y_label': 'Posición x (m)'},
        {'idx': 1, 'var': 'y', 'desc': 'Posición Vertical', 'y_label': 'Posición y (m)'},
        {'idx': 2, 'var': 'θ', 'desc': 'Orientación', 'y_label': 'Ángulo θ (rad)'},
        {'idx': 3, 'var': 'v', 'desc': 'Velocidad Lineal', 'y_label': 'Velocidad v (m/s)'},
        {'idx': 4, 'var': 'ω', 'desc': 'Velocidad Angular', 'y_label': 'Velocidad ω (rad/s)'}
    ]
    
    for state_info in states_info:
        i = state_info['idx']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=t, y=x_true[:, i],
            mode='lines',
            name='Estado Real',
            line=dict(color='#000000', width=thesis_config['line_width_true']),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=t, y=x_est_ekf[:, i],
            mode='lines',
            name='EKF-RHONN',
            line=dict(color='#1f77b4', width=thesis_config['line_width_est'], dash='dash'),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=t, y=x_est_ukf[:, i],
            mode='lines',
            name='UKF-RHONN',
            line=dict(color='#2ca02c', width=thesis_config['line_width_est'], dash='dot'),
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=t, y=x_est_pf[:, i],
            mode='lines',
            name='PF-RHONN',
            line=dict(color='#d62728', width=thesis_config['line_width_est'], dash='dashdot'),
            showlegend=True
        ))
        
        fig.update_layout(
            title={
                'text': f'Estado {state_info["var"]}: {state_info["desc"]} - Robot Diferencial',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': thesis_config['title_font_size'], 'family': thesis_config['font_family']}
            },
            xaxis_title='Tiempo (s)',
            yaxis_title=state_info['y_label'],
            xaxis=dict(
                showgrid=True,
                gridcolor=thesis_config['grid_color'],
                gridwidth=thesis_config['grid_width'],
                showline=True,
                linewidth=1.5,
                linecolor='black',
                mirror=True,
                ticks='outside',
                tickwidth=1.5,
                ticklen=5
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=thesis_config['grid_color'],
                gridwidth=thesis_config['grid_width'],
                showline=True,
                linewidth=1.5,
                linecolor='black',
                mirror=True,
                ticks='outside',
                tickwidth=1.5,
                ticklen=5
            ),
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=thesis_config['legend_font_size'], family=thesis_config['font_family'])
            ),
            font=dict(size=thesis_config['font_size'], family=thesis_config['font_family']),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=thesis_config['plot_width'],
            height=thesis_config['plot_height'],
            margin=dict(l=80, r=40, t=80, b=60)
        )
        
        # Guardar figura por estado
        filename = f"02_estado_{state_info['var']}_serie.png"
        fig.write_image(filename, width=thesis_config['plot_width'], height=thesis_config['plot_height'])
        fig.write_html(f"02_estado_{state_info['var']}_serie.html")
        fig.show()
    
    # --- Gráfica de barras comparando MSE ---
    fig_mse = go.Figure()
    
    filters = ['EKF-RHONN', 'UKF-RHONN', 'PF-RHONN']
    
    fig_mse.add_trace(go.Bar(
        name='Posición x',
        x=filters,
        y=[mse_x_ekf, mse_x_ukf, mse_x_pf],
        marker_color='#636EFA',
        text=[f'{mse_x_ekf:.2e}', f'{mse_x_ukf:.2e}', f'{mse_x_pf:.2e}'],
        textposition='outside'
    ))
    
    fig_mse.add_trace(go.Bar(
        name='Posición y',
        x=filters,
        y=[mse_y_ekf, mse_y_ukf, mse_y_pf],
        marker_color='#EF553B',
        text=[f'{mse_y_ekf:.2e}', f'{mse_y_ukf:.2e}', f'{mse_y_pf:.2e}'],
        textposition='outside'
    ))
    
    fig_mse.add_trace(go.Bar(
        name='Orientación θ',
        x=filters,
        y=[mse_θ_ekf, mse_θ_ukf, mse_θ_pf],
        marker_color='#00CC96',
        text=[f'{mse_θ_ekf:.2e}', f'{mse_θ_ukf:.2e}', f'{mse_θ_pf:.2e}'],
        textposition='outside'
    ))
    
    fig_mse.add_trace(go.Bar(
        name='Velocidad v',
        x=filters,
        y=[mse_v_ekf, mse_v_ukf, mse_v_pf],
        marker_color='#AB63FA',
        text=[f'{mse_v_ekf:.2e}', f'{mse_v_ukf:.2e}', f'{mse_v_pf:.2e}'],
        textposition='outside'
    ))
    
    fig_mse.add_trace(go.Bar(
        name='Velocidad ω',
        x=filters,
        y=[mse_ω_ekf, mse_ω_ukf, mse_ω_pf],
        marker_color='#FFA15A',
        text=[f'{mse_ω_ekf:.2e}', f'{mse_ω_ukf:.2e}', f'{mse_ω_pf:.2e}'],
        textposition='outside'
    ))
    
    fig_mse.update_layout(
        title={
            'text': 'Comparación de Error Cuadrático Medio (MSE) - Robot Diferencial',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': thesis_config['title_font_size'], 'family': thesis_config['font_family']}
        },
        xaxis_title='Tipo de Filtro',
        yaxis_title='Error Cuadrático Medio (MSE)',
        yaxis=dict(
            type='log',
            showgrid=True,
            gridcolor=thesis_config['grid_color'],
            gridwidth=thesis_config['grid_width'],
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1.5,
            ticklen=5
        ),
        xaxis=dict(
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1.5,
            ticklen=5
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=thesis_config['legend_font_size'], family=thesis_config['font_family'])
        ),
        barmode='group',
        font=dict(size=thesis_config['font_size'], family=thesis_config['font_family']),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=thesis_config['plot_width'],
        height=thesis_config['plot_height'],
        margin=dict(l=80, r=40, t=100, b=60)
    )
    
    # Guardar figura
    fig_mse.write_image("03_comparacion_MSE_serie.png", width=thesis_config['plot_width'], height=thesis_config['plot_height'])
    fig_mse.write_html("03_comparacion_MSE_serie.html")
    fig_mse.show()
    
    print("\n✅ Visualización completa.")
 
 
# Neural Identifier Training - Differential Drive Robot
# Methods: EKF, UKF, Particle Filter
# Con características específicas por neurona

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1) True nonlinear system (Differential Drive Robot)
# ============================================================
def plant_dynamics(state, u):
    """
    Dynamics of a differential drive robot.
    state = [x, y, θ, v, ω] 
        x, y: position (m)
        θ: orientation (rad)
        v: linear velocity (m/s)
        ω: angular velocity (rad/s)
    u = [v_cmd, ω_cmd]: velocity commands
    """
    # Robot Parameters
    m = 2.0          # Mass (kg)
    I = 0.5          # Moment of inertia (kg⋅m²)
    b_v = 0.5        # Linear drag coefficient
    b_ω = 0.3        # Angular drag coefficient
    tau_v = 0.2      # Time constant for linear velocity
    tau_ω = 0.15     # Time constant for angular velocity
    
    x, y, θ, v, ω = state
    v_cmd, ω_cmd = u
    
    # Kinematic equations (position and orientation)
    x_dot = v * np.cos(θ)
    y_dot = v * np.sin(θ)
    θ_dot = ω
    
    # Dynamic equations (velocities with first-order dynamics)
    # Modelo simplificado: τ*dv/dt + v = v_cmd
    v_dot = (v_cmd - v) / tau_v - (b_v / m) * v * np.abs(v)  # Con fricción no lineal
    ω_dot = (ω_cmd - ω) / tau_ω - (b_ω / I) * ω * np.abs(ω)  # Con fricción no lineal
    
    return np.array([x_dot, y_dot, θ_dot, v_dot, ω_dot])

def plant(x_k, u_k, dt=0.01, process_noise_std=1e-3):
    """
    RK4 integration step for better accuracy.
    """
    # RK4 para mayor precisión
    k1 = plant_dynamics(x_k, u_k)
    k2 = plant_dynamics(x_k + 0.5*dt*k1, u_k)
    k3 = plant_dynamics(x_k + 0.5*dt*k2, u_k)
    k4 = plant_dynamics(x_k + dt*k3, u_k)
    
    x_kp1 = x_k + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    # Add small process noise (Laplacian for heavier tails)
    noise = np.random.laplace(0, process_noise_std, size=5)
    return x_kp1 + noise

# ============================================================
# 2) RHONN structure - CARACTERÍSTICAS POR NEURONA
# ============================================================
def sigmoidal(z, beta=0.5):
    """Sigmoid S(z)."""
    z = np.clip(z, -50, 50) 
    return 1.0 / (1.0 + np.exp(-beta * z))

def construct_z_vector(x_est, u_input, neuron_index):
    """
    Features for Differential Drive Robot - ESPECÍFICAS PARA CADA NEURONA.
    
    x_est = [x, y, θ, v, ω]
    u_input = [v_cmd, ω_cmd]
    neuron_index: índice de la neurona (0=x, 1=y, 2=θ, 3=v, 4=ω)
    """
    x, y, θ, v, ω = x_est
    v_cmd, ω_cmd = u_input
    
    # Términos básicos sigmoidales
    s_x = sigmoidal(x)
    s_y = sigmoidal(y)
    s_θ = sigmoidal(θ)
    s_v = sigmoidal(v)
    s_ω = sigmoidal(ω)
    
    # Términos trigonométricos (importantes para la cinemática)
    cos_θ = np.cos(θ)
    sin_θ = np.sin(θ)
    
    # Comandos escalados
    s_v_cmd = sigmoidal(v_cmd)
    s_ω_cmd = sigmoidal(ω_cmd)
    
    # ========== CARACTERÍSTICAS ESPECÍFICAS POR NEURONA ==========
    
    if neuron_index == 0:  # Neurona para x (posición horizontal)
        # dx/dt = v*cos(θ)
        return np.array([
            s_v,                       # Velocidad lineal
            # cos_θ,                     # Componente direccional
            # s_v * cos_θ,              # Término cinemático principal
            s_θ,                       # Orientación
            s_v * s_θ,                # Interacción velocidad-orientación
            s_v**2,                    # Término cuadrático
            # 1.0                        # Bias
        ])
    
    elif neuron_index == 1:  # Neurona para y (posición vertical)
        # dy/dt = v*sin(θ)
        return np.array([
            # s_v,                       # Velocidad lineal
            # sin_θ,                     # Componente direccional
            # s_v * sin_θ,              # Término cinemático principal
            s_θ,                       # Orientación
            s_v * s_θ,                # Interacción velocidad-orientación
            s_v**2,                    # Término cuadrático
            # 1.0                        # Bias
        ])
    
    elif neuron_index == 2:  # Neurona para θ (orientación)
        # dθ/dt = ω
        return np.array([
            s_ω,                       # Velocidad angular (término principal)
            s_ω**2,                    # Término cuadrático
            # s_ω**3,                    # Término cúbico (no linealidad)
            s_v * s_ω,                # Acoplamiento con velocidad lineal
            s_θ,                       # Orientación actual
            # s_ω_cmd * 0.2,            # Comando de velocidad angular
            # 1.0                        # Bias
        ])
    
    elif neuron_index == 3:  # Neurona para v (velocidad lineal)
        # dv/dt = (v_cmd - v)/tau - friction
        return np.array([
            s_v,                       # Estado actual
            s_v**2,                    # Fricción cuadrática
            s_v**3,                    # Fricción cúbica
            # v_cmd * 0.1,              # Comando (lineal, no saturado)
            s_v_cmd,                   # Comando (sigmoidal)
            # s_v * s_v_cmd,            # Interacción estado-comando
            s_ω,                       # Acoplamiento con velocidad angular
            # s_v * np.abs(v),          # Término de fricción absoluta
            # 1.0                        # Bias
        ])
    
    elif neuron_index == 4:  # Neurona para ω (velocidad angular)
        # dω/dt = (ω_cmd - ω)/tau - friction
        return np.array([
            s_ω,                       # Estado actual
            # s_ω**2,                    # Fricción cuadrática
            s_ω**3,                    # Fricción cúbica
            # ω_cmd * 0.1,              # Comando (lineal, no saturado)
            # s_ω_cmd,                   # Comando (sigmoidal)
            s_ω * s_ω_cmd,            # Interacción estado-comando
            s_v,                       # Acoplamiento con velocidad lineal
            # s_ω * np.abs(ω),          # Término de fricción absoluta
            # 1.0                        # Bias
        ])
    
    else:
        raise ValueError(f"Índice de neurona inválido: {neuron_index}")

# Función auxiliar para obtener el tamaño de características de cada neurona
def get_z_size(neuron_index):
    """Retorna el número de características para una neurona dada."""
    if neuron_index == 0:  # x
        return 4
    elif neuron_index == 1:  # y
        return 3
    elif neuron_index == 2:  # θ
        return 4
    elif neuron_index == 3:  # v
        return 5
    elif neuron_index == 4:  # ω
        return 4
    else:
        raise ValueError(f"Índice de neurona inválido: {neuron_index}")

# ============================================================
# 3) Trainers (EKF, UKF, PF) - ADAPTADOS PARA MÚLTIPLES TAMAÑOS
# ============================================================

class Generic_RHONN_Trainer:
    """ Base class to handle the loop logic easily """
    def get_prediction(self, weights, x_k, u_k, neuron_idx):
        z = construct_z_vector(x_k, u_k, neuron_idx)
        return np.dot(weights, z)

class EKF_Trainer(Generic_RHONN_Trainer):
    def __init__(self, n_neurons, eta=1.0, P0=1.0, Q=1e-3, R=1e-5):
        self.n_neurons = n_neurons
        self.weights = [np.random.randn(get_z_size(i))*0.1 for i in range(n_neurons)]
        self.P = [np.eye(get_z_size(i))*P0 for i in range(n_neurons)]
        self.Q_matrices = [np.eye(get_z_size(i))*Q for i in range(n_neurons)]
        self.R = R
        self.eta = eta

    def update(self, x_kp1, x_k, u_k):
        for i in range(self.n_neurons):
            z = construct_z_vector(x_k, u_k, i)
            H = z.reshape(-1, 1)
            
            # Predict P
            P_pred = self.P[i] + self.Q_matrices[i]
            
            # Kalman Gain
            S = self.R + (H.T @ P_pred @ H)[0,0]
            K = (P_pred @ H).flatten() / S
            
            # Error
            y_pred = np.dot(self.weights[i], z)
            err = x_kp1[i] - y_pred
            
            # Update Weights
            self.weights[i] += self.eta * K * err
            
            # Update Covariance (Joseph form)
            I_KH = np.eye(len(z)) - np.outer(K, z)
            self.P[i] = I_KH @ P_pred @ I_KH.T + np.outer(K, K)*self.R

class UKF_Trainer(Generic_RHONN_Trainer):
    def __init__(self, n_neurons, eta=1.0, alpha=1e-2):
        self.n_neurons = n_neurons
        self.weights = [np.random.randn(get_z_size(i))*0.1 for i in range(n_neurons)]
        self.P = [np.eye(get_z_size(i)) for i in range(n_neurons)]
        self.Q_matrices = [np.eye(get_z_size(i))*1e-4 for i in range(n_neurons)]
        self.R = 1e-5
        self.eta = eta
        self.alpha = alpha

    def update(self, x_kp1, x_k, u_k):
        for i in range(self.n_neurons):
            z = construct_z_vector(x_k, u_k, i)
            n = get_z_size(i)
            
            # Sigma params
            lambda_ = self.alpha**2 * n - n
            Wm = np.full(2*n+1, 1/(2*(n+lambda_)))
            Wc = np.copy(Wm)
            Wm[0] = lambda_/(n+lambda_)
            Wc[0] = Wm[0] + (3 - self.alpha**2)
            
            # Generate Sigmas
            try:
                L = np.linalg.cholesky((n + lambda_) * self.P[i])
            except:
                L = np.eye(n) * 0.1
                
            sigmas = np.zeros((2*n+1, n))
            sigmas[0] = self.weights[i]
            for k in range(n):
                sigmas[k+1] = self.weights[i] + L[:,k]
                sigmas[n+k+1] = self.weights[i] - L[:,k]
            
            # Transform
            Y_sigmas = np.dot(sigmas, z)
            y_mean = np.sum(Wm * Y_sigmas)
            
            # Covariances
            Py = np.sum(Wc * (Y_sigmas - y_mean)**2) + self.R
            Pxy = np.zeros(n)
            for k in range(2*n+1):
                Pxy += Wc[k] * (sigmas[k] - self.weights[i]) * (Y_sigmas[k] - y_mean)
                
            # Update
            K = Pxy / Py
            err = x_kp1[i] - y_mean
            self.weights[i] += self.eta * K * err
            self.P[i] -= np.outer(K, K) * Py
            
            # Regularize P
            self.P[i] += np.eye(n)*1e-6

class PF_Trainer(Generic_RHONN_Trainer):
    def __init__(self, n_neurons, n_particles=800):
        self.n_neurons = n_neurons
        self.n_particles = n_particles
        self.particles = [np.random.randn(n_particles, get_z_size(i))*0.2 
                         for i in range(n_neurons)]
        self.weights_pf = [np.ones(n_particles)/n_particles for _ in range(n_neurons)]
        self.Q_std = 0.2
        self.R_std = 0.1

    def update(self, x_kp1, x_k, u_k):
        for i in range(self.n_neurons):
            z = construct_z_vector(x_k, u_k, i)
            n_weights = get_z_size(i)
            
            # 1. Drift
            self.particles[i] += np.random.randn(self.n_particles, n_weights) * self.Q_std
            
            # 2. Weight
            preds = self.particles[i] @ z
            err = x_kp1[i] - preds
            likelihood = np.exp(-0.5 * (err/self.R_std)**2)
            self.weights_pf[i] *= (likelihood + 1e-300)
            self.weights_pf[i] /= np.sum(self.weights_pf[i])
            
            # 3. Resample
            eff_N = 1.0 / np.sum(self.weights_pf[i]**2)
            if eff_N < self.n_particles/2:
                indices = np.random.choice(self.n_particles, self.n_particles, 
                                         p=self.weights_pf[i])
                self.particles[i] = self.particles[i][indices]
                self.weights_pf[i].fill(1.0/self.n_particles)
                
    def get_estimates(self):
        return [np.average(self.particles[i], axis=0, weights=self.weights_pf[i]) 
                for i in range(self.n_neurons)]

# ============================================================
# 4) Parameter Optimization (Optional)
# ============================================================
def run_simulation_with_params(params, filter_type='EKF', n_steps=500, verbose=False):
    """
    Run a simulation with given parameters and return MSE.
    Used for parameter optimization.
    """
    dt = 0.01
    n_states = 5
    
    # Unpack parameters based on filter type
    try:
        if filter_type == 'EKF':
            eta = params[0]
            P0 = 10 ** params[1]
            Q = 10 ** params[2]
            R = 10 ** params[3]
            trainer = EKF_Trainer(n_states, eta=eta, P0=P0, Q=Q, R=R)
            
        elif filter_type == 'UKF':
            eta = params[0]
            alpha = 10 ** params[1]
            trainer = UKF_Trainer(n_states, eta=eta, alpha=alpha)
            
        elif filter_type == 'PF':
            n_particles = int(params[0] * 100)
            trainer = PF_Trainer(n_states, n_particles=n_particles)
            
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        # Initialize arrays
        x_true = np.zeros((n_steps, 5))
        x_est = np.zeros((n_steps, 5))
        x_true[0] = [0.0, 0.0, 0.0, 0.0, 0.0]
        x_est[0] = x_true[0]
        
        # Generate excitation input
        t = np.linspace(0, (n_steps-1)*dt, n_steps)
        u_hist = np.zeros((n_steps, 2))
        for k in range(n_steps):
            v_cmd = 1.0 + 0.5 * np.sin(0.5 * t[k])
            ω_cmd = 0.8 * np.sin(1.0 * t[k])
            if 100 < k < 120:
                v_cmd = 2.0
                ω_cmd = 1.5
            elif 250 < k < 270:
                v_cmd = 0.5
                ω_cmd = -1.2
            u_hist[k] = [v_cmd, ω_cmd]
        
        # Run simulation
        for k in range(n_steps - 1):
            x_true[k+1] = plant(x_true[k], u_hist[k], dt)
            trainer.update(x_true[k+1], x_true[k], u_hist[k])
            
            if filter_type == 'PF':
                w_est = trainer.get_estimates()
                for i in range(5):
                    z = construct_z_vector(x_true[k], u_hist[k], i)
                    x_est[k+1, i] = np.dot(w_est[i], z)
            else:
                for i in range(5):
                    z = construct_z_vector(x_true[k], u_hist[k], i)
                    x_est[k+1, i] = np.dot(trainer.weights[i], z)
        
        # Calculate total MSE
        total_mse = 0.0
        for i in range(5):
            total_mse += np.mean((x_true[:, i] - x_est[:, i])**2)
        
        if verbose:
            print(f"  MSE: {total_mse:.6e} | Params: {params}")
        
        return total_mse
    
    except Exception as e:
        if verbose:
            print(f"  Error with params {params}: {e}")
        return 1e10

def optimize_filter_parameters(filter_type='EKF', method='differential_evolution', 
                               n_steps=500, maxiter=50, verbose=True):
    """
    Optimize filter parameters using scipy optimization.
    
    Parameters:
    -----------
    filter_type : str
        Type of filter to optimize: 'EKF', 'UKF', or 'PF'
    method : str
        Optimization method: 'differential_evolution' or 'nelder-mead'
    n_steps : int
        Number of simulation steps for evaluation
    maxiter : int
        Maximum iterations for optimization
    verbose : bool
        Print optimization progress
    
    Returns:
    --------
    dict : Optimized parameters and final MSE
    """
    print(f"\n{'='*70}")
    print(f"🔧 OPTIMIZING {filter_type} PARAMETERS (Robot Diferencial)")
    print(f"{'='*70}")
    print(f"Method: {method}")
    print(f"Simulation length: {n_steps} steps")
    print(f"Max iterations: {maxiter}\n")
    
    # Define parameter bounds and initial guesses
    if filter_type == 'EKF':
        bounds = [(0.1, 2.0), (-2, 1), (-6, -2), (-4, -1)]  # eta, log10(P0), log10(Q), log10(R)
        x0 = [1.0, 0.0, -4, -2]
        param_names = ['eta', 'log10(P0)', 'log10(Q)', 'log10(R)']
        
    elif filter_type == 'UKF':
        bounds = [(0.1, 2.0), (-3, 0)]  # eta, log10(alpha)
        x0 = [0.9, -2]
        param_names = ['eta', 'log10(alpha)']
        
    elif filter_type == 'PF':
        bounds = [(5, 15)]  # n_particles_ratio (multiplied by 100)
        x0 = [8]
        param_names = ['n_particles_ratio']
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Objective function
    objective = lambda params: run_simulation_with_params(params, filter_type, n_steps, verbose=False)
    
    # Optimize
    if method == 'differential_evolution':
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=maxiter,
            popsize=15,
            seed=42,
            atol=1e-6,
            tol=1e-6,
            workers=1,
            updating='deferred',
            disp=verbose
        )
    elif method == 'nelder-mead':
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'disp': verbose, 'xatol': 1e-6, 'fatol': 1e-6}
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")
    
    # Extract and display results
    optimized_params = result.x
    final_mse = result.fun
    
    print(f"\n{'='*70}")
    print(f"✅ OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Final MSE: {final_mse:.6e}")
    print(f"\nOptimized Parameters:")
    
    # Display optimized parameters
    for name, value in zip(param_names, optimized_params):
        print(f"  {name}: {value:.6f}")
    
    # Return dictionary with optimized parameters
    if filter_type == 'EKF':
        return {
            'eta': optimized_params[0],
            'P0': 10 ** optimized_params[1],
            'Q': 10 ** optimized_params[2],
            'R': 10 ** optimized_params[3],
            'mse': final_mse
        }
    elif filter_type == 'UKF':
        return {
            'eta': optimized_params[0],
            'alpha': 10 ** optimized_params[1],
            'mse': final_mse
        }
    elif filter_type == 'PF':
        return {
            'n_particles': int(optimized_params[0] * 100),
            'mse': final_mse
        }

# ============================================================
# 5) Simulation Main Loop - UPDATED FOR PARALLEL CONFIGURATION
# ============================================================
if __name__ == "__main__":
    # ============================================================
    # Main Simulation - PARALLEL CONFIGURATION
    # ============================================================
    n_steps = 1500
    dt = 0.01
    t = np.linspace(0, (n_steps-1)*dt, n_steps)
    
    n_states = 5  # [x, y, θ, v, ω]
    
    # Measurement noise parameters (simulating real sensors)
    position_noise_std = 0.01      # 1 cm position error
    angle_noise_std = 0.005        # 0.5° angle error
    velocity_noise_std = 0.02      # 2 cm/s velocity error
    omega_noise_std = 0.03         # 3°/s angular velocity error
    
    # Initialize Trainers with more realistic parameters
    ekf = EKF_Trainer(n_states, eta=0.5, P0=10.0, Q=1e-4, R=1e-3)  # Increased R for measurement noise
    ukf = UKF_Trainer(n_states, eta=0.6, alpha=1e-2, Q=1e-4, R=1e-5)  # Smaller alpha for better stability
    pf = PF_Trainer(n_states, n_particles=1000)  # Reduced particles for speed
    
    # Arrays for states
    x_true = np.zeros((n_steps, 5))
    x_est_ekf = np.zeros((n_steps, 5))
    x_est_ukf = np.zeros((n_steps, 5))
    x_est_pf = np.zeros((n_steps, 5))
    
    # Arrays for noisy measurements
    y_measured = np.zeros((n_steps, 5))
    
    # Initial Conditions (robot at origin, facing right)
    x_true[0] = [0.0, 0.0, 0.0, 0.0, 0.0]
    x_est_ekf[0] = x_true[0]
    x_est_ukf[0] = x_true[0]
    x_est_pf[0] = x_true[0]
    
    # Add noise to initial measurement
    initial_noise = np.array([
        np.random.laplace(0, position_noise_std),
        np.random.normal(0, position_noise_std),
        np.random.laplace(0, angle_noise_std),
        np.random.normal(0, velocity_noise_std),
        np.random.laplace(0, omega_noise_std)
    ])
    y_measured[0] = x_true[0] + initial_noise
    
    # Excitation Input
    u_hist = np.zeros((n_steps, 2))
    for k in range(n_steps):
        v_cmd = 1.0 + 0.5 * np.sin(0.5 * t[k])
        ω_cmd = 0.8 * np.sin(1.0 * t[k])
        
        # Add some step changes for better excitation
        if 300 < k < 320:
            v_cmd = 2.0
            ω_cmd = 1.5
        elif 700 < k < 720:
            v_cmd = 0.5
            ω_cmd = -1.2
        elif 1100 < k < 1120:
            v_cmd = -0.8
            ω_cmd = 0.0
            
        u_hist[k] = [v_cmd, ω_cmd]

    print("Simulating Differential Drive Robot with PARALLEL CONFIGURATION...")
    print("\nEstructura de características por neurona:")
    state_names = ['x (pos)', 'y (pos)', 'θ (orient)', 'v (lin vel)', 'ω (ang vel)']
    for i in range(n_states):
        print(f"  Neurona {i} ({state_names[i]}): {get_z_size(i)} características")
    
    print("\nMeasurement Noise Parameters:")
    print(f"  Position (x,y): ±{position_noise_std:.3f} m")
    print(f"  Angle (θ): ±{angle_noise_std:.3f} rad")
    print(f"  Linear velocity (v): ±{velocity_noise_std:.3f} m/s")
    print(f"  Angular velocity (ω): ±{omega_noise_std:.3f} rad/s")
    
    # Main simulation loop with PARALLEL CONFIGURATION
    for k in range(n_steps - 1):
        # 1. Generate true next state
        x_true[k+1] = plant(x_true[k], u_hist[k], dt)
        
        # 2. Create noisy measurement (simulating real sensors)
        measurement_noise = np.array([
            np.random.normal(0, position_noise_std),
            np.random.normal(0, position_noise_std),
            np.random.normal(0, angle_noise_std),
            np.random.normal(0, velocity_noise_std),
            np.random.normal(0, omega_noise_std)
        ])
        y_measured[k+1] = x_true[k+1] + measurement_noise
        
        # 3. Update filters with NOISY MEASUREMENTS (not true states)
        # Each filter uses its OWN previous estimate as input
        
        # --- EKF Update & Predict ---
        ekf.update(y_measured[k+1], x_est_ekf[k], u_hist[k])  # Update with measurement
        # Predict next state using EKF's OWN estimate
        for i in range(5):
            z_ekf = construct_z_vector(y_measured[k], u_hist[k], i)  # PARALLEL: use estimated state
            x_est_ekf[k+1, i] = np.dot(ekf.weights[i], z_ekf)
        
        # --- UKF Update & Predict ---
        ukf.update(y_measured[k+1], x_est_ukf[k], u_hist[k])  # Update with measurement
        # Predict next state using UKF's OWN estimate
        for i in range(5):
            z_ukf = construct_z_vector(y_measured[k], u_hist[k], i)  # PARALLEL: use estimated state
            x_est_ukf[k+1, i] = np.dot(ukf.weights[i], z_ukf)
        
        # --- PF Update & Predict ---
        pf.update(y_measured[k+1], x_est_pf[k], u_hist[k])  # Update with measurement
        w_pf = pf.get_estimates()
        # Predict next state using PF's OWN estimate
        for i in range(5):
            z_pf = construct_z_vector(y_measured[k], u_hist[k], i)  # PARALLEL: use estimated state
            x_est_pf[k+1, i] = np.dot(w_pf[i], z_pf)
        
        # Progress reporting
        if k % 300 == 0 and k > 0:
            print(f"Step {k}/{n_steps-1}")
            # Show current errors
            err_ekf = np.linalg.norm(x_true[k] - x_est_ekf[k])
            err_ukf = np.linalg.norm(x_true[k] - x_est_ukf[k])
            err_pf = np.linalg.norm(x_true[k] - x_est_pf[k])
            print(f"  Current errors - EKF: {err_ekf:.4f}, UKF: {err_ukf:.4f}, PF: {err_pf:.4f}")

    # ============================================================
    # 6) Calculate Statistics - UPDATED FOR FAIR COMPARISON
    # ============================================================
    
    print("\n" + "="*70)
    print("📊 RESULTS - PARALLEL CONFIGURATION WITH MEASUREMENT NOISE")
    print("="*70)
    
    # Calculate MSE against TRUE states (not measurements)
    mse_x_ekf = np.mean((x_true[:, 0] - x_est_ekf[:, 0])**2)
    mse_y_ekf = np.mean((x_true[:, 1] - x_est_ekf[:, 1])**2)
    mse_θ_ekf = np.mean((x_true[:, 2] - x_est_ekf[:, 2])**2)
    mse_v_ekf = np.mean((x_true[:, 3] - x_est_ekf[:, 3])**2)
    mse_ω_ekf = np.mean((x_true[:, 4] - x_est_ekf[:, 4])**2)
    
    mse_x_ukf = np.mean((x_true[:, 0] - x_est_ukf[:, 0])**2)
    mse_y_ukf = np.mean((x_true[:, 1] - x_est_ukf[:, 1])**2)
    mse_θ_ukf = np.mean((x_true[:, 2] - x_est_ukf[:, 2])**2)
    mse_v_ukf = np.mean((x_true[:, 3] - x_est_ukf[:, 3])**2)
    mse_ω_ukf = np.mean((x_true[:, 4] - x_est_ukf[:, 4])**2)
    
    mse_x_pf = np.mean((x_true[:, 0] - x_est_pf[:, 0])**2)
    mse_y_pf = np.mean((x_true[:, 1] - x_est_pf[:, 1])**2)
    mse_θ_pf = np.mean((x_true[:, 2] - x_est_pf[:, 2])**2)
    mse_v_pf = np.mean((x_true[:, 3] - x_est_pf[:, 3])**2)
    mse_ω_pf = np.mean((x_true[:, 4] - x_est_pf[:, 4])**2)
    
    # Total MSE for comparison
    mse_total_ekf = mse_x_ekf + mse_y_ekf + mse_θ_ekf + mse_v_ekf + mse_ω_ekf
    mse_total_ukf = mse_x_ukf + mse_y_ukf + mse_θ_ukf + mse_v_ukf + mse_ω_ukf
    mse_total_pf = mse_x_pf + mse_y_pf + mse_θ_pf + mse_v_pf + mse_ω_pf
    
    # Report
    print("\n--- Comparación de Desempeño (MSE contra estado REAL) ---")
    print(f"EKF MSE total: {mse_total_ekf:.6f}")
    print(f"  x: {mse_x_ekf:.6f} | y: {mse_y_ekf:.6f} | θ: {mse_θ_ekf:.6f} | v: {mse_v_ekf:.6f} | ω: {mse_ω_ekf:.6f}")
    print(f"UKF MSE total: {mse_total_ukf:.6f}")
    print(f"  x: {mse_x_ukf:.6f} | y: {mse_y_ukf:.6f} | θ: {mse_θ_ukf:.6f} | v: {mse_v_ukf:.6f} | ω: {mse_ω_ukf:.6f}")
    print(f"PF  MSE total: {mse_total_pf:.6f}")
    print(f"  x: {mse_x_pf:.6f} | y: {mse_y_pf:.6f} | θ: {mse_θ_pf:.6f} | v: {mse_v_pf:.6f} | ω: {mse_ω_pf:.6f}")
    
    print("\n" + "="*70)
    print("🏆 MEJOR FILTRO: ", end="")
    mse_dict = {'EKF-RHONN': mse_total_ekf, 'UKF-RHONN': mse_total_ukf, 'PF-RHONN': mse_total_pf}
    best_filter = min(mse_dict, key=mse_dict.get)
    print(f"{best_filter} (MSE total: {mse_dict[best_filter]:.6f})")
    print("="*70)
    
    # ============================================================
    # 7) Visualization - Updated for realistic results
    # ============================================================
    
    print("\nGenerando visualizaciones...")
    
    # Configuración de formato para tesis
    thesis_config = {
        'font_family': 'Computer Modern, serif',
        'font_size': 14,
        'title_font_size': 16,
        'legend_font_size': 12,
        'line_width_true': 2.5,
        'line_width_est': 2.0,
        'line_width_meas': 1.0,
        'plot_width': 1000,
        'plot_height': 500,
        'grid_color': 'rgba(200, 200, 200, 0.3)',
        'grid_width': 0.5
    }
    
    # --- Trayectoria 2D (Vista superior) ---
    fig_traj = go.Figure()
    
    # Trayectoria real
    fig_traj.add_trace(go.Scatter(
        x=x_true[:, 0], y=x_true[:, 1],
        mode='lines',
        name='Trayectoria Real',
        line=dict(color='#000000', width=thesis_config['line_width_true']),
    ))
    
    # Medidas ruidosas (puntos dispersos)
    fig_traj.add_trace(go.Scatter(
        x=y_measured[:, 0], y=y_measured[:, 1],
        mode='markers',
        name='Medidas Ruidosas',
        marker=dict(size=2, color='rgba(100, 100, 100, 0.3)', symbol='circle'),
    ))
    
    # Trayectorias estimadas
    fig_traj.add_trace(go.Scatter(
        x=x_est_ekf[:, 0], y=x_est_ekf[:, 1],
        mode='lines',
        name='EKF-RHONN',
        line=dict(color='#1f77b4', width=thesis_config['line_width_est'], dash='dash'),
    ))
    
    fig_traj.add_trace(go.Scatter(
        x=x_est_ukf[:, 0], y=x_est_ukf[:, 1],
        mode='lines',
        name='UKF-RHONN',
        line=dict(color='#2ca02c', width=thesis_config['line_width_est'], dash='dot'),
    ))
    
    fig_traj.add_trace(go.Scatter(
        x=x_est_pf[:, 0], y=x_est_pf[:, 1],
        mode='lines',
        name='PF-RHONN',
        line=dict(color='#d62728', width=thesis_config['line_width_est'], dash='dashdot'),
    ))
    
    # Puntos de inicio y final
    fig_traj.add_trace(go.Scatter(
        x=[x_true[0, 0]], y=[x_true[0, 1]],
        mode='markers',
        name='Inicio',
        marker=dict(size=15, color='green', symbol='circle'),
    ))
    
    fig_traj.add_trace(go.Scatter(
        x=[x_true[-1, 0]], y=[x_true[-1, 1]],
        mode='markers',
        name='Final',
        marker=dict(size=15, color='red', symbol='square'),
    ))
    
    fig_traj.update_layout(
        title={
            'text': 'Trayectoria del Robot Diferencial - Configuración Paralela',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': thesis_config['title_font_size'], 'family': thesis_config['font_family']}
        },
        xaxis_title='Posición x (m)',
        yaxis_title='Posición y (m)',
        xaxis=dict(
            showgrid=True,
            gridcolor=thesis_config['grid_color'],
            gridwidth=thesis_config['grid_width'],
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1.5,
            ticklen=5,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=thesis_config['grid_color'],
            gridwidth=thesis_config['grid_width'],
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1.5,
            ticklen=5
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=thesis_config['legend_font_size'], family=thesis_config['font_family'])
        ),
        font=dict(size=thesis_config['font_size'], family=thesis_config['font_family']),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=thesis_config['plot_width'],
        height=thesis_config['plot_width'],  # Cuadrado
        margin=dict(l=80, r=40, t=80, b=60)
    )
    
    # Guardar figura
    fig_traj.write_image("04_trayectoria_xy_serieparalelo.png", width=thesis_config['plot_width'], height=thesis_config['plot_width'])
    fig_traj.write_html("04_trayectoria_xy_serieparalelo.html")
    fig_traj.show()
    
    # --- Gráficas por Estado (con medidas ruidosas) ---
    states_info = [
        {'idx': 0, 'var': 'x', 'desc': 'Posición Horizontal', 'y_label': 'Posición x (m)'},
        {'idx': 1, 'var': 'y', 'desc': 'Posición Vertical', 'y_label': 'Posición y (m)'},
        {'idx': 2, 'var': 'θ', 'desc': 'Orientación', 'y_label': 'Ángulo θ (rad)'},
        {'idx': 3, 'var': 'v', 'desc': 'Velocidad Lineal', 'y_label': 'Velocidad v (m/s)'},
        {'idx': 4, 'var': 'ω', 'desc': 'Velocidad Angular', 'y_label': 'Velocidad ω (rad/s)'}
    ]
    
    for state_info in states_info:
        i = state_info['idx']
        
        fig = go.Figure()
        
        # EKF estimate
        fig.add_trace(go.Scatter(
            x=t, y=x_est_ekf[:, i],
            mode='lines',
            name='EKF-RHONN',
            line=dict(color='#1f77b4', width=thesis_config['line_width_est']),
            showlegend=True
        ))
        
        # UKF estimate
        fig.add_trace(go.Scatter(
            x=t, y=x_est_ukf[:, i],
            mode='lines',
            name='UKF-RHONN',
            line=dict(color='#2ca02c', width=thesis_config['line_width_est']),
            showlegend=True
        ))
        
        # PF estimate
        fig.add_trace(go.Scatter(
            x=t, y=x_est_pf[:, i],
            mode='lines',
            name='PF-RHONN',
            line=dict(color='#d62728', width=thesis_config['line_width_est']),
            showlegend=True
        ))
        
        fig.update_layout(
            title={
                'text': f'Estado {state_info["var"]}: {state_info["desc"]} - Configuración Serie-Paralela',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': thesis_config['title_font_size'], 'family': thesis_config['font_family']}
            },
            xaxis_title='Tiempo (s)',
            yaxis_title=state_info['y_label'],
            xaxis=dict(
                showgrid=True,
                gridcolor=thesis_config['grid_color'],
                gridwidth=thesis_config['grid_width'],
                showline=True,
                linewidth=1.5,
                linecolor='black',
                mirror=True,
                ticks='outside',
                tickwidth=1.5,
                ticklen=5
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=thesis_config['grid_color'],
                gridwidth=thesis_config['grid_width'],
                showline=True,
                linewidth=1.5,
                linecolor='black',
                mirror=True,
                ticks='outside',
                tickwidth=1.5,
                ticklen=5
            ),
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=thesis_config['legend_font_size'], family=thesis_config['font_family'])
            ),
            font=dict(size=thesis_config['font_size'], family=thesis_config['font_family']),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=thesis_config['plot_width'],
            height=thesis_config['plot_height'],
            margin=dict(l=80, r=40, t=80, b=60)
        )
        
        # Guardar figura por estado
        filename = f"05_estado_{state_info['var']}_serieparalelo.png"
        fig.write_image(filename, width=thesis_config['plot_width'], height=thesis_config['plot_height'])
        fig.write_html(f"05_estado_{state_info['var']}_serieparalelo.html")
        fig.show()
    
    # --- Gráfica de errores acumulados ---
    fig_errors = go.Figure()
    
    # Calculate cumulative errors over time
    cumulative_error_ekf = np.zeros(n_steps)
    cumulative_error_ukf = np.zeros(n_steps)
    cumulative_error_pf = np.zeros(n_steps)
    
    for k in range(1, n_steps):
        error_ekf = np.linalg.norm(x_true[k] - x_est_ekf[k])
        error_ukf = np.linalg.norm(x_true[k] - x_est_ukf[k])
        error_pf = np.linalg.norm(x_true[k] - x_est_pf[k])
        
        cumulative_error_ekf[k] = cumulative_error_ekf[k-1] + error_ekf
        cumulative_error_ukf[k] = cumulative_error_ukf[k-1] + error_ukf
        cumulative_error_pf[k] = cumulative_error_pf[k-1] + error_pf
    
    fig_errors.add_trace(go.Scatter(
        x=t, y=cumulative_error_ekf,
        mode='lines',
        name='EKF-RHONN',
        line=dict(color='#1f77b4', width=2),
    ))
    
    fig_errors.add_trace(go.Scatter(
        x=t, y=cumulative_error_ukf,
        mode='lines',
        name='UKF-RHONN',
        line=dict(color='#2ca02c', width=2),
    ))
    
    fig_errors.add_trace(go.Scatter(
        x=t, y=cumulative_error_pf,
        mode='lines',
        name='PF-RHONN',
        line=dict(color='#d62728', width=2),
    ))
    
    fig_errors.update_layout(
        title={
            'text': 'Error Acumulado en el Tiempo - Norma Euclidiana',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': thesis_config['title_font_size'], 'family': thesis_config['font_family']}
        },
        xaxis_title='Tiempo (s)',
        yaxis_title='Error Acumulado',
        xaxis=dict(
            showgrid=True,
            gridcolor=thesis_config['grid_color'],
            gridwidth=thesis_config['grid_width'],
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1.5,
            ticklen=5
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor=thesis_config['grid_color'],
            gridwidth=thesis_config['grid_width'],
            showline=True,
            linewidth=1.5,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=1.5,
            ticklen=5
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=thesis_config['legend_font_size'], family=thesis_config['font_family'])
        ),
        font=dict(size=thesis_config['font_size'], family=thesis_config['font_family']),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=thesis_config['plot_width'],
        height=thesis_config['plot_height'],
        margin=dict(l=80, r=40, t=80, b=60)
    )
    
    # Guardar figura
    fig_errors.write_image("06_error_acumulado_serieparalelo.png", width=thesis_config['plot_width'], height=thesis_config['plot_height'])
    fig_errors.write_html("06_error_acumulado_serieparalelo.html")
    fig_errors.show()
    
    print("\n✅ Visualización completa con configuración paralela.")

    # ============================================================
    # 8) Additional Analysis
    # ============================================================
    
    print("\n" + "="*70)
    print("📈 ANÁLISIS ADICIONAL")
    print("="*70)
    
    # Analyze convergence
    convergence_window = 100  # Look at last 100 steps
    if n_steps > convergence_window:
        final_errors = {
            'EKF': np.mean((x_true[-convergence_window:, :] - x_est_ekf[-convergence_window:, :])**2, axis=0),
            'UKF': np.mean((x_true[-convergence_window:, :] - x_est_ukf[-convergence_window:, :])**2, axis=0),
            'PF': np.mean((x_true[-convergence_window:, :] - x_est_pf[-convergence_window:, :])**2, axis=0)
        }
        
        print(f"\nError promedio en los últimos {convergence_window} pasos:")
        for filter_name, errors in final_errors.items():
            print(f"\n{filter_name}:")
            print(f"  x: {errors[0]:.6f}, y: {errors[1]:.6f}, θ: {errors[2]:.6f}")
            print(f"  v: {errors[3]:.6f}, ω: {errors[4]:.6f}")

 