# Neural Identifier Training - 2-DOF Planar Manipulator
# Versión Mejorada con Estructura RHONN Personalizable
# Methods: EKF (Alanis), UKF (Rios), PF Optimizado

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Callable, Dict
import time

np.random.seed(42)

# ============================================================
# 1) True nonlinear system (2-DOF Robot Arm)
# ============================================================
def plant_dynamics(state, u):
    """
    Dynamics of a 2-link planar robot arm.
    state = [q1, q2, dq1, dq2] (Angles and Velocities)
    u     = [tau1, tau2]       (Torques)
    """
    # Robot Parameters
    m1, m2 = 1.0, 1.0  # Mass (kg)
    l1, l2 = 1.0, 1.0  # Lengths (m)
    g = 9.81
    
    q1, q2, dq1, dq2 = state
    tau1, tau2 = u

    # --- Mass Matrix M(q) ---
    c2 = np.cos(q2)
    s2 = np.sin(q2)
    
    M11 = (m1 + m2) * l1**2 + m2 * l2**2 + 2 * m2 * l1 * l2 * c2
    M12 = m2 * l2**2 + m2 * l1 * l2 * c2
    M21 = M12
    M22 = m2 * l2**2
    M = np.array([[M11, M12], [M21, M22]])

    # --- Coriolis/Centrifugal Matrix C(q, dq) ---
    h = -m2 * l1 * l2 * s2
    C11 = h * dq2
    C12 = h * (dq1 + dq2)
    C21 = -h * dq1
    C22 = 0.0
    C = np.array([[C11, C12], [C21, C22]])

    # --- Gravity Vector G(q) ---
    s1 = np.sin(q1)
    s12 = np.sin(q1 + q2)
    G1 = (m1 + m2) * g * l1 * s1 + m2 * g * l2 * s12
    G2 = m2 * g * l2 * s12
    G = np.array([G1, G2])

    # --- Equation of Motion: M*ddq + C*dq + G = tau ---
    damping = 0.5 * np.array([dq1, dq2])
    torque_vector = np.array([tau1, tau2])
    
    rhs = torque_vector - (C @ np.array([dq1, dq2])) - G - damping
    
    # Solve for accelerations
    ddq = np.linalg.solve(M, rhs)
    
    return np.concatenate(([dq1, dq2], ddq))

def plant(x_k, u_k, dt=0.01, process_noise_std=1e-4):
    """Euler integration step with process noise."""
    x_dot = plant_dynamics(x_k, u_k)
    x_kp1 = x_k + dt * x_dot
    
    # Add small process noise
    noise = np.random.randn(4) * process_noise_std
    return x_kp1 + noise

# ============================================================
# 2) Estructura RHONN Personalizable
# ============================================================
class RHONNStructure:
    """
    Estructura personalizable de RHONN para manipulador 2-DOF.
    Permite definir interacciones de alto orden de manera flexible.
    """
    
    def __init__(self, n_states: int = 4):
        self.n_states = n_states
        self.interactions = [[] for _ in range(n_states)]
        self.n_weights = [0] * n_states
    
    def set_interactions(self, state_idx: int, interaction_list: List[str]):
        """Define todas las interacciones para un estado."""
        self.interactions[state_idx] = interaction_list.copy()
        self.n_weights[state_idx] = len(interaction_list)
    
    def compute_S(self, x: np.ndarray, u: np.ndarray) -> List[np.ndarray]:
        """
        Calcula los vectores de interacciones S_i para cada estado.
        
        Args:
            x: Vector de estados [q1, q2, dq1, dq2]
            u: Vector de entradas [tau1, tau2]
        
        Returns:
            Lista de vectores S_i, uno por cada estado
        """
        S_list = []
        
        for i in range(self.n_states):
            S_i = np.zeros(self.n_weights[i])
            
            for j, interaction in enumerate(self.interactions[i]):
                # Crear diccionario de variables
                local_vars = {
                    'q1': x[0], 'q2': x[1], 'dq1': x[2], 'dq2': x[3],
                    'tau1': u[0], 'tau2': u[1],
                    'sin': np.sin, 'cos': np.cos, 'tanh': np.tanh,
                    'exp': np.exp, 'abs': np.abs
                }
                
                try:
                    S_i[j] = eval(interaction, {"__builtins__": {}}, local_vars)
                except:
                    S_i[j] = 0.0
            
            S_list.append(S_i)
        
        return S_list
    
    def get_total_weights(self) -> int:
        """Retorna el número total de pesos en la RHONN."""
        return sum(self.n_weights)
    
    def print_structure(self):
        """Imprime la estructura de la RHONN."""
        print("\n" + "="*70)
        print("ESTRUCTURA RHONN - MANIPULADOR 2-DOF")
        print("="*70)
        state_names = ['q₁', 'q₂', 'dq₁', 'dq₂']
        for i in range(self.n_states):
            print(f"\nEstado {state_names[i]} (L={self.n_weights[i]} pesos):")
            for j, inter in enumerate(self.interactions[i]):
                print(f"  w{i}[{j}] * {inter}")
        print(f"\nTotal de pesos: {self.get_total_weights()}")
        print("="*70 + "\n")

# Configurar estructura RHONN para manipulador 2-DOF
def create_manipulator_rhonn_structure():
    """Crea estructura RHONN optimizada para manipulador 2-DOF."""
    rhonn = RHONNStructure(n_states=4)
    
    # Estado 0: q₁(k+1) - principalmente función de dq₁
    rhonn.set_interactions(0, [
        '1.0',              # Bias
        'q1',               # Posición actual
        'dq1',              # Velocidad (término dominante para integración)
        'q1*dq1',           # Interacción no lineal
        'sin(q1)',          # Término trigonométrico
        'q2',               # Acoplamiento
        'dq1*dq1',          # Término cuadrático
    ])
    
    # Estado 1: q₂(k+1) - principalmente función de dq₂
    rhonn.set_interactions(1, [
        '1.0',              # Bias
        'q2',               # Posición actual
        'dq2',              # Velocidad (término dominante)
        'q2*dq2',           # Interacción no lineal
        'sin(q2)',          # Término trigonométrico
        'q1',               # Acoplamiento
        'dq2*dq2',          # Término cuadrático
    ])
    
    # Estado 2: dq₁(k+1) - dinámica compleja con acoplamientos y torques
    rhonn.set_interactions(2, [
        '1.0',              # Bias
        'q1',               # Ángulo 1
        'q2',               # Ángulo 2 (acoplamiento importante)
        'dq1',              # Velocidad 1
        'dq2',              # Velocidad 2 (acoplamiento)
        'sin(q1)',          # Gravedad en q1
        'sin(q2)',          # Acoplamiento gravitacional
        'cos(q2)',          # Términos de masa variable
        'sin(q1+q2)',       # Gravedad en efector final
        'dq1*dq1',          # Centrífuga
        'dq2*dq2',          # Centrífuga acoplada
        'dq1*dq2',          # Coriolis
        'sin(q2)*dq2',      # Coriolis no lineal
        'cos(q2)*dq1',      # Acoplamiento dinámico
        'tau1',             # Entrada de control (lineal)
        'tau2',             # Acoplamiento de entrada
    ])
    
    # Estado 3: dq₂(k+1) - similar a dq₁ pero con diferentes acoplamientos
    rhonn.set_interactions(3, [
        '1.0',              # Bias
        'q1',               # Ángulo 1
        'q2',               # Ángulo 2
        'dq1',              # Velocidad 1 (acoplamiento)
        'dq2',              # Velocidad 2
        'sin(q1)',          # Gravedad acoplada
        'sin(q2)',          # Gravedad en q2
        'cos(q2)',          # Términos de masa variable
        'sin(q1+q2)',       # Gravedad en efector final
        'dq1*dq1',          # Centrífuga acoplada
        'dq2*dq2',          # Centrífuga
        'dq1*dq2',          # Coriolis
        'sin(q2)*dq1',      # Coriolis no lineal
        'cos(q2)*dq2',      # Acoplamiento dinámico
        'tau1',             # Acoplamiento de entrada
        'tau2',             # Entrada de control (lineal)
    ])
    
    return rhonn

# ============================================================
# 3) EKF Trainer (FIEL a Alanis et al.)
# ============================================================
class EKF_RHONN_Trainer:
    """
    Extended Kalman Filter para RHONN.
    Implementación FIEL a Alanis et al. - SIN MODIFICACIONES.
    """
    
    def __init__(self, structure: RHONNStructure, Q: float = 1e-4, 
                 R: float = 1e-2, P0: float = 1.0, eta: float = 1.0):
        self.structure = structure
        self.n_states = structure.n_states
        self.eta = eta
        
        # Inicializar pesos para cada estado
        self.weights = []
        self.P = []
        for i in range(self.n_states):
            n_w = structure.n_weights[i]
            self.weights.append(np.random.randn(n_w) * 0.1)
            self.P.append(np.eye(n_w) * P0)
        
        # Matrices de ruido
        self.Q_matrices = [np.eye(structure.n_weights[i]) * Q 
                          for i in range(self.n_states)]
        self.R = R
        
        self.error_history = []
    
    def update(self, x_kp1: np.ndarray, x_k: np.ndarray, u_k: np.ndarray):
        """Actualización EKF - UN PASO."""
        S_list = self.structure.compute_S(x_k, u_k)
        total_error = 0.0
        
        for i in range(self.n_states):
            # 1. Predicción de covarianza
            P_pred = self.P[i] + self.Q_matrices[i]
            
            # 2. Jacobiano H (en este caso, H = S^T)
            H = S_list[i].reshape(-1, 1)
            
            # 3. Innovación
            y_pred = np.dot(self.weights[i], S_list[i])
            innovation = x_kp1[i] - y_pred
            
            # 4. Ganancia de Kalman
            S_kal = self.R + (H.T @ P_pred @ H)[0, 0]
            K = (P_pred @ H).flatten() / S_kal
            
            # 5. Actualización de pesos
            self.weights[i] += self.eta * K * innovation
            
            # 6. Actualización de covarianza (forma de Joseph)
            I_KH = np.eye(len(S_list[i])) - np.outer(K, S_list[i])
            self.P[i] = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * self.R
            
            total_error += innovation**2
        
        self.error_history.append(np.sqrt(total_error))
    
    def predict(self, x_k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        """Predicción un paso adelante."""
        S_list = self.structure.compute_S(x_k, u_k)
        x_pred = np.zeros(self.n_states)
        for i in range(self.n_states):
            x_pred[i] = np.dot(self.weights[i], S_list[i])
        return x_pred

# ============================================================
# 4) UKF Trainer (FIEL a Rios et al.)
# ============================================================
class UKF_RHONN_Trainer:
    """
    Unscented Kalman Filter para RHONN.
    Implementación FIEL a Rios et al. - SIN MODIFICACIONES.
    """
    
    def __init__(self, structure: RHONNStructure, Q: float = 1e-4,
                 R: float = 1e-2, P0: float = 1.0, eta: float = 1.0,
                 alpha: float = 1e-2, beta: float = 2.0, kappa: float = 0.0):
        self.structure = structure
        self.n_states = structure.n_states
        self.eta = eta
        
        # Inicializar pesos
        self.weights = []
        self.P = []
        for i in range(self.n_states):
            n_w = structure.n_weights[i]
            self.weights.append(np.random.randn(n_w) * 0.1)
            self.P.append(np.eye(n_w) * P0)
        
        self.Q_matrices = [np.eye(structure.n_weights[i]) * Q 
                          for i in range(self.n_states)]
        self.R = R
        
        # Parámetros UKF por estado
        self.ukf_params = []
        for i in range(self.n_states):
            L = structure.n_weights[i]
            lambda_ = alpha**2 * (L + kappa) - L
            
            Wm = np.full(2*L + 1, 1.0 / (2.0 * (L + lambda_)))
            Wc = Wm.copy()
            Wm[0] = lambda_ / (L + lambda_)
            Wc[0] = Wm[0] + (1 - alpha**2 + beta)
            
            self.ukf_params.append({
                'L': L,
                'lambda': lambda_,
                'Wm': Wm,
                'Wc': Wc
            })
        
        self.error_history = []
    
    def generate_sigma_points(self, W: np.ndarray, P: np.ndarray, 
                            params: dict) -> np.ndarray:
        """Genera sigma points."""
        L = params['L']
        lambda_ = params['lambda']
        
        sigma_points = np.zeros((2*L + 1, L))
        sigma_points[0] = W
        
        try:
            U = np.linalg.cholesky((L + lambda_) * P + 1e-9*np.eye(L)).T
        except:
            U = np.eye(L) * 0.1
        
        for i in range(L):
            sigma_points[i+1] = W + U[i]
            sigma_points[L+i+1] = W - U[i]
        
        return sigma_points
    
    def update(self, x_kp1: np.ndarray, x_k: np.ndarray, u_k: np.ndarray):
        """Actualización UKF - UN PASO."""
        S_list = self.structure.compute_S(x_k, u_k)
        total_error = 0.0
        
        for i in range(self.n_states):
            params = self.ukf_params[i]
            
            # 1. Predicción de covarianza
            P_pred = self.P[i] + self.Q_matrices[i]
            
            # 2. Generar sigma points
            sigma_points = self.generate_sigma_points(
                self.weights[i], P_pred, params)
            
            # 3. Propagar sigma points (predicción es lineal: w^T * S)
            Y_sigma = sigma_points @ S_list[i]
            
            # 4. Media predicha
            y_pred = np.sum(params['Wm'] * Y_sigma)
            
            # 5. Covarianza de innovación
            P_yy = self.R
            for k in range(2*params['L'] + 1):
                diff = Y_sigma[k] - y_pred
                P_yy += params['Wc'][k] * diff**2
            
            # 6. Covarianza cruzada
            P_xy = np.zeros(params['L'])
            for k in range(2*params['L'] + 1):
                diff_w = sigma_points[k] - self.weights[i]
                diff_y = Y_sigma[k] - y_pred
                P_xy += params['Wc'][k] * diff_w * diff_y
            
            # 7. Ganancia de Kalman
            K = P_xy / P_yy
            
            # 8. Actualización
            innovation = x_kp1[i] - y_pred
            self.weights[i] += self.eta * K * innovation
            self.P[i] = P_pred - np.outer(K, K) * P_yy
            
            # Regularización
            self.P[i] += np.eye(params['L']) * 1e-6
            
            total_error += innovation**2
        
        self.error_history.append(np.sqrt(total_error))
    
    def predict(self, x_k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        """Predicción un paso adelante."""
        S_list = self.structure.compute_S(x_k, u_k)
        x_pred = np.zeros(self.n_states)
        for i in range(self.n_states):
            x_pred[i] = np.dot(self.weights[i], S_list[i])
        return x_pred

# ============================================================
# 5) PF Trainer OPTIMIZADO
# ============================================================
class PF_RHONN_Trainer:
    """
    Particle Filter OPTIMIZADO para RHONN.
    COMPLETAMENTE VECTORIZADO - Alta eficiencia computacional.
    """
    
    def __init__(self, structure: RHONNStructure, N_particles: int = 150,
                 Q: float = 1e-4, R: float = 1e-2, P0: float = 0.1,
                 resample_threshold: float = 0.5,
                 adaptive_regularization: bool = True):
        self.structure = structure
        self.n_states = structure.n_states
        self.N = N_particles
        self.Q_std = np.sqrt(Q)  # Convert variance to std
        self.R_std = np.sqrt(R)  # Convert variance to std
        self.resample_threshold = resample_threshold * N_particles
        self.adaptive_regularization = adaptive_regularization
        
        # Inicializar partículas para cada estado
        self.particles = []
        self.weights_pf = []
        for i in range(self.n_states):
            n_w = structure.n_weights[i]
            # Inicialización pequeña similar a EKF/UKF
            self.particles.append(np.random.randn(N_particles, n_w) * np.sqrt(P0))
            self.weights_pf.append(np.ones(N_particles) / N_particles)
        
        self.error_history = []
        self.neff_history = []
        self.resample_count = 0
    
    def neff(self, weights: np.ndarray) -> float:
        """Tamaño efectivo de muestra."""
        return 1.0 / np.sum(weights**2)
    
    def systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """Remuestreo sistemático."""
        positions = (np.arange(self.N) + np.random.random()) / self.N
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.0
        return np.searchsorted(cumsum, positions)
    
    def update(self, x_kp1: np.ndarray, x_k: np.ndarray, u_k: np.ndarray):
        """Actualización PF - COMPLETAMENTE VECTORIZADO."""
        S_list = self.structure.compute_S(x_k, u_k)
        total_error = 0.0
        
        for i in range(self.n_states):
            # 1. PREDICCIÓN - Difusión (VECTORIZADO)
            noise = np.random.randn(self.N, len(self.particles[i][0])) * self.Q_std
            self.particles[i] += noise
            
            # 2. ACTUALIZACIÓN DE PESOS (VECTORIZADO)
            # Predicciones: (N x n_weights) @ (n_weights,) = (N,)
            preds = self.particles[i] @ S_list[i]
            
            # Errores y likelihood
            errors = x_kp1[i] - preds
            likelihood = np.exp(-0.5 * (errors / self.R_std)**2)
            
            # Actualizar pesos
            self.weights_pf[i] *= (likelihood + 1e-300)
            self.weights_pf[i] /= np.sum(self.weights_pf[i])
            
            # 3. REMUESTREO ADAPTATIVO
            neff_val = self.neff(self.weights_pf[i])
            
            if neff_val < self.resample_threshold:
                indices = self.systematic_resample(self.weights_pf[i])
                self.particles[i] = self.particles[i][indices]
                self.weights_pf[i] = np.ones(self.N) / self.N
                
                if i == 0:  # Contar solo una vez por iteración
                    self.resample_count += 1
                
                # REGULARIZACIÓN ADAPTATIVA
                if self.adaptive_regularization:
                    h = 0.01 * np.std(self.particles[i], axis=0)  # Mucho más pequeño
                    jitter = np.random.randn(self.N, len(h)) * h[np.newaxis, :]
                    self.particles[i] += jitter
            
            # Guardar métricas
            mean_pred = self.weights_pf[i] @ preds
            total_error += (x_kp1[i] - mean_pred)**2
            
            if i == 0:  # Guardar Neff solo una vez
                self.neff_history.append(neff_val)
        
        self.error_history.append(np.sqrt(total_error))
    
    def predict(self, x_k: np.ndarray, u_k: np.ndarray) -> np.ndarray:
        """Predicción un paso adelante."""
        S_list = self.structure.compute_S(x_k, u_k)
        x_pred = np.zeros(self.n_states)
        for i in range(self.n_states):
            # Para cada partícula, calcular la predicción
            predictions = self.particles[i] @ S_list[i]  # (N,)
            # Media ponderada de las predicciones
            x_pred[i] = np.average(predictions, weights=self.weights_pf[i])
        return x_pred

# ============================================================
# 6) Simulation Main Loop
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("IDENTIFICACIÓN DE MANIPULADOR 2-DOF CON RHONN")
    print("="*70)
    
    # Crear estructura RHONN
    rhonn_structure = create_manipulator_rhonn_structure()
    rhonn_structure.print_structure()
    
    # Parámetros de simulación
    n_steps = 1000
    dt = 0.01
    t = np.linspace(0, (n_steps-1)*dt, n_steps)
    
    n_states = 4
    
    # Inicializar trainers
    print("Inicializando filtros...")
    ekf = EKF_RHONN_Trainer(rhonn_structure, Q=1e-4, R=1e-2, P0=1.0, eta=0.9)
    ukf = UKF_RHONN_Trainer(rhonn_structure, Q=1e-4, R=1e-2, P0=1.0, eta=1.0)
    pf = PF_RHONN_Trainer(rhonn_structure, N_particles=200, 
                          Q=1e-4, R=1e-2, P0=1.0,  # SAME as EKF/UKF!
                          resample_threshold=0.5, 
                          adaptive_regularization=True)
    
    # Arrays de almacenamiento
    x_true = np.zeros((n_steps, 4))
    x_est_ekf = np.zeros((n_steps, 4))
    x_est_ukf = np.zeros((n_steps, 4))
    x_est_pf = np.zeros((n_steps, 4))
    
    # Condiciones iniciales (brazo colgando)
    x_true[0] = [-np.pi/2, 0, 0, 0]
    x_est_ekf[0] = x_true[0]
    x_est_ukf[0] = x_true[0]
    x_est_pf[0] = x_true[0]
    
    # Señal de excitación (torques variables)
    u_hist = np.zeros((n_steps, 2))
    for k in range(n_steps):
        tau1 = 30.0 * np.sin(2.0 * t[k])
        tau2 = 15.0 * np.cos(3.0 * t[k])
        u_hist[k] = [tau1, tau2]
    
    # Simulación
    print("\nSimulando sistema y entrenando identificadores...")
    print("Progreso:")
    
    start_time = time.time()
    training_times = {}
    
    for k in range(n_steps - 1):
        # 1. Simular planta real
        x_true[k+1] = plant(x_true[k], u_hist[k], dt)
        
        # 2. EKF
        t_start = time.time()
        ekf.update(x_true[k+1], x_true[k], u_hist[k])
        x_est_ekf[k+1] = ekf.predict(x_true[k], u_hist[k])
        if k == 0:
            training_times['EKF'] = 0
        training_times['EKF'] += time.time() - t_start
        
        # 3. UKF
        t_start = time.time()
        ukf.update(x_true[k+1], x_true[k], u_hist[k])
        x_est_ukf[k+1] = ukf.predict(x_true[k], u_hist[k])
        if k == 0:
            training_times['UKF'] = 0
        training_times['UKF'] += time.time() - t_start
        
        # 4. PF
        t_start = time.time()
        pf.update(x_true[k+1], x_true[k], u_hist[k])
        x_est_pf[k+1] = pf.predict(x_true[k], u_hist[k])
        if k == 0:
            training_times['PF'] = 0
        training_times['PF'] += time.time() - t_start
        
        if (k+1) % 100 == 0:
            print(f"  {k+1}/{n_steps-1} pasos completados")
    
    total_time = time.time() - start_time
    
    print(f"\nSimulación completada en {total_time:.2f} s")
    print(f"\nTiempos de entrenamiento por filtro:")
    for method, t_train in training_times.items():
        print(f"  {method}: {t_train:.3f} s ({t_train/total_time*100:.1f}% del total)")
    
    # ============================================================
    # 7) Cálculo de Métricas
    # ============================================================
    print("\nCalculando métricas de desempeño...")
    
    # MSE por estado
    mse_ekf = np.mean((x_true - x_est_ekf)**2, axis=0)
    mse_ukf = np.mean((x_true - x_est_ukf)**2, axis=0)
    mse_pf = np.mean((x_true - x_est_pf)**2, axis=0)
    
    mse_total_ekf = np.sum(mse_ekf)
    mse_total_ukf = np.sum(mse_ukf)
    mse_total_pf = np.sum(mse_pf)
    
    # Reporte
    print("\n" + "="*70)
    print("RESULTADOS DE IDENTIFICACIÓN")
    print("="*70)
    
    mse_dict = {'EKF-RHONN': mse_total_ekf, 'UKF-RHONN': mse_total_ukf, 
                'PF-RHONN': mse_total_pf}
    best_filter = min(mse_dict, key=mse_dict.get)
    
    print(f"\n🏆 MEJOR FILTRO: {best_filter}")
    print(f"   MSE Total: {mse_dict[best_filter]:.6f}")
    
    # Mejora porcentual
    improvement_vs_ekf = (mse_total_ekf - mse_total_pf) / mse_total_ekf * 100
    improvement_vs_ukf = (mse_total_ukf - mse_total_pf) / mse_total_ukf * 100
    
    print(f"\nMejora de PF-RHONN:")
    print(f"   vs EKF-RHONN: {improvement_vs_ekf:+.2f}%")
    print(f"   vs UKF-RHONN: {improvement_vs_ukf:+.2f}%")
    
    print(f"\n--- MSE por Estado ---")
    state_names = ['q₁', 'q₂', 'dq₁', 'dq₂']
    for i, name in enumerate(state_names):
        print(f"{name:4s}: EKF={mse_ekf[i]:.6f}  UKF={mse_ukf[i]:.6f}  PF={mse_pf[i]:.6f}")
    
    print(f"\nEstadísticas PF:")
    print(f"  Remuestreos: {pf.resample_count}/{n_steps-1} ({100*pf.resample_count/(n_steps-1):.1f}%)")
    print(f"  Neff medio: {np.mean(pf.neff_history):.1f}")
    print(f"  Neff mínimo: {np.min(pf.neff_history):.1f}")
    
    print("="*70)
    
    # ============================================================
    # 8) Visualización con Plotly
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
    
    # --- Gráficas por Estado ---
    states_info = [
        {'idx': 0, 'var': 'q₁', 'desc': 'Ángulo Articulación 1', 
         'y_label': 'Ángulo q₁ (rad)'},
        {'idx': 1, 'var': 'q₂', 'desc': 'Ángulo Articulación 2', 
         'y_label': 'Ángulo q₂ (rad)'},
        {'idx': 2, 'var': 'dq₁', 'desc': 'Velocidad Articulación 1', 
         'y_label': 'Velocidad dq₁ (rad/s)'},
        {'idx': 3, 'var': 'dq₂', 'desc': 'Velocidad Articulación 2', 
         'y_label': 'Velocidad dq₂ (rad/s)'}
    ]
    
    for state_info in states_info:
        i = state_info['idx']
        
        fig = go.Figure()
        
        # Estado real
        fig.add_trace(go.Scatter(
            x=t, y=x_true[:, i],
            mode='lines',
            name='Estado Real',
            line=dict(color='#000000', width=thesis_config['line_width_true']),
        ))
        
        # Estimaciones
        fig.add_trace(go.Scatter(
            x=t, y=x_est_ekf[:, i],
            mode='lines',
            name=f'EKF-RHONN (MSE={mse_ekf[i]:.2e})',
            line=dict(color='#1f77b4', width=thesis_config['line_width_est'], 
                     dash='dash'),
        ))
        
        fig.add_trace(go.Scatter(
            x=t, y=x_est_ukf[:, i],
            mode='lines',
            name=f'UKF-RHONN (MSE={mse_ukf[i]:.2e})',
            line=dict(color='#2ca02c', width=thesis_config['line_width_est'], 
                     dash='dot'),
        ))
        
        fig.add_trace(go.Scatter(
            x=t, y=x_est_pf[:, i],
            mode='lines',
            name=f'PF-RHONN (MSE={mse_pf[i]:.2e})',
            line=dict(color='#d62728', width=thesis_config['line_width_est'], 
                     dash='dashdot'),
        ))
        
        fig.update_layout(
            title={
                'text': f'Estado {state_info["var"]}: {state_info["desc"]} - Manipulador 2-DOF',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': thesis_config['title_font_size'], 
                        'family': thesis_config['font_family']}
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
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor=thesis_config['grid_color'],
                gridwidth=thesis_config['grid_width'],
                showline=True,
                linewidth=1.5,
                linecolor='black',
                mirror=True,
            ),
            legend=dict(
                x=0.02, y=0.98,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='black',
                borderwidth=1,
            ),
            font=dict(size=thesis_config['font_size'], 
                     family=thesis_config['font_family']),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=thesis_config['plot_width'],
            height=thesis_config['plot_height'],
        )
        
        fig.show()
    
    # --- Gráfica de errores de entrenamiento ---
    fig_train = go.Figure()
    
    fig_train.add_trace(go.Scatter(
        x=list(range(len(ekf.error_history))),
        y=ekf.error_history,
        mode='lines',
        name='EKF-RHONN',
        line=dict(color='#1f77b4', width=1.5)
    ))
    
    fig_train.add_trace(go.Scatter(
        x=list(range(len(ukf.error_history))),
        y=ukf.error_history,
        mode='lines',
        name='UKF-RHONN',
        line=dict(color='#2ca02c', width=1.5)
    ))
    
    fig_train.add_trace(go.Scatter(
        x=list(range(len(pf.error_history))),
        y=pf.error_history,
        mode='lines',
        name='PF-RHONN',
        line=dict(color='#d62728', width=1.5)
    ))
    
    fig_train.update_layout(
        title='Convergencia de Error Durante el Entrenamiento',
        xaxis_title='Iteración',
        yaxis_title='Error RMSE',
        yaxis_type='log',
        plot_bgcolor='white',
        showlegend=True
    )
    
    fig_train.show()
    
    # --- Gráfica de barras MSE ---
    fig_mse = go.Figure()
    
    filters = ['EKF-RHONN', 'UKF-RHONN', 'PF-RHONN']
    
    fig_mse.add_trace(go.Bar(
        name='Ángulo q₁',
        x=filters,
        y=[mse_ekf[0], mse_ukf[0], mse_pf[0]],
        marker_color='#636EFA',
        text=[f'{mse_ekf[0]:.2e}', f'{mse_ukf[0]:.2e}', f'{mse_pf[0]:.2e}'],
        textposition='outside'
    ))
    
    fig_mse.add_trace(go.Bar(
        name='Ángulo q₂',
        x=filters,
        y=[mse_ekf[1], mse_ukf[1], mse_pf[1]],
        marker_color='#EF553B',
        text=[f'{mse_ekf[1]:.2e}', f'{mse_ukf[1]:.2e}', f'{mse_pf[1]:.2e}'],
        textposition='outside'
    ))
    
    fig_mse.add_trace(go.Bar(
        name='Velocidad dq₁',
        x=filters,
        y=[mse_ekf[2], mse_ukf[2], mse_pf[2]],
        marker_color='#00CC96',
        text=[f'{mse_ekf[2]:.2e}', f'{mse_ukf[2]:.2e}', f'{mse_pf[2]:.2e}'],
        textposition='outside'
    ))
    
    fig_mse.add_trace(go.Bar(
        name='Velocidad dq₂',
        x=filters,
        y=[mse_ekf[3], mse_ukf[3], mse_pf[3]],
        marker_color='#AB63FA',
        text=[f'{mse_ekf[3]:.2e}', f'{mse_ukf[3]:.2e}', f'{mse_pf[3]:.2e}'],
        textposition='outside'
    ))
    
    fig_mse.update_layout(
        title='Comparación de MSE por Estado - Manipulador 2-DOF',
        xaxis_title='Tipo de Filtro',
        yaxis_title='MSE',
        yaxis_type='log',
        barmode='group',
        plot_bgcolor='white'
    )
    
    fig_mse.show()
    
    print("\n✅ Visualización completada.")
    print("\nNOTA: Para animación del robot, ejecutar en entorno con soporte Plotly interactivo.")