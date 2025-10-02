
#!/usr/bin/env python3
# Neural Identifier Training with Particle Filters - Differential Drive Mobile Robot (with animation)
import argparse
import numpy as np
import plotly.graph_objects as go

# ============================================================
# 1) True nonlinear system (Differential Drive Mobile Robot)
# ============================================================
def plant_dynamics(x, u, L=0.5, friction_coeff=0.1, slip_factor=0.05):
    """
    Continuous dynamics for differential drive mobile robot: x = [x_pos, y_pos, theta]. 
    Returns x_dot.
    """
    x_pos, y_pos, theta = x
    v_l, v_r = u  # left and right wheel velocities

    # Add realistic wheel slip effects
    v_l_actual = v_l * (1 - slip_factor * np.random.randn())
    v_r_actual = v_r * (1 - slip_factor * np.random.randn())

    # Compute linear and angular velocities
    v = (v_r_actual + v_l_actual) / 2.0
    omega = (v_r_actual - v_l_actual) / L

    # Add friction effects (velocity-dependent)
    v_friction = v * (1 - friction_coeff * np.abs(v))
    omega_friction = omega * (1 - friction_coeff * np.abs(omega))

    # Robot kinematics
    x_dot = v_friction * np.cos(theta)
    y_dot = v_friction * np.sin(theta)
    theta_dot = omega_friction
    return np.array([x_dot, y_dot, theta_dot])

def plant(x_k, u_k, dt=0.01, process_noise_type='mixed', process_noise_std=0.05, 
          terrain_roughness=0.02, sensor_bias=[0.0, 0.0, 0.0]):
    """
    One Euler step of the discrete plant with realistic mobile robot disturbances.
    """
    x_dot = plant_dynamics(x_k, u_k)
    x_kp1 = x_k + dt * x_dot

    # Realistic mobile robot disturbances
    terrain_noise = terrain_roughness * np.array([
        np.sin(0.5 * x_kp1[0]) * np.random.randn(),
        np.cos(0.3 * x_kp1[1]) * np.random.randn(),
        0.1 * np.sin(x_kp1[2]) * np.random.randn()
    ])
    velocity_magnitude = np.linalg.norm(u_k)
    velocity_noise_factor = 1 + 0.2 * velocity_magnitude
    if process_noise_type == 'mixed':
        gaussian_noise = np.random.normal(0, process_noise_std * velocity_noise_factor, size=x_kp1.shape)
        impulse_prob = 0.02
        impulse_noise = np.zeros_like(x_kp1)
        if np.random.rand() < impulse_prob:
            impulse_noise = np.random.normal(0, process_noise_std * 5, size=x_kp1.shape)
        laplacian_noise = np.random.laplace(0, process_noise_std * 0.3, size=x_kp1.shape)
        total_noise = gaussian_noise + impulse_noise + laplacian_noise
    elif process_noise_type == 'laplacian':
        total_noise = np.random.laplace(0, process_noise_std * velocity_noise_factor, size=x_kp1.shape)
    elif process_noise_type == 'uniform':
        a = np.sqrt(3) * process_noise_std * velocity_noise_factor
        total_noise = np.random.uniform(-a, a, size=x_kp1.shape)
    else:  # gaussian
        total_noise = np.random.normal(0, process_noise_std * velocity_noise_factor, size=x_kp1.shape)
    bias_noise = np.array(sensor_bias) * dt
    encoder_resolution = 0.001
    quantization_noise = encoder_resolution * (np.random.rand(3) - 0.5)
    x_kp1 += terrain_noise + total_noise + bias_noise + quantization_noise
    x_kp1[2] = np.arctan2(np.sin(x_kp1[2]), np.cos(x_kp1[2]))  # wrap angle to [-π, π]
    return x_kp1

def generate_realistic_trajectory(t, trajectory_type='mixed'):
    """
    Generate realistic control inputs for mobile robot.
    Returns u: [v_left, v_right]
    """
    if trajectory_type == 'straight':
        v_base = 1.0 + 0.2 * np.sin(0.5 * t)
        return np.array([v_base, v_base])
    elif trajectory_type == 'circle':
        v_l = 1.0 + 0.1 * np.sin(t)
        v_r = 1.5 + 0.1 * np.cos(t)
        return np.array([v_l, v_r])
    elif trajectory_type == 'figure8':
        v_l = 1.0 + 0.8 * np.sin(0.5 * t)
        v_r = 1.0 - 0.8 * np.sin(0.5 * t)
        return np.array([v_l, v_r])
    elif trajectory_type == 'obstacle_avoidance':
        base_speed = 1.2
        avoidance_maneuver = 0.5 * np.sin(2 * t) * np.exp(-0.1 * t)
        v_l = base_speed + avoidance_maneuver
        v_r = base_speed - avoidance_maneuver
        return np.array([v_l, v_r])
    else:  # mixed
        phase = (t % 20.0) / 20.0
        if phase < 0.3:
            v_base = 1.5
            return np.array([v_base, v_base])
        elif phase < 0.6:
            v_l = 0.8; v_r = 1.8
            return np.array([v_l, v_r])
        elif phase < 0.8:
            v_base = -0.5
            return np.array([v_base, v_base])
        else:
            v_l = 1.0 + 0.5 * np.sin(10 * t)
            v_r = 1.0 + 0.5 * np.cos(10 * t)
            return np.array([v_l, v_r])

# ============================================================
# 2) Realistic Robot Controller (Trajectory Tracking)
# ============================================================
class TrajectoryTrackingController:
    """
    Lyapunov-based trajectory tracking controller for differential drive robot.
    Follows a desired trajectory with realistic control constraints.
    """
    def __init__(self, L=0.5, kp_v=1.0, kp_w=2.0, kd_v=0.1, kd_w=0.2,
                 v_max=2.0, w_max=3.0, v_min=-1.0, w_min=-3.0,
                 acceleration_limit=2.0, jerk_limit=5.0):
        self.L = L  # wheelbase
        self.kp_v = kp_v  # proportional gain for linear velocity
        self.kp_w = kp_w  # proportional gain for angular velocity
        self.kd_v = kd_v  # derivative gain for linear velocity
        self.kd_w = kd_w  # derivative gain for angular velocity
        
        # Control constraints
        self.v_max = v_max; self.v_min = v_min
        self.w_max = w_max; self.w_min = w_min
        self.acceleration_limit = acceleration_limit
        self.jerk_limit = jerk_limit
        
        # Control history for derivative terms and smoothing
        self.prev_v_cmd = 0.0; self.prev_w_cmd = 0.0
        self.prev_prev_v_cmd = 0.0; self.prev_prev_w_cmd = 0.0
        self.prev_error_x = 0.0; self.prev_error_y = 0.0; self.prev_error_theta = 0.0
    
    def get_desired_trajectory_point(self, t, trajectory_type='circle'):
        """Generate desired trajectory point [x_d, y_d, theta_d, v_d, w_d]"""
        if trajectory_type == 'circle':
            R = 3.0  # radius
            omega = 0.3  # angular frequency
            x_d = R * np.cos(omega * t)
            y_d = R * np.sin(omega * t)
            theta_d = omega * t + np.pi/2
            v_d = R * omega
            w_d = omega
        elif trajectory_type == 'figure8':
            a = 2.0; b = 1.0; omega = 0.2
            x_d = a * np.sin(omega * t)
            y_d = b * np.sin(2 * omega * t)
            # Compute desired velocity and angular velocity
            dx_dt = a * omega * np.cos(omega * t)
            dy_dt = 2 * b * omega * np.cos(2 * omega * t)
            v_d = np.sqrt(dx_dt**2 + dy_dt**2)
            theta_d = np.arctan2(dy_dt, dx_dt)
            
            # Angular velocity from derivative of theta_d
            d2x_dt2 = -a * omega**2 * np.sin(omega * t)
            d2y_dt2 = -4 * b * omega**2 * np.sin(2 * omega * t)
            w_d = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (dx_dt**2 + dy_dt**2) if v_d > 0.01 else 0.0
        elif trajectory_type == 'straight_line':
            v_d = 1.0
            x_d = v_d * t
            y_d = 0.5 * np.sin(0.1 * t)  # slight sinusoidal perturbation
            theta_d = np.arctan2(0.05 * np.cos(0.1 * t), 1.0)
            w_d = 0.005 * np.sin(0.1 * t)
        else:  # 'square'
            side_length = 4.0; speed = 1.0; side_time = side_length / speed
            total_time = 4 * side_time
            t_cycle = t % total_time
            
            if t_cycle < side_time:  # side 1: move right
                x_d = speed * t_cycle; y_d = 0.0; theta_d = 0.0; v_d = speed; w_d = 0.0
            elif t_cycle < 2 * side_time:  # side 2: move up
                x_d = side_length; y_d = speed * (t_cycle - side_time); theta_d = np.pi/2; v_d = speed; w_d = 0.0
            elif t_cycle < 3 * side_time:  # side 3: move left
                x_d = side_length - speed * (t_cycle - 2*side_time); y_d = side_length; theta_d = np.pi; v_d = speed; w_d = 0.0
            else:  # side 4: move down
                x_d = 0.0; y_d = side_length - speed * (t_cycle - 3*side_time); theta_d = -np.pi/2; v_d = speed; w_d = 0.0
        
        return np.array([x_d, y_d, theta_d, v_d, w_d])
    
    def compute_control(self, current_pose, desired_traj_point, dt):
        """
        Compute control inputs [v_linear, w_angular] using trajectory tracking controller.
        current_pose: [x, y, theta]
        desired_traj_point: [x_d, y_d, theta_d, v_d, w_d]
        """
        x, y, theta = current_pose
        x_d, y_d, theta_d, v_d, w_d = desired_traj_point
        
        # Position errors in global frame
        error_x_global = x_d - x
        error_y_global = y_d - y
        
        # Transform errors to robot local frame
        cos_theta = np.cos(theta); sin_theta = np.sin(theta)
        error_x_local = cos_theta * error_x_global + sin_theta * error_y_global
        error_y_local = -sin_theta * error_x_global + cos_theta * error_y_global
        
        # Angular error (wrap to [-pi, pi])
        error_theta = theta_d - theta
        error_theta = np.arctan2(np.sin(error_theta), np.cos(error_theta))
        
        # Derivative terms (simple numerical differentiation)
        d_error_x = (error_x_local - self.prev_error_x) / dt if dt > 0 else 0.0
        d_error_y = (error_y_local - self.prev_error_y) / dt if dt > 0 else 0.0
        d_error_theta = (error_theta - self.prev_error_theta) / dt if dt > 0 else 0.0
        
        # Control law (Lyapunov-based with PD terms)
        k1 = self.kp_v; k2 = self.kp_w; k3 = self.kp_w
        
        # Linear velocity control
        v_control = v_d * np.cos(error_theta) + k1 * error_x_local + self.kd_v * d_error_x
        
        # Angular velocity control  
        w_control = w_d + k2 * v_d * error_y_local + k3 * np.sin(error_theta) + self.kd_w * d_error_theta
        
        # Apply control constraints with smooth saturation
        v_control = np.clip(v_control, self.v_min, self.v_max)
        w_control = np.clip(w_control, self.w_min, self.w_max)
        
        # Acceleration limiting for smoother control
        if dt > 0:
            dv_dt = (v_control - self.prev_v_cmd) / dt
            dw_dt = (w_control - self.prev_w_cmd) / dt
            
            if abs(dv_dt) > self.acceleration_limit:
                v_control = self.prev_v_cmd + np.sign(dv_dt) * self.acceleration_limit * dt
            if abs(dw_dt) > self.acceleration_limit * 2:  # higher limit for angular accel
                w_control = self.prev_w_cmd + np.sign(dw_dt) * self.acceleration_limit * 2 * dt
            
            # Jerk limiting (rate of acceleration change)
            d2v_dt2 = ((v_control - self.prev_v_cmd) / dt - (self.prev_v_cmd - self.prev_prev_v_cmd) / dt) / dt
            d2w_dt2 = ((w_control - self.prev_w_cmd) / dt - (self.prev_w_cmd - self.prev_prev_w_cmd) / dt) / dt
            
            if abs(d2v_dt2) > self.jerk_limit:
                # Limit jerk by adjusting current command
                max_dv = self.jerk_limit * dt**2 + (self.prev_v_cmd - self.prev_prev_v_cmd)
                v_control = self.prev_v_cmd + np.sign(v_control - self.prev_v_cmd) * min(abs(v_control - self.prev_v_cmd), abs(max_dv))
        
        # Convert to wheel velocities
        v_left = v_control - (w_control * self.L) / 2.0
        v_right = v_control + (w_control * self.L) / 2.0
        
        # Update history
        self.prev_prev_v_cmd = self.prev_v_cmd; self.prev_prev_w_cmd = self.prev_w_cmd
        self.prev_v_cmd = v_control; self.prev_w_cmd = w_control
        self.prev_error_x = error_x_local; self.prev_error_y = error_y_local; self.prev_error_theta = error_theta
        
        # Add realistic actuator noise and dynamics
        actuator_noise = 0.02  # 2% noise
        v_left *= (1 + actuator_noise * np.random.randn())
        v_right *= (1 + actuator_noise * np.random.randn())
        
        return np.array([v_left, v_right]), np.array([v_control, w_control])

# ============================================================
# 3) RHONN structure
# ============================================================
def sigmoidal(z, beta=1.0):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-beta * z))

def construct_z_vector(x_est, u_input=None):
    """
    Features for a 3-state mobile robot system with control inputs:
    x = [x_pos, y_pos, theta], u = [v_left, v_right]
    """
    s_x = sigmoidal(x_est[0])
    s_y = sigmoidal(x_est[1])
    s_theta = sigmoidal(x_est[2])
    features = [
        s_x, s_y, s_theta,
        s_x*s_y, s_x*s_theta, s_y*s_theta,
        s_x**2, s_y**2, s_theta**2,
        np.cos(x_est[2]), np.sin(x_est[2]),
    ]
    if u_input is not None and len(u_input) >= 2:
        s_vl = sigmoidal(u_input[0])
        s_vr = sigmoidal(u_input[1])
        features.extend([s_vl, s_vr, s_vl*s_vr])
    else:
        features.extend([0.0, 0.0, 0.0])
    features.extend([x_est[0], x_est[1], 1.0])
    return np.array(features)

def RHONN_predict(x_state_for_z, w_neuron, u_input=None):
    z_i = construct_z_vector(x_state_for_z, u_input)
    if len(z_i) != len(w_neuron):
        raise ValueError(f"Dimension mismatch: z({len(z_i)}) vs w({len(w_neuron)})")
    return float(np.dot(w_neuron, z_i))

# ============================================================
# 4) EKF trainer over weights
# ============================================================
class EKF_RHONN_Trainer:
    def __init__(self, num_neurons, num_weights_per_neuron, initial_weights=None,
                 Q_init=1e-4, R_init=1e-2, P_init=1.0, eta=1.0):
        self.num_neurons = num_neurons
        self.num_weights_per_neuron = num_weights_per_neuron
        self.eta = eta
        self.weights = []; self.P = []; self.Q = []; self.R = []
        for i in range(num_neurons):
            w_i = np.copy(initial_weights[i]) if (initial_weights is not None and i < len(initial_weights)) \
                  else np.random.randn(num_weights_per_neuron) * 0.1
            self.weights.append(w_i)
            self.P.append(np.eye(num_weights_per_neuron) * P_init)
            self.Q.append(np.eye(num_weights_per_neuron) * Q_init)
            self.R.append(np.array([R_init]))
    def update(self, chi_kp1, chi_k, x_hat_previous, u_input=None):
        x_state_for_z = np.copy(x_hat_previous)
        x_state_for_z[0] = chi_k[0]  # measured x at k
        z_i = construct_z_vector(x_state_for_z, u_input)
        H_i = z_i.reshape(-1, 1)
        for i in range(self.num_neurons):
            P_pred = self.P[i] + self.Q[i] + np.eye(self.num_weights_per_neuron)*1e-8
            M_i = self.R[i][0] + (H_i.T @ P_pred @ H_i)[0, 0]
            M_i = max(M_i, 1e-10)
            x_hat_pred_i = float(self.weights[i] @ z_i)
            e_i = float(np.clip(chi_kp1[i] - x_hat_pred_i, -10.0, 10.0))
            K_i = (P_pred @ H_i).flatten() / M_i
            adaptive_eta = self.eta * (1.0 / (1.0 + np.abs(e_i) * 0.1))
            self.weights[i] += adaptive_eta * K_i * e_i
            I_KH = np.eye(self.num_weights_per_neuron) - np.outer(K_i, H_i.ravel())
            self.P[i] = I_KH @ P_pred @ I_KH.T + np.outer(K_i, K_i) * self.R[i][0]
            self.P[i] = 0.5 * (self.P[i] + self.P[i].T)
            ev = np.linalg.eigvals(self.P[i])
            if np.min(ev) <= 0:
                self.P[i] += np.eye(self.num_weights_per_neuron) * 1e-6

# ============================================================
# 5) Particle Filter trainer over weights (original scalar Q/R)
# ============================================================
class PF_RHONN_Trainer:
    """
    Particle filter over neuron weights (per neuron).
    - Predict (random walk on weights)
    - Update (likelihood from chi_{k+1} vs prediction built with z at k)
    - ESS-triggered resampling
    """
    def __init__(self, num_neurons, num_weights_per_neuron, n_particles=100,
                 initial_weights=None, Q_std=0.05, R_std=0.1, ess_threshold=None):
        self.num_neurons = num_neurons
        self.num_weights_per_neuron = num_weights_per_neuron
        self.n_particles = n_particles
        self.Q_std = Q_std
        self.R_std = R_std
        self.R_var = R_std**2
        self.ess_threshold = ess_threshold if ess_threshold is not None else n_particles / 2.0
        self.particles = []; self.weights_pf = []
        for i in range(num_neurons):
            if initial_weights is not None and i < len(initial_weights):
                base = np.copy(initial_weights[i])
                particles_i = base + np.random.randn(n_particles, num_weights_per_neuron) * 0.1
            else:
                particles_i = np.random.randn(n_particles, num_weights_per_neuron) * 0.1
            self.particles.append(particles_i)
            self.weights_pf.append(np.ones(n_particles) / n_particles)
    @staticmethod
    def _ess(w):
        w = w / np.sum(w)
        return 1.0 / np.sum(w**2)
    def _resample_systematic(self, neuron_index):
        w = self.weights_pf[neuron_index]
        p = self.particles[neuron_index]
        w = w / np.sum(w)
        N = len(w)
        u0 = np.random.uniform(0.0, 1.0 / N)
        cdf = np.cumsum(w)
        indexes = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            u = u0 + i / N
            while u > cdf[j]:
                j += 1
            indexes[i] = j
            i += 1
        self.particles[neuron_index] = p[indexes]
        self.weights_pf[neuron_index] = np.ones(N) / N
    def update(self, chi_kp1, chi_k, x_hat_previous, u_input=None):
        # Build z from time k (series-parallel)
        x_state_for_z = np.copy(x_hat_previous)
        x_state_for_z[0] = chi_k[0]  # measured x at k
        z = construct_z_vector(x_state_for_z, u_input)  # (num_features,)
        # 1) Predict
        for i in range(self.num_neurons):
            self.particles[i] += np.random.randn(self.n_particles, self.num_weights_per_neuron) * self.Q_std
        # 2) Update
        for i in range(self.num_neurons):
            w_mat = self.particles[i]
            x_pred_particles = w_mat @ z
            innov = chi_kp1[i] - x_pred_particles
            ll = -0.5 * (innov**2) / self.R_var
            ll -= np.max(ll)
            like = np.exp(ll)
            self.weights_pf[i] *= like
            s = np.sum(self.weights_pf[i])
            if s < 1e-300:
                self.weights_pf[i] = np.ones(self.n_particles) / self.n_particles
            else:
                self.weights_pf[i] /= s
            if self._ess(self.weights_pf[i]) < self.ess_threshold:
                self._resample_systematic(i)
    def get_estimate(self):
        return [np.mean(self.particles[i], axis=0) for i in range(self.num_neurons)]

# ============================================================
# 5b) Unscented Kalman Filter (UKF) trainer over weights
# ============================================================
class UKF_RHONN_Trainer:
    def __init__(self, num_neurons, num_weights_per_neuron, initial_weights=None,
                 Q_init=1e-4, R_init=1e-2, P_init=1.0, eta=1.0, 
                 alpha=1e-3, beta=2.0, kappa=None):
        self.num_neurons = num_neurons
        self.num_weights_per_neuron = num_weights_per_neuron
        self.eta = eta
        self.alpha = alpha; self.beta = beta
        self.kappa = kappa if kappa is not None else 3 - num_weights_per_neuron
        self.n = num_weights_per_neuron
        self.lambda_ = alpha**2 * (self.n + self.kappa) - self.n
        self.Wm = np.zeros(2 * self.n + 1); self.Wc = np.zeros(2 * self.n + 1)
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.lambda_ / (self.n + self.lambda_) + (1 - alpha**2 + beta)
        for i in range(1, 2 * self.n + 1):
            self.Wm[i] = 1.0 / (2 * (self.n + self.lambda_))
            self.Wc[i] = 1.0 / (2 * (self.n + self.lambda_))
        self.weights = []; self.P = []; self.Q = []; self.R = []
        for i in range(num_neurons):
            w_i = np.copy(initial_weights[i]) if (initial_weights is not None and i < len(initial_weights)) \
                  else np.random.randn(num_weights_per_neuron) * 0.1
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
            sqrt = U @ np.diag(np.sqrt(s)) @ Vh
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
            predicted_sigma_points = sigma_points.copy()
            predicted_mean = np.sum(self.Wm[:, np.newaxis] * predicted_sigma_points, axis=0)
            predicted_cov = self.Q[i].copy()
            for j in range(2 * self.n + 1):
                diff = predicted_sigma_points[j] - predicted_mean
                predicted_cov += self.Wc[j] * np.outer(diff, diff)
            measurement_sigma_points = np.zeros(2 * self.n + 1)
            for j in range(2 * self.n + 1):
                measurement_sigma_points[j] = self._measurement_function(predicted_sigma_points[j], z_i)
            predicted_measurement = np.sum(self.Wm * measurement_sigma_points)
            innovation_cov = self.R[i][0]
            for j in range(2 * self.n + 1):
                diff_meas = measurement_sigma_points[j] - predicted_measurement
                innovation_cov += self.Wc[j] * diff_meas**2
            cross_cov = np.zeros(self.n)
            for j in range(2 * self.n + 1):
                diff_state = predicted_sigma_points[j] - predicted_mean
                diff_meas = measurement_sigma_points[j] - predicted_measurement
                cross_cov += self.Wc[j] * diff_state * diff_meas
            if innovation_cov < 1e-12: innovation_cov = 1e-12
            K = cross_cov / innovation_cov
            innovation = chi_kp1[i] - predicted_measurement
            self.weights[i] = predicted_mean + self.eta * K * innovation
            self.P[i] = predicted_cov - np.outer(K, K) * innovation_cov
            self.P[i] = 0.5 * (self.P[i] + self.P[i].T)
            ev = np.linalg.eigvals(self.P[i])
            if np.min(ev) <= 0:
                self.P[i] += np.eye(self.n) * 1e-6

# ============================================================
# 6) Animation helper
# ============================================================
def build_animation(t, x_true, x_ekf, x_ukf, x_pf, stride=5, arrow_len=0.25, out_html="robot_animation.html", desired_traj=None):
    """
    Creates an animated 2D playback of the trajectory and estimates, and saves to HTML.
    """
    n = len(t)
    idxs = list(range(0, n, stride))
    # Axis ranges with margins
    x_all = np.concatenate([x_true[:,0], x_ekf[:,0], x_ukf[:,0], x_pf[:,0]])
    y_all = np.concatenate([x_true[:,1], x_ekf[:,1], x_ukf[:,1], x_pf[:,1]])
    
    # Include desired trajectory in range calculation if available
    if desired_traj is not None:
        x_all = np.concatenate([x_all, desired_traj[:,0]])
        y_all = np.concatenate([y_all, desired_traj[:,1]])
    
    x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    dx = max(1e-3, 0.1*(x_max - x_min)); dy = max(1e-3, 0.1*(y_max - y_min))
    x_range = [x_min - dx, x_max + dx]; y_range = [y_min - dy, y_max + dy]

    def head_point(p, th):
        return (p[0] + arrow_len*np.cos(th), p[1] + arrow_len*np.sin(th))

    # Static background: final true path (light gray)
    static_path = go.Scatter(x=x_true[:,0], y=x_true[:,1], mode='lines',
                             line=dict(color='lightgray', width=2), name='True Path (full)', showlegend=True)
    
    # Static background: desired trajectory (if available)
    static_desired = None
    if desired_traj is not None:
        static_desired = go.Scatter(x=desired_traj[:,0], y=desired_traj[:,1], mode='lines',
                                   line=dict(color='red', width=2, dash='dash'), 
                                   name='Desired Trajectory', showlegend=True)

    # Initial data placeholders for animated traces
    data = [
        go.Scatter(x=[x_true[0,0]], y=[x_true[0,1]], mode='lines',
                   line=dict(color='black', width=3), name='True Path (progress)'),
        go.Scatter(x=[x_true[0,0]], y=[x_true[0,1]], mode='markers',
                   marker=dict(size=8), name='True pose'),
        go.Scatter(x=[x_true[0,0], head_point(x_true[0], x_true[0,2])[0]],
                   y=[x_true[0,1], head_point(x_true[0], x_true[0,2])[1]],
                   mode='lines', line=dict(width=3), name='True heading'),
        go.Scatter(x=[x_ekf[0,0]], y=[x_ekf[0,1]], mode='markers',
                   marker=dict(size=7), name='EKF pose'),
        go.Scatter(x=[x_ekf[0,0], head_point(x_ekf[0], x_ekf[0,2])[0]],
                   y=[x_ekf[0,1], head_point(x_ekf[0], x_ekf[0,2])[1]],
                   mode='lines', name='EKF heading'),
        go.Scatter(x=[x_ukf[0,0]], y=[x_ukf[0,1]], mode='markers',
                   marker=dict(size=7), name='UKF pose'),
        go.Scatter(x=[x_ukf[0,0], head_point(x_ukf[0], x_ukf[0,2])[0]],
                   y=[x_ukf[0,1], head_point(x_ukf[0], x_ukf[0,2])[1]],
                   mode='lines', name='UKF heading'),
        go.Scatter(x=[x_pf[0,0]], y=[x_pf[0,1]], mode='markers',
                   marker=dict(size=7), name='PF pose'),
        go.Scatter(x=[x_pf[0,0], head_point(x_pf[0], x_pf[0,2])[0]],
                   y=[x_pf[0,1], head_point(x_pf[0], x_pf[0,2])[1]],
                   mode='lines', name='PF heading'),
        static_path
    ]

    frames = []
    for k in idxs:
        # path up to k
        path_k = go.Scatter(x=x_true[:k+1,0], y=x_true[:k+1,1])
        # markers and headings
        tru_head = head_point(x_true[k], x_true[k,2])
        ekf_head = head_point(x_ekf[k], x_ekf[k,2])
        ukf_head = head_point(x_ukf[k], x_ukf[k,2])
        pf_head  = head_point(x_pf[k],  x_pf[k,2])
        frame = go.Frame(
            data=[
                go.Scatter(x=path_k.x, y=path_k.y),                       # 0 path progress
                go.Scatter(x=[x_true[k,0]], y=[x_true[k,1]]),             # 1 true pose
                go.Scatter(x=[x_true[k,0], tru_head[0]], y=[x_true[k,1], tru_head[1]]),  # 2 true heading
                go.Scatter(x=[x_ekf[k,0]], y=[x_ekf[k,1]]),               # 3 ekf pose
                go.Scatter(x=[x_ekf[k,0], ekf_head[0]], y=[x_ekf[k,1], ekf_head[1]]),    # 4 ekf heading
                go.Scatter(x=[x_ukf[k,0]], y=[x_ukf[k,1]]),               # 5 ukf pose
                go.Scatter(x=[x_ukf[k,0], ukf_head[0]], y=[x_ukf[k,1], ukf_head[1]]),    # 6 ukf heading
                go.Scatter(x=[x_pf[k,0]], y=[x_pf[k,1]]),                 # 7 pf pose
                go.Scatter(x=[x_pf[k,0], pf_head[0]],  y=[x_pf[k,1], pf_head[1]]),       # 8 pf heading
            ],
            name=f"frame{k}"
        )
        frames.append(frame)

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title="Differential-Drive Robot – Animated Trajectory and Estimates",
            xaxis=dict(title="X [m]", range=x_range, scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y [m]", range=y_range),
            showlegend=True,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, {"frame": {"duration": 40, "redraw": True},
                                      "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}])
                ]
            )],
            sliders=[dict(
                steps=[dict(method="animate",
                            args=[[f"frame{k}"],
                                  {"mode": "immediate",
                                   "frame": {"duration": 0, "redraw": True},
                                   "transition": {"duration": 0}}],
                            label=f"{t[k]:.2f}s") for k in idxs],
                transition={"duration": 0},
                x=0, y=0, currentvalue=dict(prefix="t=", visible=True), len=1.0
            )]
        ),
        frames=frames
    )
    fig.write_html(out_html, include_plotlyjs="cdn", auto_play=False)
    return out_html

# ============================================================
# 7) Simulation
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true", help="Skip interactive plots")
    parser.add_argument("--animate", action="store_true", help="Generate HTML animation")
    parser.add_argument("--frames-stride", type=int, default=5, help="Animation stride (higher=fewer frames)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--use-controller", action="store_true", help="Use realistic trajectory tracking controller")
    parser.add_argument("--controller-trajectory", type=str, default="circle", 
                       choices=["circle", "figure8", "straight_line", "square"],
                       help="Trajectory type for controller")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # --- Simulation settings ---
    n_steps = 1500
    dt = 0.02
    t_history = np.linspace(0, (n_steps-1) * dt, n_steps)
    process_noise_type = 'mixed'
    process_noise_std = 0.03
    terrain_roughness = 0.02
    sensor_bias = [0.001, 0.001, 0.002]

    # --- True system init ---
    x_true = np.zeros((n_steps, 3))
    x_true[0] = [0.0, 0.0, 0.0]

    # --- Control trajectory ---
    trajectory_type = 'mixed'
    
    # --- Controller setup ---
    controller = None
    desired_trajectory = np.zeros((n_steps, 5))  # [x_d, y_d, theta_d, v_d, w_d]
    control_commands = np.zeros((n_steps, 2))    # [v_left, v_right]
    control_signals = np.zeros((n_steps, 2))     # [v_linear, w_angular]
    
    if args.use_controller:
        controller = TrajectoryTrackingController(
            L=0.5, kp_v=1.5, kp_w=2.5, kd_v=0.2, kd_w=0.3,
            v_max=2.0, w_max=2.5, v_min=-1.5, w_min=-2.5,
            acceleration_limit=1.5, jerk_limit=4.0
        )
        print(f"Using trajectory tracking controller with {args.controller_trajectory} trajectory")
        
        # Pre-compute desired trajectory
        for k in range(n_steps):
            t_k = k * dt
            desired_trajectory[k] = controller.get_desired_trajectory_point(t_k, args.controller_trajectory)

    # --- RHONN config ---
    num_neurons = 3
    num_features = 17
    num_weights_per_neuron = num_features

    # --- Common initial weights ---
    common_initial_weights = [np.random.uniform(-0.5, 0.5, num_weights_per_neuron) for _ in range(num_neurons)]
    print("Common Initial Weights:")
    for i, w in enumerate(common_initial_weights):
        print(f"  Neuron {i}: {w}")

    # --- EKF ---
    ekf_trainer = EKF_RHONN_Trainer(
        num_neurons, num_weights_per_neuron,
        initial_weights=common_initial_weights,
        Q_init=2e-4, R_init=8e-3, P_init=1.5, eta=0.4
    )
    x_hat_ekf = np.zeros((n_steps, 3))
    x_hat_ekf[0] = x_true[0]

    # --- UKF ---
    ukf_trainer = UKF_RHONN_Trainer(
        num_neurons, num_weights_per_neuron,
        initial_weights=common_initial_weights,
        Q_init=2e-4, R_init=8e-3, P_init=1.5, eta=0.6,
        alpha=1e-2, beta=2.0
    )
    x_hat_ukf = np.zeros((n_steps, 3))
    x_hat_ukf[0] = x_true[0]

    # --- PF ---
    n_particles = 700
    pf_trainer = PF_RHONN_Trainer(
        num_neurons, num_weights_per_neuron,
        n_particles=n_particles,
        initial_weights=common_initial_weights,
        Q_std=0.02, R_std=np.sqrt(0.005), ess_threshold=n_particles / 2
    )

    # Optional: identical particle init
    def initialize_pf_with_common_weights(pf_trainer_instance, common_weights_list):
        for i in range(pf_trainer_instance.num_neurons):
            pf_trainer_instance.particles[i] = np.tile(
                common_weights_list[i], (pf_trainer_instance.n_particles, 1)
            )
            pf_trainer_instance.weights_pf[i] = np.ones(pf_trainer_instance.n_particles) / pf_trainer_instance.n_particles
    initialize_pf_with_common_weights(pf_trainer, common_initial_weights)

    x_hat_pf = np.zeros((n_steps, 3))
    x_hat_pf[0] = x_true[0]

    print("Starting mobile robot simulation...")
    for k in range(n_steps - 1):
        # 1) Generate control input
        t_current = k * dt
        
        if args.use_controller and controller is not None:
            # Use realistic trajectory tracking controller
            u_current, control_signals[k] = controller.compute_control(
                x_true[k], desired_trajectory[k], dt
            )
            control_commands[k] = u_current
        else:
            # Use original open-loop trajectory generation
            u_current = generate_realistic_trajectory(t_current, trajectory_type)
            control_commands[k] = u_current
        
        # 2) true system -> k+1
        x_true[k+1] = plant(x_true[k], u_current, dt, process_noise_type, process_noise_std, terrain_roughness, sensor_bias)

        # 3) EKF update + predict
        ekf_trainer.update(chi_kp1=x_true[k+1], chi_k=x_true[k], x_hat_previous=x_hat_ekf[k], u_input=u_current)
        x_state_for_z_ekf = np.copy(x_hat_ekf[k]); x_state_for_z_ekf[0] = x_true[k][0]
        x_hat_ekf[k+1, 0] = RHONN_predict(x_state_for_z_ekf, ekf_trainer.weights[0], u_current)
        x_hat_ekf[k+1, 1] = RHONN_predict(x_state_for_z_ekf, ekf_trainer.weights[1], u_current)
        x_hat_ekf[k+1, 2] = RHONN_predict(x_state_for_z_ekf, ekf_trainer.weights[2], u_current)

        # 4) UKF update + predict
        ukf_trainer.update(chi_kp1=x_true[k+1], chi_k=x_true[k], x_hat_previous=x_hat_ukf[k], u_input=u_current)
        x_state_for_z_ukf = np.copy(x_hat_ukf[k]); x_state_for_z_ukf[0] = x_true[k][0]
        x_hat_ukf[k+1, 0] = RHONN_predict(x_state_for_z_ukf, ukf_trainer.weights[0], u_current)
        x_hat_ukf[k+1, 1] = RHONN_predict(x_state_for_z_ukf, ukf_trainer.weights[1], u_current)
        x_hat_ukf[k+1, 2] = RHONN_predict(x_state_for_z_ukf, ukf_trainer.weights[2], u_current)

        # 5) PF update + predict
        pf_trainer.update(chi_kp1=x_true[k+1], chi_k=x_true[k], x_hat_previous=x_hat_pf[k], u_input=u_current)
        pfW = pf_trainer.get_estimate()
        x_state_for_z_pf = np.copy(x_hat_pf[k]); x_state_for_z_pf[0] = x_true[k][0]
        x_hat_pf[k+1, 0] = RHONN_predict(x_state_for_z_pf, pfW[0], u_current)
        x_hat_pf[k+1, 1] = RHONN_predict(x_state_for_z_pf, pfW[1], u_current)
        x_hat_pf[k+1, 2] = RHONN_predict(x_state_for_z_pf, pfW[2], u_current)

        if k % (n_steps // 10) == 0:
            print(f"Simulation progress: {k/n_steps*100:.1f}%")
    print("Simulation finished.")

    # ============================================================
    # 6) Results & basic stats
    # ============================================================
    mse_x_ekf = np.mean((x_true[:, 0] - x_hat_ekf[:, 0])**2)
    mse_y_ekf = np.mean((x_true[:, 1] - x_hat_ekf[:, 1])**2)
    mse_theta_ekf = np.mean((x_true[:, 2] - x_hat_ekf[:, 2])**2)
    mse_x_ukf = np.mean((x_true[:, 0] - x_hat_ukf[:, 0])**2)
    mse_y_ukf = np.mean((x_true[:, 1] - x_hat_ukf[:, 1])**2)
    mse_theta_ukf = np.mean((x_true[:, 2] - x_hat_ukf[:, 2])**2)
    mse_x_pf = np.mean((x_true[:, 0] - x_hat_pf[:, 0])**2)
    mse_y_pf = np.mean((x_true[:, 1] - x_hat_pf[:, 1])**2)
    mse_theta_pf = np.mean((x_true[:, 2] - x_hat_pf[:, 2])**2)
    print("\n--- Performance Comparison (MSE) ---")
    print(f"EKF:  x {mse_x_ekf:.6e}  y {mse_y_ekf:.6e}  th {mse_theta_ekf:.6e}")
    print(f"UKF:  x {mse_x_ukf:.6e}  y {mse_y_ukf:.6e}  th {mse_theta_ukf:.6e}")
    print(f"PF :  x {mse_x_pf:.6e}  y {mse_y_pf:.6e}  th {mse_theta_pf:.6e}")
    
    # Controller performance metrics
    if args.use_controller and controller is not None:
        # Trajectory tracking errors
        tracking_error_x = np.mean((desired_trajectory[:-1, 0] - x_true[:-1, 0])**2)
        tracking_error_y = np.mean((desired_trajectory[:-1, 1] - x_true[:-1, 1])**2)
        tracking_error_theta = np.mean(np.array([
            np.arctan2(np.sin(desired_trajectory[k, 2] - x_true[k, 2]), 
                      np.cos(desired_trajectory[k, 2] - x_true[k, 2]))**2 
            for k in range(n_steps-1)
        ]))
        
        # Control effort metrics
        control_effort_v = np.mean(control_signals[:-1, 0]**2)
        control_effort_w = np.mean(control_signals[:-1, 1]**2)
        control_variation = np.mean(np.diff(control_commands[:-1], axis=0)**2)
        
        print(f"\n--- Controller Performance ---")
        print(f"Trajectory tracking MSE: x {tracking_error_x:.6e}, y {tracking_error_y:.6e}, θ {tracking_error_theta:.6e}")
        print(f"Control effort: linear {control_effort_v:.4f}, angular {control_effort_w:.4f}")
        print(f"Control smoothness (variation): {control_variation:.6e}")
        print(f"Trajectory type: {args.controller_trajectory}")

    # Optional: build animation
    if args.animate and not args.no_plots:
        out = build_animation(t_history, x_true, x_hat_ekf, x_hat_ukf, x_hat_pf,
                              stride=max(1, args.frames_stride), out_html="robot_animation.html")
        print(f"Saved animation to {out}")

if __name__ == "__main__":
    main()
