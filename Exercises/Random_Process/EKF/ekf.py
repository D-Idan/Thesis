import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, A, B, R, initial_state, initial_covariance):
        self.A = A  # Drift coefficient
        self.B = B  # Noise coefficient
        self.R = R  # Measurement noise covariance
        self.Q = 1  # Process noise covariance
        self.state = initial_state  # Initial state estimate (x_posterior)
        self.covariance = initial_covariance  # Initial covariance (P_posterior)
        self.jacobian = lambda dt: 1 - self.A * dt
        self.motion_model_step = lambda x, dt, n: (1 - self.A * dt) * x + self.B * dt * n

    def predict(self, delta_t, n = None, norm_noise=None):
        if n is None:
            n = np.random.randn()
        if norm_noise:
            # Q = (self.B) ** 2 * np.sqrt(delta_t)  # Normalized process noise covariance
            # Q = (self.B) ** 2 * delta_t # Because the noise is already normalized
            Q = (self.B * delta_t) ** 2
            n = n / np.sqrt(delta_t)
        else:
            Q = (self.B * delta_t) ** 2    # Process noise covariance

        self.Q = Q

        F = self.jacobian(delta_t)  # State transition Jacobian
        self.x_prior = self.motion_model_step(self.state, delta_t, n)
        self.P_prior = F * self.covariance * F + Q
        return self.x_prior, self.P_prior

    def update(self, z):
        # Predict step
        x_prior, P_prior = self.x_prior, self.P_prior
        H = 1.0  # Measurement Jacobian
        # Innovation
        y = z - H * x_prior
        # Innovation covariance
        S = H * P_prior * H + self.R
        # Kalman gain
        K = (P_prior * H) / S
        # Update state
        x_posterior = x_prior + K * y
        # Update covariance
        P_posterior = (1 - K * H) * P_prior
        self.state = x_posterior
        self.covariance = P_posterior
        return x_posterior, P_posterior