class ExtendedKalmanFilter:
    def __init__(self, A, B, R, initial_state, initial_covariance):
        self.A = A  # Drift coefficient
        self.B = B  # Noise coefficient
        self.R = R  # Measurement noise covariance
        self.state = initial_state  # Initial state estimate (x_posterior)
        self.covariance = initial_covariance  # Initial covariance (P_posterior)

    def predict(self, delta_t):
        F = 1 - self.A * delta_t  # State transition Jacobian
        Q = (self.B * delta_t) ** 2  # Process noise covariance
        x_prior = F * self.state
        P_prior = F * self.covariance * F + Q
        return x_prior, P_prior

    def update(self, z, delta_t):
        # Predict step
        x_prior, P_prior = self.predict(delta_t)
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