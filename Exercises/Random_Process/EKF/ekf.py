import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, A, B, Q, R, delta_t):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.delta_t = delta_t
        self.x_pred = 0
        self.p_pred = 10  # Initial covariance
        self.x_cor = 0
        self.p_cor = 10

    def predict(self):
        # Predict step
        self.x_pred = self.x_cor - self.A * self.x_cor * self.delta_t
        self.p_pred = self.p_cor + self.Q

    def correct(self, z):
        # Update step
        K = self.p_pred / (self.p_pred + self.R)
        self.x_cor = self.x_pred + K * (z - self.x_pred)
        self.p_cor = (1 - K) * self.p_pred

    def step(self, z):
        self.predict()
        self.correct(z)
        return self.x_pred, self.x_cor, self.p_pred, self.p_cor
