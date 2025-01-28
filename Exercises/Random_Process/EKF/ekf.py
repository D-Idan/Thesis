import torch
from torch.autograd.functional import jacobian
from functools import partial

class EKF:
    def __init__(self, motion_model, measurement_model, a, b, x0, p0, Q, R, m, n):
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.a = a
        self.b = b
        self.Q = torch.tensor([Q])
        self.R = R
        self.m = m
        self.n = n
        p0 = torch.tensor([p0])
        self.initialize(x0, p0)

    def initialize(self, x0, p0):
        self.x_posterior = torch.tensor(x0, dtype=torch.float32)
        self.p_posterior = torch.tensor(p0, dtype=torch.float32)

    def diffential(self, eq, inputs):
        return jacobian(eq, inputs)

    def predictor(self):
        self.x_prior = self.motion_model(self.x_posterior, Q=self.Q)
        F = self.diffential(partial(self.motion_model, Q=self.Q), self.x_posterior)
        self.p_prior = torch.tensor([F @ self.p_posterior]) @ F.T + self.Q

    def corrector(self, z):
        h_x_prior = self.measurement_model(self.x_prior, self.a, self.b, R=self.R)
        y_innov = z - h_x_prior
        H = self.diffential(partial(self.measurement_model, a=self.a, b=self.b, R=self.R), self.x_prior)
        S = H @ self.p_prior.view(H.size()) @ H.T + self.R
        KG = self.p_prior @ H.T @ torch.inverse(S.resize(self.m, self.m))
        self.x_posterior = self.x_prior + KG @ y_innov
        self.p_posterior = (torch.eye(self.n) - KG @ H) @ self.p_prior

    def kalman_step(self, z):
        self.predictor()
        self.corrector(z)