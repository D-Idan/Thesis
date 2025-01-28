
from torch import tensor
import torch
from torch.autograd.functional import jacobian
from functools import partial
import matplotlib.pyplot as plt

class EKF:
     
    def __init__(self, motion_model, measurement_model, a, b,
                 x0, p0, Q, R, m, n):
        
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.a = a
        self.b = b
        self.Q = Q
        self.R = R
        self.m = m
        self.n = n
        self.initialize(x0, p0)
    
    def initialize(self, x0, p0):
        self.x_posterior = tensor(x0)
        self.p_posterior = tensor(p0)
        self.z = 0

    def diffential(self, eq, inputs):
        return jacobian(eq, inputs)
    
    def predictor(self):
        self.x_prior = self.motion_model(self.x_posterior, Q=self.Q)
        
        F = self.diffential(partial(self.motion_model, Q=self.Q), self.x_posterior)
        self.p_prior = F @ self.p_posterior @ F.T + self.Q
        
    
    def corrector(self,z):

        h_x_prior = self.measurement_model(self.x_prior,self.a,self.b, R=self.R)
        y_innov = z - h_x_prior
        
        H = self.diffential(partial(self.measurement_model, a=self.a, b=self.b, R=self.R), self.x_prior)
        S = H @ self.p_prior.view(H.size()) @ H.T + self.R
        S_inv = torch.inverse(S)
        KG = self.p_prior @ H.T @ S_inv
        
        self.x_posterior = self.x_prior + KG @ y_innov
        self.p_posterior = (torch.eye(self.n) - KG @ H) @ self.p_prior
    
    def kalman_step(self,z):
        
        self.predictor()
        self.corrector(z)
        


class IEKF(EKF):
    def __init__(self, motion_model, measurement_model, a, b, x0, p0, Q, R, m, n, i):
        super().__init__(motion_model, measurement_model, a, b, x0, p0, Q, R, m, n)

        self.i = i

    def corrector(self, z):

        x_0 = self.x_prior
        p_0 = self.p_prior
        h_x_prior = self.measurement_model(x_0, self.a, self.b, R=self.R)
        x_k = x_0
        for _ in range(self.i):
            H_k = self.diffential(partial(self.measurement_model, a=self.a, b=self.b, R=self.R), x_k)
            S = H_k @ p_0.view(H_k.size()) @ H_k.T + self.R
            S_inv = torch.inverse(S)
            KG_k = p_0 @ H_k.T @ S_inv

            h_x_prior = self.measurement_model(x_k, self.a, self.b, R=self.R)
            y_innov = z - h_x_prior

            x_k_1 = x_0 + KG_k @ (y_innov - self.diffential(partial
                                                            (self.measurement_model,a=self.a, b=self.b, R=self.R),
                                                            x_k) @
                                  (x_0-x_k) )

            x_k = x_k_1

            p_k_1 = (torch.eye(self.n) - KG_k @ H_k) @ self.p_prior

        self.x_posterior = x_k
        self.p_posterior = p_k_1



from numpy.random import choice
from scipy.stats import norm
from plots import plot_test, par2time_plot


class Particle_Filter:

    def __init__(self, motion_model, measurement_model, a, b, x0, p0, Q, R, m, n,N):
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.a = a
        self.b = b
        self.Q = Q
        self.R = R
        self.m = m
        self.n = n
        self.N = N
        self.iteration = 0
        self.timeI_particles = []
        self.initialize(x0, p0)


    def initialize(self, x0, p0):
        self.particles = -10 + torch.rand(self.N, dtype=torch.float64) * 20
        self.weights = torch.ones(self.N) / self.N

        # $$$$$$$$$$$$$$$$$$$$$$$$
        # self.particles = torch.range(-10,11)
        # self.weights = torch.ones(self.particles.size(0)) / self.particles.size(0)
        # self.timeI_particles.append(self.particles)
        # # plot_test(self.particles, self.weights)
        # $$$$$$$$$$$$$$$$$$$$$$$$

        self.x_posterior = tensor(x0)
        self.p_posterior = tensor(p0)

    def resample(self, z):

        self.particles = tensor(choice(a=self.particles, size=self.N, replace=True, p=self.weights))
        # self.particles = self.particles + torch.sqrt(self.Q) * torch.randn(self.N)
        # spred = -6 + torch.rand(self.N//2, dtype=torch.float64) * 12
        # self.particles = torch.cat((self.particles, spred))

    def kalman_step(self,z):
        # Apply Motion Model
        self.particles = self.motion_model(self.particles, Q=self.Q, noise=True)

        # Prior
        self.x_prior = self.particles[torch.argmax(self.weights)]
        self.p_prior = self.weights.max()

        # Apply Measurement Model
        measure_particles = self.measurement_model(self.particles, self.a, self.b, R=self.R, noise=True)
        # # Likelihood to weights
        # norm.pdf ( it is the probability density function of the normal distribution)
        self.weights = tensor(norm.pdf(measure_particles, loc=z, scale=10))
        # ## Weights Normalized
        self.weights = torch.nn.functional.normalize(self.weights, dim=0, p=1)

        # self.particles[abs(self.particles - z) <= 100] = 0.0
        # x = torch.ones(self.particles.size())
        # plt.scatter(measure_particles, self.weights, c='green', label='particles')
        # plt.scatter(z, 0, c='r', label='measurement')
        # plt.xlabel('value')
        # plt.ylabel('weight')
        # plt.legend()
        # self.iteration += 1
        # plt.title(f'Iteration number: {self.iteration}')
        # # plt.show()
        # # plt.cla()
        # Posterior
        self.x_posterior = torch.matmul(self.weights, self.particles.t())
        self.p_posterior = tensor(norm.pdf(self.x_posterior, loc=z, scale=1))

        # Resample
        self.resample(z)



