import torch

from ekf import EKF
from model import simulate_system
from plotter import plot_results

# Define parameters
A = 3
B = 1
T = 10
delta_t = 0.1
initial_state = 0
measurement_noise_variance = 0.5

# Initialize EKF
Q = torch.tensor([[0.1]])
R = torch.tensor([[measurement_noise_variance]])
ekf = EKF(
    motion_model=lambda x, Q: x * (1 + delta_t * A) + delta_t * B * torch.randn(1),
    measurement_model=lambda x, a, b, R: x + torch.sqrt(R) * torch.randn(1),
    a=A,
    b=B,
    x0=0,
    p0=1,
    Q=Q,
    R=R,
    m=1,
    n=1,
)

# Simulate the system
time, x_true, noise = simulate_system(A, B, T, delta_t)
x_prior_array = []
x_posterior_array = []
p_prior_array = []
p_posterior_array = []

for i in range(len(time)):
    ekf.kalman_step(torch.tensor(x_true[i]))
    x_prior_array.append(ekf.x_prior.item())
    x_posterior_array.append(ekf.x_posterior.item())
    p_prior_array.append(ekf.p_prior.item())
    p_posterior_array.append(ekf.p_posterior.item())

# Plot results
plot_results(time, x_true, x_prior_array, x_posterior_array, p_prior_array, p_posterior_array, A, B)