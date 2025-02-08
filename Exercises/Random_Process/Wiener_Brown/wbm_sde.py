#!/usr/bin/env python
"""
Student Assignment:
Simulating the SDE: dX/dt = -A*X + dW/dt
using Forward Euler discretization.
This code generates trajectories for different delta_t values
and demonstrates convergence issues when using large delta_t.
For higher delta_t, we sub-sample the increments from a high-resolution Wiener process.
"""

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
A = 3.0  # Drift coefficient
T = 10.0  # Total time in seconds
# Choose a set of delta_t values; the smallest one is used for the full Wiener process.
# delta_t_values = [0.001, 0.005, 0.01, 0.05, 0.1]
delta_t_values = [0.0001, 0.1]
delta_t_values = [0.4, 0.1, 4e-02, 4e-03]

# Choose the smallest dt to generate the full Brownian motion path
dt_min = min(delta_t_values)
num_steps_min = int(T / dt_min)

# Generate one Wiener process sample path (increments)
# The Wiener increments: sqrt(dt)*N(0,1)
# We generate increments for dt_min and then form the cumulative sum for the Wiener path
dW_full = np.sqrt(dt_min) * np.random.randn(num_steps_min)
W_full = np.concatenate(([0], np.cumsum(dW_full)))

# Time vector for the high-resolution path
time_full = np.linspace(0, T, num_steps_min + 1)


def simulate_SDE(delta_t, A, T, dW_full, dt_min):
    """
    Simulate the SDE using the Euler method.

    For delta_t == dt_min, we use the dW_full directly.
    For delta_t > dt_min, we sub-sample the dW_full array according to the ratio.
    """
    # Determine the ratio between the desired dt and the minimum dt
    k_ratio = int(delta_t / dt_min)
    print(f"delta_t: {delta_t}, dt_min: {dt_min}, ratio: {k_ratio}")
    if delta_t < dt_min or delta_t % dt_min > 10**-5:
        raise ValueError("delta_t must be an integer multiple of the minimum dt.")

    # Compute number of steps
    num_steps = int(T / delta_t)

    # Sub-sample the Wiener increments:
    # For each time step of size delta_t, sum up the corresponding increments from dW_full.
    dW = np.array([np.sum(dW_full[i * k_ratio:(i + 1) * k_ratio]) for i in range(num_steps)])
    ################################# Explanation START ###############################
    # dW = np.zeros(num_steps)
    #
    # # Loop over each time step
    # for i in range(num_steps):
    #     # Calculate the start and end indices for the current sub-sample
    #     start_idx = i * k_ratio
    #     end_idx = (i + 1) * k_ratio
    #
    #     # Sum the increments from dW_full for the current sub-sample
    #     dW[i] = np.sum(dW_full[start_idx:end_idx])
    ################################# Explanation END #################################

    # Initialize state array
    X = np.zeros(num_steps + 1)
    # Forward Euler integration for the SDE:
    # X[i] = X[i-1]*(1 - A*delta_t) + (W(t_i)-W(t_{i-1})) i.e., dW[i-1]
    for i in range(1, num_steps + 1):
        X[i] = X[i - 1] * (1 - A * delta_t) + dW[i - 1]

    # Time vector for the current delta_t simulation
    t = np.linspace(0, T, num_steps + 1)
    return t, X


# Simulate for each delta_t and store results
trajectories = {}
for dt in delta_t_values:
    t, X = simulate_SDE(dt, A, T, dW_full, dt_min)
    trajectories[dt] = (t, X)

# Plot all trajectories together
plt.figure(figsize=(10, 6))
for dt in sorted(trajectories.keys()):
    t, X = trajectories[dt]
    plt.plot(t, X, label=f'Δt = {dt}')
plt.title('Trajectories of the SDE for different Δt values')
plt.xlabel('Time [s]')
plt.ylabel('X(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("SDE_Trajectories.png")
plt.show()


# Discussion:
#
# When we simulate the SDE with different Δt values, we observe that for very coarse
# time steps (e.g., Δt = 0.1), the discrete solution deviates noticeably from the solution
# obtained with finer discretizations (e.g., Δt = 0.001). This is due to the numerical error
# inherent in the Euler–Maruyama method which is of order O(Δt).
#
# To ensure consistency between simulations with different Δt values, we use the same
# high-resolution Wiener process (generated with dt_min) and sub-sample it appropriately for
# larger Δt values. This way, the only difference between trajectories is the numerical error,
# not the randomness. However, if one needs accurate convergence, one should:
#
# 1. Use sufficiently small Δt values.
# 2. Consider higher order schemes (like the Milstein method) or, when available,
#    use an exact solution (for the Ornstein-Uhlenbeck process the exact update is known).
#
# Exact update for the Ornstein-Uhlenbeck process:
#   X(t + Δt) = X(t)*exp(-A Δt) + sqrt((1-exp(-2A Δt))/(2A))*η,   η ~ N(0,1)
#
# Implementing the exact update can greatly improve convergence even with larger Δt.

# Example: Implementing the exact update for comparison
def simulate_exact_SDE(A, T, delta_t, dW_full, dt_min):
    """
    Simulate the Ornstein-Uhlenbeck process using the exact update.
    Note: Although we don't strictly need the Wiener process for the exact simulation,
    we demonstrate the idea by computing the correct variance for the noise term.
    """
    num_steps = int(T / delta_t)
    t = np.linspace(0, T, num_steps + 1)
    X = np.zeros(num_steps + 1)
    # Precompute constants
    exp_factor = np.exp(-A * delta_t)
    noise_std = np.sqrt((1 - np.exp(-2 * A * delta_t)) / (2 * A))
    # Use our high-resolution Wiener increments (sub-sampled) only to generate a consistent noise
    ratio = int(delta_t / dt_min)
    dW = np.array([np.sum(dW_full[i * ratio:(i + 1) * ratio]) for i in range(num_steps)])
    # Instead of dW, we now use independent standard normals scaled by noise_std.
    # (They are not exactly the same as dW, but this shows the idea of the exact scheme.)
    eta = np.random.randn(num_steps)
    for i in range(1, num_steps + 1):
        X[i] = X[i - 1] * exp_factor + noise_std * eta[i - 1]
    return t, X


# Simulate the exact solution for delta_t = 0.1 for comparison
t_exact, X_exact = simulate_exact_SDE(A, T, 0.1, dW_full, dt_min)

plt.figure(figsize=(10, 6))
plt.plot(trajectories[0.1][0], trajectories[0.1][1], 'r--', label='Euler (Δt = 0.1)')
plt.plot(t_exact, X_exact, 'b-', label='Exact update (Δt = 0.1)')
plt.title('Comparison: Euler vs. Exact Update for Δt = 0.1')
plt.xlabel('Time [s]')
plt.ylabel('X(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
Summary:
-----------
- We simulated the SDE using Euler–Maruyama with different time steps Δt.
- The same high-resolution Wiener process was used (with Δt = 0.001) and sub-sampled for coarser grids.
- The plots demonstrate that for larger Δt the solution deviates from that obtained with finer discretization.
- To resolve convergence issues when using large Δt, one should use smaller Δt or implement an exact or
  higher order method (like the exact update for the Ornstein-Uhlenbeck process shown above).

This code is submitted as part of my thesis assignment.
"""
