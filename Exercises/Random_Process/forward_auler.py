# import numpy as np
# import matplotlib.pyplot as plt
#
# # Parameters
# A = 3         # Drift coefficient
# B = 1         # Coefficient for U(t)
# U = lambda t: 1  # Step function (constant value)
# T_max = 5      # Total simulation time
# X0 = 0         # Initial value
#
# # Function to compute X(t) using Forward Euler
# def simulate(A, B, U, delta_t, T_max, X0):
#     time_steps = int(T_max / delta_t)
#     t = np.linspace(0, T_max, time_steps)
#     X = np.zeros(time_steps)
#     X[0] = X0
#     for i in range(1, time_steps):
#         X[i] = X[i-1] * (1 + delta_t * A) + delta_t * B * U(t[i-1])
#     return t, X
#
# # Different delta_t values
# delta_t_values = [0.0001, 0.01, 0.05, 0.2, 0.5]
#
# # Plot results for different delta_t
# plt.figure(figsize=(12, 8))
# for delta_t in delta_t_values:
#     t, X = simulate(A, B, U, delta_t, T_max, X0)
#     plt.plot(t, X, label=f"delta_t = {delta_t}")
#
# # Add plot details
# plt.title("Forward Euler Simulation for Different delta_t Values")
# plt.xlabel("Time (t)")
# plt.ylabel("X(t)")
# plt.axhline(y=(1/A), color="gray", linestyle="--", label="Steady State (1/A)")
# plt.legend()
# plt.grid()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

def forward_euler(A, B, U, delta_t, T):
    t = np.arange(0, T + delta_t, delta_t)
    X = np.zeros_like(t)

    for i in range(1, len(t)):
        X[i] = X[i - 1] * (1 + delta_t * A) + delta_t * B * U(t[i])

    return t, X

def runge_kutta_4(A, B, U, delta_t, T):
    t = np.arange(0, T + delta_t, delta_t)
    X = np.zeros_like(t)

    for i in range(1, len(t)):
        k1 = delta_t * (A * X[i - 1] + B * U(t[i - 1]))
        k2 = delta_t * (A * (X[i - 1] + 0.5 * k1) + B * U(t[i - 1] + 0.5 * delta_t))
        k3 = delta_t * (A * (X[i - 1] + 0.5 * k2) + B * U(t[i - 1] + 0.5 * delta_t))
        k4 = delta_t * (A * (X[i - 1] + k3) + B * U(t[i - 1] + delta_t))

        X[i] = X[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t, X

# Parameters
A = -3  # Drift coefficient (negative for stability)
B = 1   # Step response magnitude
U = lambda t: 1 if t >= 0 else 0  # Step function
T = 5   # Total simulation time in seconds

# Test different delta_t values
delta_t_values = [0.01, 0.05, 0.1, 0.5, 1]

plt.figure(figsize=(10, 6))
for delta_t in delta_t_values:
    t_fe, X_fe = forward_euler(A, B, U, delta_t, T)
    plt.plot(t_fe, X_fe, label=f"Forward Euler (delta_t={delta_t})")

# Add Runge-Kutta 4 (stable reference)
delta_t_rk4 = 0.01
t_rk4, X_rk4 = runge_kutta_4(A, B, U, delta_t_rk4, T)
plt.plot(t_rk4, X_rk4, 'k--', label="Runge-Kutta 4 (delta_t=0.01, reference)")

# Plot steady-state line (X_ss = 1/A)
steady_state = -B / A
plt.axhline(steady_state, color='gray', linestyle='--', label="Steady State (1/A)")

# Plot settings
plt.title("Numerical Solution for Different delta_t Values")
plt.xlabel("Time (t)")
plt.ylabel("X(t)")
plt.legend()
plt.grid()
plt.show()
