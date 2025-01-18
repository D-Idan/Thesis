import numpy as np
import matplotlib.pyplot as plt


def forward_euler_stochastic(A, B, N, delta_t, T):
    t = np.arange(0, T + delta_t, delta_t)
    X = np.zeros_like(t)
    Nn = N * np.sqrt(delta_t)
    # Nn = N #* np.sqrt(delta_t)


    for i in range(1, len(t)):
        X[i] = X[i - 1] * (1 + delta_t * A) + delta_t * B * Nn[i]

    return t, X

# Test different delta_t values
# delta_t_values = [0.00001, 0.0001, 0.001, 0.01, 0.1]
delta_t_values = [0.001, 0.01, 0.1]

# Parameters
A = -3  # Drift coefficient (negative for stability)
B = 1  # Response magnitude# White noise input
T = 50  # Total simulation time in seconds
t = np.arange(0, T + min(delta_t_values), min(delta_t_values))
N = np.random.normal(0, 1, size=len(t)) # White noise input
# N = np.random.normal(0, 1, size=len(t)) *0 + 1 # White noise input
# N = np.random.normal(0, 0.0001, size=len(t)) # White noise input with variance delta_t



# plt.figure(figsize=(12, 6))
# for delta_t in delta_t_values:
#     t_fe, X_fe = forward_euler_stochastic(A, B, N, delta_t, T)
#     plt.plot(t_fe, X_fe, label=f"Forward Euler (delta_t={delta_t})")
#
# # Plot steady-state line (X_ss = 1/A)
# steady_state = -B / A
# plt.axhline(steady_state, color='gray', linestyle='--', label="Steady State (1/A)")
#
# # Plot settings
# plt.title("Stochastic Numerical Solution for Different delta_t Values")
# plt.xlabel("Time (t)")
# plt.ylabel("X(t)")
# plt.legend()
# plt.grid()
# plt.show()

# Calculate running variance for each delta_t

plt.figure(figsize=(12, 6))
final_variances = {}

for delta_t in delta_t_values:
    t_fe, X_fe = forward_euler_stochastic(A, B, N, delta_t, T)

    # Number of transient steps to discard (based on smallest delta_t)
    transient_steps = int(10.0 / delta_t)  # Discard the first 10 seconds of data

    # Discard transient steps
    valid_X = X_fe[transient_steps:]
    valid_t = t_fe[transient_steps:]

    # Check for sufficient data points
    if len(valid_X) < 2:
        print(f"Not enough data points to calculate variance for delta_t={delta_t}")
        final_variances[delta_t] = np.nan
        continue

    # Calculate running variance
    running_variance = [np.var(valid_X[:i]) for i in range(1, len(valid_X))]

    # Store final variance for comparison
    final_variances[delta_t] = np.var(valid_X)

    # Plot running variance
    plt.plot(valid_t[1:], running_variance, label=f"Running Variance (delta_t={delta_t})")

# Print final numerical variances
print("Final Variances for each delta_t:")
# Print header
print(f"{'delta_t':<10} | {'Variance':<20}")
print("-" * 32)
for delta_t, variance in final_variances.items():
    print(f"{delta_t:<10} | {variance:.16f}")

# Plot settings for variance convergence
plt.title("Running Variance Convergence for Different delta_t Values")
plt.xlabel("Time (t)")
plt.ylabel("Variance")
plt.legend()
plt.grid()
plt.show()