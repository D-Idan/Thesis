import numpy as np
import matplotlib.pyplot as plt

# Parameters
np.random.seed(42)
A = 3
B = 1
T = 10
delta_ts = [0.01, 0.1, 0.5]  # Time steps to compare
delta_ts = [0.1, 0.5]  # Time steps to compare
dt_min = 0.001  # Finest resolution
dt_min = min(delta_ts)  # Finest resolution

# Generate base noise vector at dt_min
n_steps_base = int(T / dt_min)
epsilon_fine = np.random.randn(n_steps_base)  # ~N(0,1)
noise_fine = B * np.sqrt(dt_min) * epsilon_fine  # Correctly scaled


def simulate_sde(dt, noise_fine, dt_min):
    k = int(dt / dt_min)
    n_steps = int(T / dt)

    # Aggregate noise for coarse Δt
    noise_coarse = noise_fine[:n_steps * k].reshape(n_steps, k).sum(axis=1)

    # Simulate X
    X = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):
        X[i] = X[i - 1] * (1 - A * dt) + noise_coarse[i - 1]
    t = np.linspace(0, T, n_steps + 1)
    return X, t


# Plotting
plt.figure(figsize=(10, 6))

# Plot for original (incorrect) discretization
for dt in delta_ts:
    n_steps = int(T / dt)
    shift = int(dt / dt_min)
    epsilon = epsilon_fine[::shift]
    X_wrong = np.zeros(n_steps + 1)
    for i in range(1, n_steps + 1):
        X_wrong[i] = X_wrong[i - 1] * (1 - A * dt) + B * dt * epsilon[i - 1]
    t_wrong = np.linspace(0, T, n_steps + 1)
    plt.plot(t_wrong, X_wrong, '--', alpha=0.5, label=f'Wrong (dt={dt})')

# Plot for corrected discretization
for dt in delta_ts:
    X_correct, t_correct = simulate_sde(dt, noise_fine, dt_min)
    plt.plot(t_correct, X_correct, '.-', markersize=4, label=f'Correct (dt={dt})')

plt.xlabel("Time")
plt.ylabel("X(t)")
plt.title("SDE Solution: Effect of Noise Scaling")
plt.legend()
plt.grid(True)
plt.show()