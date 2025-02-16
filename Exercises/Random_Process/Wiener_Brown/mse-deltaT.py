import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
A = 3.0  # Drift coefficient
T = 100.0  # Total time in seconds

# List of delta_t values (each must be an integer multiple of the smallest dt)
delta_t_values = [0.4, 0.1, 0.04, 0.004]
delta_t_values = np.arange(0.0001, 0.68, 0.0002)

# The smallest delta_t will be used as the "true" simulation.
dt_min = min(delta_t_values)
num_steps_min = int(T / dt_min)

# Generate one Wiener process sample path for the highest resolution simulation.
dW_full = np.sqrt(dt_min) * np.random.randn(num_steps_min)
W_full = np.concatenate(([0], np.cumsum(dW_full)))
time_full = np.linspace(0, T, num_steps_min + 1)


def simulate_SDE(delta_t, A, T, dW_full, dt_min):
    """
    Simulate the SDE using the Euler method.

    For delta_t == dt_min, we use the dW_full directly.
    For delta_t > dt_min, we sub-sample the dW_full array according to the ratio.
    """
    # Determine the ratio between the desired dt and the minimum dt.
    k_ratio = int(delta_t / dt_min)
    if abs(delta_t - k_ratio * dt_min) > 1e-5:
        raise ValueError("delta_t must be an integer multiple of the minimum dt.")

    num_steps = int(T / delta_t)

    # For each time step of size delta_t, sum up the corresponding increments from dW_full.
    dW = np.array([np.sum(dW_full[i * k_ratio:(i + 1) * k_ratio]) for i in range(num_steps)])

    # Initialize state array; X[0] = 0.
    X = np.zeros(num_steps + 1)
    # Forward Euler integration:
    # X[i] = X[i-1]*(1 - A*delta_t) + dW[i-1]
    for i in range(1, num_steps + 1):
        X[i] = X[i - 1] * (1 - A * delta_t) + dW[i - 1]

    # Time vector for the current delta_t simulation.
    t = np.linspace(0, T, num_steps + 1)
    return t, X


# ---------------------------
# 1. Compute the "true" simulation using dt_min.
# ---------------------------
t_true, X_true = simulate_SDE(dt_min, A, T, dW_full, dt_min)

# ---------------------------
# 2. Run simulations for each delta_t and compute the MSE against the true simulation.
# ---------------------------
mse_list = []
delta_ts = []

for dt in delta_t_values:
    t_sim, X_sim = simulate_SDE(dt, A, T, dW_full, dt_min)

    # Determine how many dt_min steps correspond to this dt.
    k_ratio = int(dt / dt_min)

    # Sub-sample the true simulation at the time points of the coarser simulation.
    X_true_subsample = X_true[::k_ratio]

    min_len = min(len(X_sim), len(X_true_subsample))
    X_sim = X_sim[:min_len]
    X_true_subsample = X_true_subsample[:min_len]

    # Compute the mean squared error (MSE) between the simulation and the true values.
    mse = np.mean((X_sim - X_true_subsample) ** 2)
    mse_list.append(mse)
    delta_ts.append(dt)

    print(f"Δt = {dt:.4f} (ratio = {k_ratio}), MSE = {mse:.6e}")

# ---------------------------
# 3. Plot the log-log graph of delta_t vs. MSE.
# ---------------------------
plt.figure(figsize=(8, 6))
plt.loglog(delta_ts[1:], mse_list[1:], 'o-')
# Set labels
plt.xlabel(r'$\log(\Delta t)$')
plt.ylabel(r'$\log(\mathrm{MSE})$')
# plt.xlabel("Δt (log scale)")
# plt.ylabel("MSE (log scale)")
plt.title("Convergence of Euler–Forward for the SDE")
plt.grid(True, which="both", ls="--", alpha=0.7, label='MSE vs Δt')
plt.tight_layout()
plt.savefig("SDE_LogLog_MSE.png")
plt.show()
