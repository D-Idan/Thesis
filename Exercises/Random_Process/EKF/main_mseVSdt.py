import numpy as np
from kneed import KneeLocator
import matplotlib.pyplot as plt
from ekf import ExtendedKalmanFilter
from simulation import simulate_true_state, generate_measurements

# ---------------------------
# Simulation Parameters
# ---------------------------
A = 3.0
B = 1.0
T = 100.0

# List of delta_t values to test
delta_t_values = np.arange(0.0001, 0.6, 0.002)
# We choose one measurement noise variance R for the experiment
R = 0.1

initial_state = 1.0
initial_covariance = 1.0
norm_noise = True
use_rolling_average_measurements = False  # Change to True if needed

# ---------------------------
# Precompute noise arrays based on the smallest delta_t
# ---------------------------
min_delta_t = min(delta_t_values)
max_steps = int(T / min_delta_t)
np.random.seed(42)  # Ensure reproducibility
Nn_true_full = np.random.randn(max_steps)
Nn_measurements_full = np.random.randn(max_steps)

# Dictionary to store simulation results keyed by delta_t
results = {}

for delta_t in delta_t_values:
    # Determine the number of steps and subsample the noise arrays.
    n_steps = int(T / delta_t)
    shift = int(delta_t / min_delta_t)

    current_Nn_true = Nn_true_full[::shift][:n_steps]
    current_Nn_measurements = Nn_measurements_full[::shift][:n_steps]
    if norm_noise:
        current_Nn_true = current_Nn_true / np.sqrt(delta_t)
        current_Nn_measurements = current_Nn_measurements / np.sqrt(delta_t)

    # Generate the true state using the given delta_t
    X_true = simulate_true_state(A, B, delta_t, T, current_Nn_true)
    # Keep a copy of the unmodified true state (used for measurement generation)
    X_true_orig = X_true.copy()
    time_inx = np.linspace(0, T, len(X_true))

    # Generate measurements from the true state
    measurements = generate_measurements(X_true_orig, R)

    # For simplicity, we use the same R value in the EKF (adjust if using rolling average)
    R_kf = R
    if use_rolling_average_measurements:
        R_kf = R_kf / shift

    # Initialize the Extended Kalman Filter
    ekf = ExtendedKalmanFilter(
        A=A,
        B=B,
        R=R_kf,
        initial_state=initial_state,
        initial_covariance=initial_covariance
    )

    # Arrays to store the EKF estimates
    x_posterior_array = np.zeros_like(X_true)
    x_posterior_array[0] = initial_state

    # Run the filter over the simulation
    for i in range(1, len(X_true)):
        x_prior, p_prior = ekf.predict(delta_t, current_Nn_true[i - 1], norm_noise=norm_noise)
        # Use measurement if available
        if i - 1 < len(measurements):
            z = measurements[i - 1]
            x_posterior, p_posterior = ekf.update(z)
        else:
            x_posterior, p_posterior = x_prior, p_prior
        x_posterior_array[i] = x_posterior

    # Save results for this delta_t
    results[delta_t] = {
        'time': time_inx,
        'x_posterior': x_posterior_array,
        'X_true': X_true  # true state computed at this delta_t
    }

# ---------------------------
# Compute MSE relative to the ground truth (delta_t = min_delta_t)
# ---------------------------
gt = results[min_delta_t]
gt_time = gt['time']
gt_true = gt['X_true']

dt_list = []  # To hold the delta_t values (x-axis)
mse_list = []  # To hold the corresponding MSE values (y-axis)

for dt, res in results.items():
    times = res['time']
    est = res['x_posterior']

    # For each simulation time in the current run, find the corresponding "true" state
    # from the ground-truth simulation. Since the ground truth is computed with min_delta_t,
    # the index for a given time t is int(t/min_delta_t).
    gt_aligned = []
    for t in times:
        idx = int(round(t / min_delta_t))
        # Protect against index out-of-range due to rounding:
        idx = min(idx, len(gt_true) - 1)
        gt_aligned.append(gt_true[idx])
    gt_aligned = np.array(gt_aligned)

    # Compute mean squared error between the EKF estimate and the ground truth at matching times.
    mse = np.mean((est - gt_aligned) ** 2)
    mse_list.append(mse)
    dt_list.append(dt)
    print(f"delta_t = {dt:.3f}, MSE = {mse:.6f}")

# ---------------------------
# Plot log-log graph: log(delta_t) vs. log(MSE)
# ---------------------------
plt.figure(figsize=(8, 6))
plt.loglog(dt_list[1:], mse_list[1:], marker='o', linestyle='-', color='b')
plt.xlabel('delta_t (log scale)')
plt.ylabel('MSE (log scale)')
plt.title('MSE vs. delta_t (Ground Truth from delta_t = {:.4f})'.format(min_delta_t))
plt.grid(True, which="both", ls="--")
plt.show()
