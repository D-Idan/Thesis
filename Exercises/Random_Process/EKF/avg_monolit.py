import numpy as np
import matplotlib.pyplot as plt

seed = 17
np.random.seed(seed)

class ExtendedKalmanFilter:
    def __init__(self, A, B, R, initial_state, initial_covariance):
        self.A = A  # Drift coefficient
        self.B = B  # Noise coefficient
        self.R = R  # Measurement noise covariance
        self.Q = 1  # Process noise covariance
        self.state = initial_state  # Initial state estimate (x_posterior)
        self.covariance = initial_covariance  # Initial covariance (P_posterior)

    def predict(self, delta_t):
        self.Q = (self.B) ** 2 * delta_t
        F = 1 - self.A * delta_t  # State transition Jacobian
        self.x_prior = (1 - self.A * delta_t) * self.state + self.Q * np.random.randn()
        self.P_prior = F * self.covariance * F + self.Q
        self.state = self.x_prior
        self.covariance = self.P_prior
        return self.x_prior, self.P_prior

    def update(self, z):
        # Predict step
        x_prior, P_prior = self.x_prior, self.P_prior
        H = 1.0  # Measurement Jacobian
        # Innovation
        y = z - H * x_prior
        # Innovation covariance
        S = H * P_prior * H + self.R
        # Kalman gain
        K = (P_prior * H) / S
        # Update state
        x_posterior = x_prior + K * y
        # Update covariance
        P_posterior = (1 - K * H) * P_prior
        self.state = x_posterior
        self.covariance = P_posterior
        return x_posterior, P_posterior


A = 3.0
B = 1.0
T = 5.0

# Use the lowest delta_t value (finest resolution)
PRF = 2000
delta_t = 1/PRF  # Adjust as needed; this is our "finest" dt for this study.
R = 0.00001
initial_state = 1.0
initial_covariance = 1.0
transient = 0.2
norm_noise = True
norm_R = True
one_prediction_step = True

# List of averaging window sizes to test.
avg_window_list = np.unique(np.logspace(0.1, 2.0, num=100, dtype=int))

# Pre-generate noise arrays at the finest resolution.
max_steps = int(T / delta_t)
Nn_true = np.random.randn(max_steps)
Nn_measurements = np.random.randn(max_steps)
if norm_noise:
    Nn_true = Nn_true / np.sqrt(delta_t)
    Nn_measurements = Nn_measurements / np.sqrt(delta_t)

# Simulate the true state at every delta_t step.
X_true = np.zeros(max_steps + 1)
X_true[0] = 0.0  # Initial condition
for i in range(1, max_steps + 1):
    X_true[i] = X_true[i-1] * (1 - A * delta_t) + B * delta_t * Nn_true[i-1]

# Generate measurements at every delta_t step.
measurements = X_true[1:] + Nn_measurements * np.sqrt(R)

# --------------------------
# Run loop for each averaging window size.
# --------------------------
mse_prior_list = []
mse_post_list = []
p_prior_list = []
p_post_list = []
results_all = {}

for avg_window in avg_window_list:

    # Adjust the measurement noise covariance if desired.
    R_kf = R if not norm_R else R / (avg_window * delta_t)

    # Initialize the EKF.
    ekf = ExtendedKalmanFilter(
        A=A,
        B=B,
        R=R_kf,
        initial_state=initial_state,
        initial_covariance=initial_covariance
    )


    x_prior_array = []
    x_posterior_array = []
    p_prior_array = []
    p_post_array = []
    time_inx = []


    for kalman_idx in range(max_steps // avg_window):
        # Predict step
        if one_prediction_step:
            x_prior, p_prior = ekf.predict(avg_window * delta_t)
        else:
            x_prior, p_prior = [ekf.predict(delta_t) for _ in range(avg_window)][-1]

        # Update step
        idx_start = kalman_idx * avg_window
        idx_end = (kalman_idx + 1) * avg_window
        z = np.mean(measurements[idx_start:idx_end])
        x_posterior, p_posterior = ekf.update(z)

        x_prior_array.append(x_prior)
        x_posterior_array.append(x_posterior)
        p_prior_array.append(p_prior)
        p_post_array.append(p_posterior)
        time_inx.append(idx_end)

    # Save the results for this averaging window size.
    transient_steps = int(transient * len(p_prior_array))
    result = {
        'time': np.array(time_inx)[transient_steps:],
        'x_prior': np.array(x_prior_array)[transient_steps:],
        'x_posterior': np.array(x_posterior_array)[transient_steps:],
        'p_prior': np.array(p_prior_array)[transient_steps:],
        'p_posterior': np.array(p_post_array)[transient_steps:],
        'X_true': X_true[time_inx][transient_steps:],
    }
    results_all[avg_window] = result

    mse_prior_list.append(np.mean((result['X_true'] - result['x_prior']) ** 2))
    mse_post_list.append(np.mean((result['X_true'] - result['x_posterior']) ** 2))
    p_prior_list.append(np.mean(result['p_prior'][-1]))
    p_post_list.append(np.mean(result['p_posterior'][-1]))




# --------------------------
# Plot the MSE for Prior and Posterior as a function of the averaging window.
# Both curves are shown on the same figure.
# --------------------------
import matplotlib.ticker as ticker

plt.figure(figsize=(8, 6))

plt.plot(avg_window_list, mse_prior_list, marker='o', linestyle='-',
         label='Prior MSE')
plt.plot(avg_window_list, p_prior_list, marker='p', linestyle='-',
         label='Prior P')

plt.plot(avg_window_list, mse_post_list, marker='s', linestyle='--',
         label='Posterior MSE')
plt.plot(avg_window_list, p_post_list, marker='p', linestyle='--',
         label='Posterior P')

plt.xlabel('Averaging Window')
plt.ylabel('MSE')
plt.title('MSE vs. Averaging Window (Δt = {:.4f}, PRF = {})'.format(delta_t, PRF))
plt.xscale('log')
plt.yscale('log')
# # Get the current y-axis major ticks (which correspond to log grid lines)
# ax = plt.gca()
# ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=None))  # Ensures major ticks are at log scale
#
# # Get the tick locations
# y_ticks = ax.yaxis.get_majorticklocs()
#
# # Set the y-axis ticks explicitly
# plt.yticks(y_ticks, labels=[f"{ytick:.1e}" for ytick in y_ticks])
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.7)
plt.tight_layout()
plt.show()

from plot_results import plot_single_result
results_number = 1
if results_number not in results_all: # get the nearest key value
    results_number = min(results_all.keys(), key=lambda x:abs(x-results_number))

ran = results_all[results_number] # Results average number
plot_single_result(ran['time']/PRF, ran['X_true'], ran['x_prior'], ran['x_posterior'], ran['p_prior'], ran['p_posterior'], A, B, delta_t, R, save_path=None, extra_info=f'Sliding Window = {results_number}')

# ---------------------------
# print mean square on true state
print(f"Mean square on true state: {np.mean((ran['X_true']) ** 2)}")