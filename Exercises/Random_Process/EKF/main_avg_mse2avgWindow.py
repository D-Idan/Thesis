import numpy as np
import matplotlib.pyplot as plt
from ekf import ExtendedKalmanFilter
from simulation import simulate_true_state, generate_measurements

def run_simulation_fixed_dt(A, B, T, delta_t, R, avg_window,
                            Nn_true, Nn_measurements,
                            initial_state, initial_covariance,
                            norm_noise, norm_R):
    """
    Runs one simulation/EKF filtering cycle for a given averaging window.
    Returns a dictionary containing:
      - 'time': the time indices of the averaged simulation,
      - 'X_true': the averaged true state,
      - 'x_prior': the EKF prior estimates,
      - 'x_posterior': the EKF posterior estimates.
    """
    # Total simulation steps at this delta_t:
    n_steps = int(T / delta_t)

    # Downsample the pre-generated noise arrays (here, delta_t is the finest resolution)
    # current_Nn_true = Nn_true[:n_steps]
    # current_Nn_measurements = Nn_measurements[:n_steps]
    current_Nn_true = Nn_true
    current_Nn_measurements = Nn_measurements


    if norm_noise:
        current_Nn_true = current_Nn_true / np.sqrt(delta_t)
        current_Nn_measurements = current_Nn_measurements / np.sqrt(delta_t)

    # Simulate the true state at resolution delta_t
    X_true_full = simulate_true_state(A, B, delta_t, T, current_Nn_true)
    time_full = np.arange(0, T + delta_t, delta_t)

    # Generate raw measurements
    measurements_full = generate_measurements(X_true_full, R, delta_t=None)

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

    # Average the measurements (and true state) over non-overlapping windows.
    k = len(measurements_full) // avg_window
    averaged_measurements = []
    averaged_X_true = []
    averaged_time = []
    x_prior_array = np.zeros(k)
    x_posterior_array = np.zeros(k)
    p_prior_array = np.zeros(k)
    p_post_array = np.zeros(k)


    for i in range(k):
        start_idx = i * avg_window
        end_idx = (i + 1) * avg_window
        measurements_window = measurements_full[start_idx:end_idx]
        averaged_measurements.append(np.mean(measurements_window))
        # Choose the true state and time at the end of the averaging window.
        true_index = min(end_idx, len(X_true_full) - 1)
        averaged_X_true.append(X_true_full[true_index])
        averaged_time.append(time_full[true_index])

        # Run the EKF filtering loop.

        for ind in range(len(measurements_window)):
        # Prediction step.
            x_prior, p_prior = ekf.predict(delta_t, norm_noise=norm_noise)
        x_prior_array[i] = x_prior
        p_prior_array[i] = p_prior

        # Update step
        z = np.mean(measurements_window)
        x_post, p_post = ekf.update(z)
        x_posterior_array[i] = x_post
        p_post_array[i] = p_post

    result = {
        'time': averaged_time,
        'X_true': averaged_X_true,
        'x_prior': x_prior_array,
        'p_prior': p_prior_array,
        'x_posterior': x_posterior_array,
        'p_post': p_post_array,
        'avg_window': avg_window
    }

    return result


def run_simulation_fixed_dt2(A, B, T, delta_t, R, avg_window,
                            Nn_true, Nn_measurements,
                            initial_state, initial_covariance,
                            norm_noise, norm_R):
    """
    Runs one simulation/EKF filtering cycle for a given averaging window.
    Returns a dictionary containing:
      - 'time': the time indices of the averaged simulation,
      - 'X_true': the averaged true state,
      - 'x_prior': the EKF prior estimates,
      - 'x_posterior': the EKF posterior estimates.
    """
    # Total simulation steps at this delta_t:
    n_steps = int(T / delta_t)

    # Downsample the pre-generated noise arrays (here, delta_t is the finest resolution)
    # current_Nn_true = Nn_true[:n_steps]
    # current_Nn_measurements = Nn_measurements[:n_steps]
    current_Nn_true = Nn_true
    current_Nn_measurements = Nn_measurements


    if norm_noise:
        current_Nn_true = current_Nn_true / np.sqrt(delta_t)
        current_Nn_measurements = current_Nn_measurements / np.sqrt(delta_t)

    # Simulate the true state at resolution delta_t
    X_true_full = simulate_true_state(A, B, delta_t, T, current_Nn_true)
    time_full = np.arange(0, T + delta_t, delta_t)

    # Generate raw measurements
    measurements_full = generate_measurements(X_true_full, R, delta_t=None)

    # Average the measurements (and true state) over non-overlapping windows.
    k = len(measurements_full) // avg_window
    averaged_measurements = []
    averaged_X_true = []
    averaged_time = []

    for i in range(k):
        start_idx = i * avg_window
        end_idx = (i + 1) * avg_window
        averaged_measurements.append(np.mean(measurements_full[start_idx:end_idx]))
        # Choose the true state and time at the end of the averaging window.
        true_index = min(end_idx, len(X_true_full) - 1)
        averaged_X_true.append(X_true_full[true_index])
        averaged_time.append(time_full[true_index])

    averaged_measurements = np.array(averaged_measurements)
    averaged_X_true = np.array(averaged_X_true)
    averaged_time = np.array(averaged_time)

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

    n_points = len(averaged_X_true)
    x_prior_array = np.zeros(n_points)
    x_posterior_array = np.zeros(n_points)
    # (Covariance arrays could also be returned if desired.)

    # Set initial state.
    x_posterior_array[0] = initial_state

    # Also average the process noise over each window.
    averaged_Nn = []
    for i in range(k):
        start_idx = i * avg_window
        end_idx = (i + 1) * avg_window
        # averaged_Nn.append(np.mean(current_Nn_measurements[start_idx:end_idx]))
        averaged_Nn.append(current_Nn_measurements[end_idx-1])
    averaged_Nn = np.array(averaged_Nn)

    # The effective time step for prediction is the window length.
    dt_avg = avg_window * delta_t

    # Run the EKF filtering loop.
    for i in range(1, n_points):
        # Prediction step.
        x_prior, _ = ekf.predict(dt_avg, averaged_Nn[i - 1], norm_noise=norm_noise)
        x_prior_array[i] = x_prior

        # Update step (using the measurement from the previous window).
        if i - 1 < len(averaged_measurements):
            z = averaged_measurements[i - 1]
            x_post, _ = ekf.update(z)
        else:
            raise ValueError("This should not happen.")
            x_post = x_prior
        x_posterior_array[i] = x_post

    result = {
        'time': averaged_time,
        'X_true': averaged_X_true,
        'x_prior': x_prior_array,
        'x_posterior': x_posterior_array,
        'avg_window': avg_window
    }
    return result


def main():
    # --------------------------
    # Simulation and EKF Settings
    # --------------------------
    A = 3.0
    B = 1.0
    T = 10.0

    # Use the lowest delta_t value (finest resolution)
    delta_t = 1/2000  # Adjust as needed; this is our "finest" dt for this study.
    transients = (T * 0.1) // delta_t   # Number of transient steps to skip for MSE computation.
    R = 0.1
    initial_state = 1.0
    initial_covariance = 1.0
    norm_noise = True
    norm_R = True

    # List of averaging window sizes to test.
    avg_window_list = np.logspace(0.1, 3.2, num=100, dtype=int)

    # Pre-generate noise arrays at the finest resolution.
    max_steps = int(T / delta_t)
    # np.random.seed(42)
    Nn_true = np.random.randn(max_steps)
    Nn_measurements = np.random.randn(max_steps)

    # --------------------------
    # Run baseline simulation (avg_window = 1) to serve as the ground truth.
    # --------------------------
    baseline_result = run_simulation_fixed_dt(A, B, T, delta_t, R, 1,
                                              Nn_true, Nn_measurements,
                                              initial_state, initial_covariance,
                                              norm_noise, norm_R)
    baseline_time = baseline_result['time']
    baseline_true = baseline_result['X_true']

    # --------------------------
    # For each averaging window, run the simulation and compute the MSE (for both prior and posterior)
    # relative to the baseline true state.
    # --------------------------
    mse_prior_list = []
    mse_post_list = []
    p_prior_list = []
    p_post_list = []
    for avg_window in avg_window_list:
        result = run_simulation_fixed_dt(A, B, T, delta_t, R, avg_window,
                                         Nn_true, Nn_measurements,
                                         initial_state, initial_covariance,
                                         norm_noise, norm_R)
        # Because the simulation time grids differ for different averaging windows,
        # interpolate the baseline true state onto the current simulation's time grid.
        interp_true = np.interp(result['time'], baseline_time, baseline_true)
        mse_prior = np.mean((result['x_prior'] - interp_true) ** 2)
        mse_post = np.mean((result['x_posterior'] - interp_true) ** 2)
        p_post = np.mean(result['p_post'][-1])
        p_prior = np.mean(result['p_prior'][-1])

        mse_prior_list.append(mse_prior)
        mse_post_list.append(mse_post)
        p_prior_list.append(np.sqrt(p_prior))
        p_post_list.append(np.sqrt(p_post))

    # --------------------------
    # Plot the MSE for Prior and Posterior as a function of the averaging window.
    # Both curves are shown on the same figure.
    # --------------------------
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
    plt.title('MSE vs. Averaging Window (Δt = {:.4f}, PRF = {})'.format(delta_t, 1/delta_t))
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
