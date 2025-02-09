import numpy as np
from ekf import ExtendedKalmanFilter
from simulation import simulate_true_state, generate_measurements
from plot_results import plot_results, plot_single_result

def main():
    # Parameters for simulation and filtering
    A = 3.0
    B = 1.0
    T = 10.0
    delta_t_values = [0.1, 0.05, 0.01]  # simulation time steps
    R_values = [0.1, 1.0]               # measurement noise variances
    initial_state = 1.0
    initial_covariance = 1.0
    norm_noise = True
    save_path = './results_plots'

    # Define the list of averaging window sizes (number of measurements per average)
    # For example, 1 means no averaging, 3 and 5 are additional cases.
    avg_window_sizes = [1, 3, 5]

    # Generate the noise arrays at the highest (finest) resolution:
    min_delta_t = min(delta_t_values)
    max_steps = int(T / min_delta_t)
    np.random.seed(42)
    Nn_true = np.random.randn(max_steps)
    Nn_measurements = np.random.randn(max_steps)

    all_results = []  # to collect results for all cases

    # Loop over the simulation time steps
    for delta_t in delta_t_values:
        n_steps = int(T / delta_t)
        # Downsample the noise arrays to the current delta_t resolution.
        shift = int(delta_t / min_delta_t)
        current_Nn_true = Nn_true[::shift][:n_steps]
        current_Nn_measurements = Nn_measurements[::shift][:n_steps]
        if norm_noise:
            current_Nn_true = current_Nn_true / np.sqrt(delta_t)
            current_Nn_measurements = current_Nn_measurements / np.sqrt(delta_t)

        # Simulate the true state at this resolution.
        X_true_full = simulate_true_state(A, B, delta_t, T, current_Nn_true)
        # Keep a copy for generating measurements.
        X_true_orig = X_true_full.copy()
        time_full = np.arange(0, T + delta_t, delta_t)

        for R in R_values:
            # Generate raw measurements from the true state.
            measurements_full = generate_measurements(X_true_orig, R)

            # Now loop over different averaging window sizes.
            for avg_window in avg_window_sizes:
                # Determine how many complete averaging windows we have.
                k = len(measurements_full) // avg_window
                averaged_measurements = []
                averaged_X_true = []
                averaged_time = []

                # Average measurements in non-overlapping windows.
                for i in range(k):
                    start_idx = i * avg_window
                    end_idx = (i + 1) * avg_window
                    chunk_mean = np.mean(measurements_full[start_idx:end_idx])
                    averaged_measurements.append(chunk_mean)
                    # Use the true state and time at the end of the window.
                    true_index = min(end_idx, len(X_true_orig) - 1)
                    averaged_X_true.append(X_true_orig[true_index])
                    averaged_time.append(time_full[true_index])

                # Convert lists to arrays.
                averaged_measurements = np.array(averaged_measurements)
                averaged_X_true = np.array(averaged_X_true)
                averaged_time = np.array(averaged_time)

                # Scale the measurement noise covariance to reflect the averaging.
                # (Here we use a simple scaling: R_kf = R / avg_window.)
                R_kf = R / avg_window

                # Initialize the Extended Kalman Filter.
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
                p_prior_array = np.zeros(n_points)
                p_posterior_array = np.zeros(n_points)

                # Set initial condition.
                x_posterior_array[0] = initial_state
                p_posterior_array[0] = initial_covariance

                # For prediction we need the corresponding process noise for each averaged step.
                # Here we average the raw process noise in each window.
                averaged_Nn = []
                for i in range(k):
                    start_idx = i * avg_window
                    end_idx = (i + 1) * avg_window
                    noise_avg = np.mean(current_Nn_measurements[start_idx:end_idx])
                    averaged_Nn.append(noise_avg)
                averaged_Nn = np.array(averaged_Nn)

                # The effective time step for prediction is the duration of the averaging window.
                dt_avg = avg_window * delta_t

                # Run the EKF filtering loop.
                for i in range(1, n_points):
                    # Prediction step.
                    x_prior, p_prior = ekf.predict(dt_avg, averaged_Nn[i - 1], norm_noise=norm_noise)
                    x_prior_array[i] = x_prior
                    p_prior_array[i] = p_prior

                    # Update step using the corresponding averaged measurement.
                    # (The measurement at index i-1 is used to update state at time index i.)
                    if i - 1 < len(averaged_measurements):
                        z = averaged_measurements[i - 1]
                        x_post, p_post = ekf.update(z)
                    else:
                        x_post, p_post = x_prior, p_prior

                    x_posterior_array[i] = x_post
                    p_posterior_array[i] = p_post

                # Save the results for later comparison.
                result = {
                    'delta_t': delta_t,
                    'R': R,
                    'avg_window': avg_window,
                    'time': averaged_time,
                    'X_true': averaged_X_true,
                    'x_prior': x_prior_array,
                    'x_posterior': x_posterior_array,
                    'p_prior': p_prior_array,
                    'p_posterior': p_posterior_array
                }
                all_results.append(result)

                # Plot the result for this configuration.
                # (The extra_info parameter is used to label the plot with the averaging window.)
                plot_single_result(
                    time_inx=averaged_time,
                    x_true=averaged_X_true,
                    x_prior=x_prior_array,
                    x_posterior=x_posterior_array,
                    p_prior=p_prior_array,
                    p_posterior=p_posterior_array,
                    A=A,
                    B=B,
                    delta_t=delta_t,
                    R=R,
                    extra_info=f'avg_window={avg_window}',
                    save_path=save_path,
                )

    # Finally, generate a summary plot comparing all cases.
    plot_results(all_results, save_path=save_path)

if __name__ == "__main__":
    main()
