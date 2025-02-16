import numpy as np
import matplotlib.pyplot as plt
from ekf import ExtendedKalmanFilter
from simulation import simulate_true_state, generate_measurements
from plot_results import plot_results, plot_single_result


def main():
    # --------------------------
    # Simulation and EKF settings
    # --------------------------
    A = 3.0
    B = 1.0
    T = 100.0
    delta_t_values = [0.1, 0.05, 0.01]  # different simulation time steps
    delta_t_values = np.arange(0.0001, 0.69, 0.002)  # different simulation time steps 0.66 / 0.68
    R_values = [0.1]  # measurement noise variance(s)
    initial_state = 1.0
    initial_covariance = 1.0
    norm_noise = True
    norm_R = False

    # Compute the MSE for each delta_t (using the case with avg_window==1 and R==0.1)
    R_mse2dt_plot = 0.1
    avg_window_mse2dt_plot = 5  # averaging window for the MSE vs. delta_t plot

    save_path = './results_plots'

    # We will compare the results for the simplest (no averaging) case.
    avg_window_sizes = [avg_window_mse2dt_plot]

    # --------------------------
    # Pre-generate noise arrays at the finest resolution.
    # --------------------------
    min_delta_t = min(delta_t_values)
    max_steps = int(T / min_delta_t)
    np.random.seed(42)
    Nn_true = np.random.randn(max_steps)
    Nn_measurements = np.random.randn(max_steps)

    all_results = []  # store simulation/EKF results for all configurations

    # --------------------------
    # Run simulation/EKF for each delta_t, R, and averaging window.
    # --------------------------
    for delta_t in delta_t_values:
        n_steps = int(T / delta_t)
        # Downsample noise arrays to current resolution.
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

            for avg_window in avg_window_sizes:
                # Average measurements (non-overlapping windows)
                k = len(measurements_full) // avg_window
                averaged_measurements = []
                averaged_X_true = []
                averaged_time = []

                for i in range(k):
                    start_idx = i * avg_window
                    end_idx = (i + 1) * avg_window
                    chunk_mean = np.mean(measurements_full[start_idx:end_idx])
                    averaged_measurements.append(chunk_mean)
                    # Choose the true state and time at the end of the window.
                    true_index = min(end_idx, len(X_true_orig) - 1)
                    averaged_X_true.append(X_true_orig[true_index])
                    averaged_time.append(time_full[true_index])

                # Convert lists to arrays.
                averaged_measurements = np.array(averaged_measurements)
                averaged_X_true = np.array(averaged_X_true)
                averaged_time = np.array(averaged_time)

                # Scale the measurement noise covariance (if desired)
                R_kf = R  # Here you could scale by 1/avg_window if needed.
                if norm_R:
                    R_kf = R_kf / avg_window

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
                p_prior_array = np.zeros(n_points)
                p_posterior_array = np.zeros(n_points)

                # Set initial conditions.
                x_posterior_array[0] = initial_state
                p_posterior_array[0] = initial_covariance

                # Average the raw process noise over each averaging window.
                averaged_Nn = []
                for i in range(k):
                    start_idx = i * avg_window
                    end_idx = (i + 1) * avg_window
                    noise_avg = np.mean(current_Nn_measurements[start_idx:end_idx])
                    averaged_Nn.append(noise_avg)
                averaged_Nn = np.array(averaged_Nn)

                # The effective time step for prediction is the window length.
                dt_avg = avg_window * delta_t

                # Run the EKF filtering loop.
                for i in range(1, n_points):
                    # Prediction step.
                    x_prior, p_prior = ekf.predict(dt_avg, averaged_Nn[i - 1], norm_noise=norm_noise)
                    x_prior_array[i] = x_prior
                    p_prior_array[i] = p_prior

                    # Update step (using the measurement from the previous window).
                    if i - 1 < len(averaged_measurements):
                        z = averaged_measurements[i - 1]
                        x_post, p_post = ekf.update(z)
                    else:
                        x_post, p_post = x_prior, p_prior
                    x_posterior_array[i] = x_post
                    p_posterior_array[i] = p_post

                # Save the result.
                result = {
                    'delta_t': delta_t,
                    'R': R,
                    'avg_window': avg_window,
                    'time_inx': averaged_time,
                    'X_true': averaged_X_true,
                    'x_prior': x_prior_array,
                    'x_posterior': x_posterior_array,
                    'p_prior': p_prior_array,
                    'p_posterior': p_posterior_array
                }
                all_results.append(result)

                # (Optional) Plot the individual result.
                # plot_single_result(
                #     time_inx=averaged_time,
                #     x_true=averaged_X_true,
                #     x_prior=x_prior_array,
                #     x_posterior=x_posterior_array,
                #     p_prior=p_prior_array,
                #     p_posterior=p_posterior_array,
                #     A=A,
                #     B=B,
                #     delta_t=delta_t,
                #     R=R,
                #     extra_info=f'avg_window={avg_window}',
                #     save_path=save_path,
                # )

    # Plot all simulation results (if desired).
    # plot_results(all_results, save_path=save_path)

    # --------------------------
    # Compute the MSE for each delta_t (using the case with avg_window==1 and R==0.1)
    # where the "true" state is taken from the finest simulation (lowest delta_t).
    # --------------------------

    # Filter to keep only results with no averaging and the chosen R.
    results_filtered = [res for res in all_results if res['avg_window'] == avg_window_mse2dt_plot and res['R'] == R_mse2dt_plot]

    # Identify the baseline result (lowest delta_t)
    baseline_dt = min(delta_t_values)
    baseline_result = next(res for res in results_filtered if res['delta_t'] == baseline_dt)
    baseline_time = baseline_result['time_inx']
    baseline_true = baseline_result['X_true']

    delta_t_list = []
    mse_list = []

    for res in results_filtered:
        dt = res['delta_t']
        times = res['time_inx']
        estimate = res['x_posterior']
        # Since the simulation times are multiples of the smallest delta_t,
        # we can compute the corresponding indices.
        indices = (times / baseline_dt).astype(int)
        indices = np.clip(indices, 0, len(baseline_true) - 1)
        true_interp = baseline_true[indices]

        mse = np.mean((estimate - true_interp) ** 2)
        delta_t_list.append(dt)
        mse_list.append(mse)

    # Sort the values for plotting.
    delta_t_arr = np.array(delta_t_list)
    mse_arr = np.array(mse_list)
    sort_idx = np.argsort(delta_t_arr)
    delta_t_arr = delta_t_arr[sort_idx]
    mse_arr = mse_arr[sort_idx]

    # --------------------------
    # Plot log-log graph: x-axis log(delta_t), y-axis log(mse)
    # --------------------------
    plt.figure()
    plt.loglog(delta_t_arr[1:], mse_arr[1:], marker='o', linestyle='-')
    plt.xlabel(r'$\log(\Delta t)$')
    plt.ylabel(r'$\log(\mathrm{MSE})$')
    plt.title('MSE vs. delta_t (Ground Truth from delta_t = {:.4f})'.format(min_delta_t))
    plt.grid(True, which="both", ls="--", alpha=0.7, label='MSE vs Δt')
    plt.savefig(f'{save_path}/mse_vs_delta_t.png')
    plt.show()


if __name__ == "__main__":
    main()
