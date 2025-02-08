import numpy as np
from ekf import ExtendedKalmanFilter
from simulation import simulate_true_state, generate_measurements
from plot_results import plot_results, plot_single_result


def main():
    A = 3.0
    B = 1.0
    T = 10.0
    delta_t_values = [0.1, 0.05, 0.01]
    R_values = [0.1, 1.0]
    initial_state = 1.0
    initial_covariance = 1.0 #1e-5
    norm_noise = True
    use_rolling_average_measurements = False  # Set this flag to True to use rolling average fore measurement
    save_path = './results_plots'

    # Generate process noise array for the smallest delta_t
    min_delta_t = min(delta_t_values)
    max_steps = int(T / min_delta_t)
    np.random.seed(42)
    Nn_true = np.random.randn(max_steps)
    Nn_measurements = np.random.randn(max_steps)

    results = []
    for delta_t in delta_t_values:
        n_steps = int(T / delta_t)
        shift = int(delta_t / min_delta_t)
        current_Nn_true = Nn_true[::shift][:n_steps]
        current_Nn_measurements = Nn_measurements[::shift][:n_steps]
        if norm_noise:
            current_Nn_true = current_Nn_true / np.sqrt(delta_t)
            current_Nn_measurements = current_Nn_measurements / np.sqrt(delta_t)

        X_true = simulate_true_state(A, B, delta_t, T, current_Nn_true)
        X_true_orig = X_true.copy()
        time_inx = np.arange(0, T + delta_t, delta_t)

        current_Nn = current_Nn_measurements


        for R in R_values:
            measurements = generate_measurements(X_true_orig, R)

            # If using rolling average for measurements, process them accordingly
            if use_rolling_average_measurements:
                averaged_measurements = []
                x_true_related = []

                # Loop over the measurements in chunks of size 'shift'
                k = len(measurements) // shift
                for i in range(k):
                    start_idx = i * shift
                    end_idx = (i + 1) * shift

                    # Compute the mean of the current chunk and append to the list
                    chunk_mean = np.mean(measurements[start_idx:end_idx])
                    averaged_measurements.append(chunk_mean)
                    x_true_related.append(X_true_orig[end_idx])
                    time_inx = np.arange(0, T + delta_t - delta_t * shift, delta_t * shift)

                measurements = np.array(averaged_measurements)
                X_true = np.array(x_true_related)

            # if norm_noise:
            #     R_kf = R / delta_t
            # else:
            #     R_kf = R
            R_kf = R
            if use_rolling_average_measurements:
                R_kf = R_kf / shift

            ekf = ExtendedKalmanFilter(A=A, B=B, R=R_kf, initial_state=initial_state, initial_covariance=initial_covariance)
            x_prior_array = np.zeros_like(X_true)
            x_posterior_array = np.zeros_like(X_true)
            p_prior_array = np.zeros_like(X_true)
            p_posterior_array = np.zeros_like(X_true)

            x_posterior_array[0] = initial_state
            p_posterior_array[0] = initial_covariance

            for i in range(1, len(X_true)):
                x_prior, p_prior = ekf.predict(delta_t, current_Nn[i-1], norm_noise=norm_noise)
                x_prior_array[i] = x_prior
                p_prior_array[i] = p_prior

                if i-1 < len(measurements):
                    z = measurements[i-1]
                    x_posterior, p_posterior = ekf.update(z)
                else:
                    x_posterior, p_posterior = x_prior, p_prior

                x_posterior_array[i] = x_posterior
                p_posterior_array[i] = p_posterior

            results.append({
                'delta_t': delta_t,
                'R': R,
                'time_inx': time_inx,
                'X_true': X_true,
                'x_prior': x_prior_array,
                'x_posterior': x_posterior_array,
                'p_prior': p_prior_array,
                'p_posterior': p_posterior_array
            })

            # After generating results for this delta_t and R:
            plot_single_result(
                time_inx=time_inx,
                x_true=X_true,
                x_prior=x_prior_array,
                x_posterior=x_posterior_array,
                p_prior=p_prior_array,
                p_posterior=p_posterior_array,
                A=A,
                B=B,
                delta_t=delta_t,
                R=R,
                save_path = save_path,
            )

    plot_results(results, save_path=save_path)

if __name__ == "__main__":
    main()