import numpy as np
from ekf import ExtendedKalmanFilter
from simulation import simulate_true_state, generate_measurements
from plot_results import plot_results, plot_single_result


def main():
    A = 3.0
    B = 1.0
    T = 10.0
    delta_t_values = [0.1, 0.5, 1.0]
    R_values = [0.1, 1.0, 10.0]
    initial_state = 0.0
    initial_covariance = 10 #1e-5
    save_path = './results_plots'

    # Generate process noise array for the smallest delta_t
    min_delta_t = min(delta_t_values)
    max_steps = int(T / min_delta_t)
    np.random.seed(42)
    Nn = np.random.randn(max_steps)

    results = []
    for delta_t in delta_t_values:
        n_steps = int(T / delta_t)
        # current_Nn = Nn[:n_steps]
        shift = int(delta_t / min_delta_t)
        current_Nn = Nn[::shift][:n_steps]

        X_true = simulate_true_state(A, B, delta_t, T, current_Nn)
        time_inx = np.arange(0, T + delta_t, delta_t)

        for R in R_values:
            measurements = generate_measurements(X_true, R)

            ekf = ExtendedKalmanFilter(A=A, B=B, R=R, initial_state=initial_state, initial_covariance=initial_covariance)
            x_prior_array = np.zeros_like(X_true)
            x_posterior_array = np.zeros_like(X_true)
            p_prior_array = np.zeros_like(X_true)
            p_posterior_array = np.zeros_like(X_true)

            x_posterior_array[0] = initial_state
            p_posterior_array[0] = initial_covariance

            for i in range(1, len(X_true)):
                x_prior, p_prior = ekf.predict(delta_t)
                x_prior_array[i] = x_prior
                p_prior_array[i] = p_prior

                if i-1 < len(measurements):
                    z = measurements[i-1]
                    x_posterior, p_posterior = ekf.update(z, delta_t)
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