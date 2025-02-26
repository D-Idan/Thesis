import numpy as np
import matplotlib.pyplot as plt
from ekf import ExtendedKalmanFilter
from simulation import simulate_true_state, generate_measurements
from plot_results import plot_results, plot_single_result


def main():
    # --------------------------
    # Simulation and filter parameters
    # --------------------------
    A = 3.0
    B = 1.0
    T = 10.0
    delta_t = 0.01  # base simulation time step
    R = 1.0  # measurement noise variance
    initial_state = 1.0
    initial_covariance = 1.0
    norm_noise = True
    save_path = './results_plots'

    # List of sliding window sizes (i.e. number of prediction steps before an update)
    sliding_window_list = [1, 2, 5, 10]
    sliding_window_list = range(1, 100, 1)

    # --------------------------
    # Generate process noise and simulate true state
    # --------------------------
    n_steps = int(T / delta_t)
    np.random.seed(42)
    Nn_true = np.random.randn(n_steps)
    Nn_measurements = np.random.randn(n_steps)

    if norm_noise:
        Nn_true = Nn_true / np.sqrt(delta_t)
        Nn_measurements = Nn_measurements / np.sqrt(delta_t)

    # Simulate the true state at every delta_t step.
    X_true_full = simulate_true_state(A, B, delta_t, T, Nn_true)
    time_full = np.arange(0, T + delta_t, delta_t)[:n_steps]

    # Generate measurements at every delta_t step.
    measurements_full = generate_measurements(X_true_full, R)

    # To store the MSE for each sliding window value.
    mse_prior_list = []
    mse_post_list = []

    # --------------------------
    # Loop over each sliding window value.
    # For each sliding window, the filter will run several predict steps (one per delta_t)
    # and then perform an update with the measurement corresponding to the end of that window.
    # --------------------------
    for sliding_window in sliding_window_list:
        # Reinitialize the EKF for the new sliding window simulation.
        ekf = ExtendedKalmanFilter(A=A, B=B, R=R,
                                   initial_state=initial_state,
                                   initial_covariance=initial_covariance)

        # Determine the time indices at which an update is applied.
        # The first update is at time index 0 (initial condition), then every sliding_window steps.
        update_indices = list(range(0, n_steps, sliding_window))
        # Ensure the final time is included if it doesn't align exactly.
        if update_indices[-1] != n_steps - 1:
            update_indices.append(n_steps - 1)
        n_updates = len(update_indices)

        # Arrays to store the states at the update instants.
        x_prior_updates = np.zeros(n_updates)
        x_post_updates = np.zeros(n_updates)
        p_prior_updates = np.zeros(n_updates)
        p_post_updates = np.zeros(n_updates)

        # Set the initial conditions.
        x_post_updates[0] = initial_state
        p_post_updates[0] = initial_covariance

        # The simulation proceeds from one update instant to the next.
        current_index = update_indices[0]  # starting at time index 0
        for upd in range(1, n_updates):
            # For the current sliding window, run the predict step sliding_window times.
            # Note: if the window extends beyond the simulation horizon, we break.
            for j in range(sliding_window):
                next_index = current_index + 1
                if next_index >= n_steps:
                    break
                # Use the process noise corresponding to the next time step.
                noise = Nn_measurements[next_index - 1]  # similar to original code usage
                x_prior, p_prior = ekf.predict(delta_t, noise, norm_noise=norm_noise)
                # For this block, record the last prediction as the effective prior.
                if j == sliding_window - 1 or next_index == n_steps - 1:
                    x_prior_updates[upd] = x_prior
                    p_prior_updates[upd] = p_prior
                current_index = next_index

            # Use the measurement at the current (end-of-window) time.
            measurement_index = current_index
            z = measurements_full[measurement_index]
            x_post, p_post = ekf.update(z)
            x_post_updates[upd] = x_post
            p_post_updates[upd] = p_post

        # Extract the true state at the update instants.
        X_true_updates = X_true_full[update_indices]
        time_updates = time_full[update_indices]

        # Compute the mean-squared errors over the update instants.
        mse_prior = np.mean((X_true_updates - x_prior_updates) ** 2)
        mse_post = np.mean((X_true_updates - x_post_updates) ** 2)
        mse_prior_list.append(mse_prior)
        mse_post_list.append(mse_post)

        # # Optionally, plot the result for this sliding window.
        # plot_single_result(
        #     time_inx=time_updates,
        #     x_true=X_true_updates,
        #     x_prior=x_prior_updates,
        #     x_posterior=x_post_updates,
        #     p_prior=p_prior_updates,
        #     p_posterior=p_post_updates,
        #     A=A,
        #     B=B,
        #     # The effective time step for the update cycle is sliding_window*delta_t
        #     delta_t=delta_t * sliding_window,
        #     R=R,
        #     save_path=save_path,
        #     title=f'Sliding Window = {sliding_window}'
        # )

    # --------------------------
    # Plot the MSE vs. Sliding Window for both the prior and posterior estimates.
    # --------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(sliding_window_list, mse_prior_list, marker='o', linestyle='-',
             label='Prior MSE')
    plt.plot(sliding_window_list, mse_post_list, marker='s', linestyle='--',
             label='Posterior MSE')
    plt.xlabel('Sliding Window Size')
    plt.ylabel('MSE')
    plt.title('MSE vs. Sliding Window')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Optionally, plot an overview of all results.
    # plot_results(results, save_path=save_path)   # if you wish to combine all simulations


if __name__ == "__main__":
    main()
