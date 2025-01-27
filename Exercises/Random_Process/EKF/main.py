import numpy as np
from ekf import ExtendedKalmanFilter
from model import generate_white_noise, generate_measurements
from plotter import plot_results

# Parameters
A = 3
B = 1
Q_vals = [1, 0.1] # [0.1, 0.01, 0.001]  # Process noise variance
R_vals = [0.01, 1]  # Measurement noise variance
delta_t_vals = [0.01, 1.0]
T = 10  # Total simulation time
x0 = 0  # Initial condition

# Main simulation loop
for delta_t in delta_t_vals:
    N = int(T / delta_t)
    time = np.linspace(0, T, N)
    noise = generate_white_noise(N, seed=42)
    x_true = generate_measurements(A, B, noise, delta_t, x0)

    for Q in Q_vals:
        for R in R_vals:

            ekf = ExtendedKalmanFilter(A, B, Q, R, delta_t)

            x_preds = []
            x_cors = []
            p_preds = []
            p_cors = []
            residuals_pred = []
            residuals_cor = []

            for i in range(N):
                z = x_true[i] + np.random.normal(0, np.sqrt(R))  # Simulate noisy measurement
                x_pred, x_cor, p_pred, p_cor = ekf.step(z)

                # Store values
                x_preds.append(x_pred)
                x_cors.append(x_cor)
                p_preds.append(p_pred)
                p_cors.append(p_cor)
                residuals_pred.append(z - x_pred)
                residuals_cor.append(z - x_cor)

            # Plot results for each delta_t and Q
            plot_results(
                time, x_true, x_preds, x_cors,
                residuals_pred, residuals_cor,
                p_preds, p_cors
            )

            # Compute MSE for predictions and corrections
            mse_pred = np.mean((np.array(x_preds) - x_true) ** 2)  # MSE for predictions
            mse_cor = np.mean((np.array(x_cors) - x_true) ** 2)  # MSE for corrections

            print(f"delta_t={delta_t}, R={R}, Q={Q}: MSE (Predictor)={mse_pred:.6f}, MSE (Corrector)={mse_cor:.6f}")
