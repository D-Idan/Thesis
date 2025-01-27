import matplotlib.pyplot as plt
import numpy as np


def plot_results(time, x_true, x_preds, x_cors, residuals_pred, residuals_cor, p_preds, p_cors):
    plt.figure(figsize=(10, 8))

    # Top plot: X Tracking
    plt.subplot(3, 1, 1)
    plt.plot(time, x_true, label="x_true", color="red")
    plt.plot(time, x_preds, label="x Predictor", color="blue")
    plt.plot(time, x_cors, label="x Corrector", color="green")
    plt.xlabel("time [s]")
    plt.ylabel("signal")
    plt.title("X Tracking (EKF)")
    plt.legend()

    # Bottom left: Predictor Error
    plt.subplot(3, 2, 3)
    plt.plot(time, residuals_pred, label="Residual", color="blue")
    plt.plot(time, np.sqrt(p_preds), label="p^0.5", color="red")
    plt.plot(time, -np.sqrt(p_preds), color="red")
    plt.xlabel("time [s]")
    plt.ylabel("Error")
    plt.title("X Coordinate Predictor Error (EKF)")
    plt.legend()

    # Bottom right: Corrector Error
    plt.subplot(3, 2, 4)
    plt.plot(time, residuals_cor, label="Residual", color="blue")
    plt.plot(time, np.sqrt(p_cors), label="p^0.5", color="red")
    plt.plot(time, -np.sqrt(p_cors), color="red")
    plt.xlabel("time [s]")
    plt.ylabel("Error")
    plt.title("X Coordinate Corrector Error (EKF)")
    plt.legend()

    plt.tight_layout()
    plt.show()
