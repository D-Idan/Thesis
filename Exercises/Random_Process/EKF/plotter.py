import matplotlib.pyplot as plt
import torch

def plot_results(time, x_true, x_prior_array, x_posterior_array, p_prior_array, p_posterior_array, a, b):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plot state tracking
    axes[0].plot(time, x_true, label="True State", color="red")
    axes[0].plot(time, x_prior_array, label="Predicted State", color="blue")
    axes[0].plot(time, x_posterior_array, label="Corrected State", color="green")
    axes[0].set_title(f"State Tracking (a={a}, b={b})")
    axes[0].legend()

    # Prediction errors
    prediction_error = torch.tensor(x_prior_array) - torch.tensor(x_true)
    axes[1].plot(time, prediction_error, label="Prediction Error", color="blue")
    axes[1].fill_between(time, -torch.sqrt(torch.tensor(p_prior_array)), torch.sqrt(torch.tensor(p_prior_array)), color="red", alpha=0.3)
    axes[1].set_title("Prediction Error")
    axes[1].legend()

    # Correction errors
    correction_error = torch.tensor(x_posterior_array) - torch.tensor(x_true)
    axes[2].plot(time, correction_error, label="Correction Error", color="blue")
    axes[2].fill_between(time, -torch.sqrt(torch.tensor(p_posterior_array)), torch.sqrt(torch.tensor(p_posterior_array)), color="red", alpha=0.3)
    axes[2].set_title("Correction Error")
    axes[2].legend()

    plt.tight_layout()
    plt.show()