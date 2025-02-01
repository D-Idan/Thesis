import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def plot_results(results, save_path=None):
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    linestyles = ['-', '--', '-.', ':']
    color_idx = 0
    linestyle_idx = 0

    # Track which labels have been added to avoid duplicates
    labels_added = set()

    for result in results:
        delta_t = result['delta_t']
        R = result['R']
        time_inx = result['time_inx']
        X_true = result['X_true']
        x_prior = result['x_prior']
        x_posterior = result['x_posterior']
        p_prior = result['p_prior']
        p_posterior = result['p_posterior']
        label = f"dt={delta_t}, R={R}"

        line_color = colors[color_idx % len(colors)]
        line_style = linestyles[linestyle_idx % len(linestyles)]

        # Plot true state only once per delta_t
        true_label = f"True (dt={delta_t})"
        if true_label not in labels_added:
            ax.plot(time_inx, X_true, linestyle='-', color='red', linewidth=1.5, label='True State')
            labels_added.add(true_label)

        # Prior and Posterior estimates
        ax.plot(time_inx, x_prior, linestyle=line_style, color=line_color, linewidth=0.8, label=f'Prior {label}')
        ax.plot(time_inx, x_posterior, linestyle=line_style, color=line_color, linewidth=0.8, label=f'Posterior {label}')

        # Prior Error
        prior_error = x_prior - X_true
        ax2.plot(time_inx, prior_error, linestyle=line_style, color=line_color, linewidth=0.8, label=label)
        ax2.plot(time_inx, np.sqrt(p_prior), linestyle='--', color=line_color, linewidth=0.5)
        ax2.plot(time_inx, -np.sqrt(p_prior), linestyle='--', color=line_color, linewidth=0.5)

        # Posterior Error
        posterior_error = x_posterior - X_true
        ax3.plot(time_inx, posterior_error, linestyle=line_style, color=line_color, linewidth=0.8, label=label)
        ax3.plot(time_inx, np.sqrt(p_posterior), linestyle='--', color=line_color, linewidth=0.5)
        ax3.plot(time_inx, -np.sqrt(p_posterior), linestyle='--', color=line_color, linewidth=0.5)

        color_idx += 1
        if color_idx % len(colors) == 0:
            linestyle_idx += 1

    ax.set_title('X Tracking using EKF')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal')
    ax.legend(loc='upper right', fontsize='x-small')
    ax.grid(True)

    ax2.set_title('Prior Estimation Error')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Error')
    ax2.legend(fontsize='x-small')
    ax2.grid(True)

    ax3.set_title('Posterior Estimation Error')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Error')
    ax3.legend(fontsize='x-small')
    ax3.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, 'results.png'))
    else:
        plt.show()


def plot_single_result(time_inx, x_true, x_prior, x_posterior,
                       p_prior, p_posterior, A, B, delta_t, R, model_name="EKF", save_path=None):
    """Plot a single result (one delta_t and R combination)."""
    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = gridspec.GridSpec(2, 2, figure=fig)

    # Main trajectory plot
    ax = fig.add_subplot(gs[0, :])
    ax.plot(time_inx, x_true, label='True State', color='red', linewidth=1.5)
    ax.plot(time_inx, x_prior, label='Prior Estimate', color='blue', linestyle='--', linewidth=1)
    ax.plot(time_inx, x_posterior, label='Posterior Estimate', color='green', linestyle='-.', linewidth=1)
    ax.set_title(f'EKF Tracking (A={A}, B={B}, Δt={delta_t}, R={R})')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Signal')
    ax.legend()
    ax.grid(True)

    # Prior error plot
    ax2 = fig.add_subplot(gs[1, 0])
    prior_error = x_prior - x_true
    ax2.plot(time_inx, prior_error, label='Prior Error', color='blue', linewidth=1)
    ax2.plot(time_inx, np.sqrt(p_prior), label='±p^0.5', color='red', linestyle='--', linewidth=0.8)
    # ax2.plot(time_inx, np.sqrt(p_prior), label='±σ', color='red', linestyle='--', linewidth=0.8)
    ax2.plot(time_inx, -np.sqrt(p_prior), color='red', linestyle='--', linewidth=0.8)
    ax2.set_title('Prior Error ± Uncertainty')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True)

    # Posterior error plot
    ax3 = fig.add_subplot(gs[1, 1])
    posterior_error = x_posterior - x_true
    ax3.plot(time_inx, posterior_error, label='Posterior Error', color='green', linewidth=1)
    ax3.plot(time_inx, np.sqrt(p_posterior), label='±p^0.5', color='red', linestyle='--', linewidth=0.8)
    # ax3.plot(time_inx, np.sqrt(p_posterior), label='±σ', color='red', linestyle='--', linewidth=0.8)
    ax3.plot(time_inx, -np.sqrt(p_posterior), color='red', linestyle='--', linewidth=0.8)
    ax3.set_title('Posterior Error ± Uncertainty')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Error')
    ax3.legend()
    ax3.grid(True)

    # # Determine combined y-limits
    # y_min = min(ax2.get_ylim()[0], ax3.get_ylim()[0])
    # y_max = max(ax2.get_ylim()[1], ax3.get_ylim()[1])
    #
    # # Set the same y-limits for both plots
    # ax2.set_ylim(y_min, y_max)
    # ax3.set_ylim(y_min, y_max)

    # Determine combined y-limits
    y_max = max(max(np.sqrt(p_posterior)), max(np.sqrt(p_prior)))

    # Set the same y-limits for both plots
    ax2.set_ylim(-y_max, y_max)
    ax3.set_ylim(-y_max, y_max)


    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(os.path.join(save_path, f'result_{delta_t}_{R}.png'))
    else:
        plt.show()