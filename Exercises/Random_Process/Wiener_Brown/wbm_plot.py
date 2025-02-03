import numpy as np
import matplotlib.pyplot as plt

# Parameters
np.random.seed(42)  # Reproducibility
T = 1.0  # Total simulation time
delta_ts = [0.001, 0.01, 0.05]  # Time steps to compare
dt_min = min(delta_ts) # Smallest time step (base resolution)

# Generate base noise vector at finest resolution (dt_min)
n_steps_base = int(T / dt_min)
epsilon = np.random.randn(n_steps_base)  # ~N(0,1)


# Function to build Brownian paths from the same noise vector
def generate_brownian_motion(dt, epsilon_base, dt_base):
    '''
    Generate Brownian motion from a noise vector at a given time step
    :param dt: Target time step
    :param epsilon_base: Noise vector at base resolution
    :param dt_base: Base resolution (dt_min)
    :return:
    '''
    k = int(dt / dt_base)  # Step scaling factor
    n_steps = int(T / dt)

    # Reshape noise to match target dt
    # Extract the required number of noise values from the base vector
    subset_noise = epsilon_base#[:n_steps * k] # n_steps * k = T / dt_base

    # Reshape into a 2D array where each row corresponds to a time step,
    # and each row contains `k` fine-resolution steps
    noise = subset_noise.reshape(n_steps, k)

    # Sum k small steps to create larger steps (Donsker's theorem)
    increments = np.sum(noise, axis=1) * np.sqrt(dt_base)  # Scale variance by dt
    bm = np.cumsum(increments)
    bm = np.concatenate([[0], bm])  # Initial position at 0

    # Time axis
    t = np.linspace(0, T, n_steps + 1)
    return bm, t


# Generate and plot trajectories
plt.figure(figsize=(10, 6))

# Plot noise vector (scaled by dt_min) as scatter
t_scatter = np.linspace(dt_min, T, n_steps_base)
plt.scatter(t_scatter, epsilon * np.sqrt(dt_min),
            color='gray', alpha=0.3, label=f'Noise (dt={dt_min})')

# Plot Brownian motions for different delta_t
for dt in delta_ts:
    bm, t = generate_brownian_motion(dt, epsilon, dt_min)
    plt.plot(t, bm, '.-', linewidth=1, markersize=4,
             label=f'dt={dt} (steps={int(T / dt)})')

plt.ylabel("Position (Brownian Motion)")
plt.xlabel("Time")
plt.gca().invert_yaxis()  # Time starts at bottom (t=0)
plt.title("Wiener-Brownian Motion: Effect of Time Discretization")
plt.grid(True)
plt.legend()
plt.show()