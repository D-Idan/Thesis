import numpy as np

def simulate_system(a, b, T, delta_t, initial_state=0):
    time_steps = int(T / delta_t)
    time = np.linspace(0, T, time_steps)
    x_true = np.zeros(time_steps)
    x_true[0] = initial_state

    # Generate white noise
    noise = np.random.normal(0, 1, size=time_steps)

    for i in range(1, time_steps):
        x_true[i] = x_true[i - 1] * (1 + delta_t * a) + delta_t * b * noise[i]

    return time, x_true, noise