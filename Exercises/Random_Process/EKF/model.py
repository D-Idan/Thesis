import numpy as np

def generate_white_noise(N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.normal(0, 1, N)

def generate_measurements(A, B, N, delta_t, x0=0, scale_noise=True):
    x_true = [x0]
    for i in range(1, len(N)):
        if scale_noise: N = N * np.sqrt(delta_t)
        dx = -A * x_true[-1] * delta_t + B * N[i]
        x_true.append(x_true[-1] + dx)
    return np.array(x_true)
