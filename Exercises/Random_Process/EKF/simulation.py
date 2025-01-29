import numpy as np

def simulate_true_state(A, B, delta_t, T, Nn):
    n_steps = int(T / delta_t)
    X_true = np.zeros(n_steps + 1)
    X_true[0] = 0.0  # Initial condition
    for i in range(1, n_steps + 1):
        X_true[i] = X_true[i-1] * (1 - A * delta_t) + B * delta_t * Nn[i-1]
    return X_true

def generate_measurements(X_true, R):
    n_steps = len(X_true) - 1
    measurements = X_true[1:] + np.random.randn(n_steps) * np.sqrt(R)
    return measurements