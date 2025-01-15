import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.8  # Coefficient for X(k)
B = 1.0      # Noise scaling factor
K = 100000   # Total steps
T = 10000    # Number of transient steps to discard

# Generate noise and initialize X
N = np.random.normal(0, 1, K)
X = np.zeros(K)

# Generate X sequence
for k in range(1, K):
    X[k] = alpha * X[k-1] + B * N[k]

# Discard transients
X_stationary = X[T:]

# Calculate numerical variance
numerical_variance = np.var(X_stationary)

# Analytical variance
analytical_variance = (B**2) / (1 - alpha**2)

# Calculate running variance
running_variance = [np.var(X_stationary[:i]) for i in range(2, len(X_stationary))]

# Plot running variance convergence
plt.figure(figsize=(10, 6))
plt.plot(range(2, len(X_stationary)), running_variance, label="Running Variance")
plt.axhline(y=analytical_variance, color="r", linestyle="--", label="Analytical Variance")
plt.title("Variance Convergence")
plt.xlabel("Number of Samples")
plt.ylabel("Variance")
plt.legend()
plt.grid()
plt.show()

# Print final numerical variance
numerical_variance = np.var(X_stationary)
print(f"Numerical Variance: {numerical_variance:.6f}")
print(f"Analytical Variance: {analytical_variance:.6f}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(X_stationary[:1000], label="X(k) (Stationary)")
plt.axhline(y=0, color="r", linestyle="--", label="Mean = 0")
plt.title("Sample Path of X(k) (Stationary Regime)")
plt.xlabel("Time Step (k)")
plt.ylabel("X(k)")
plt.legend()
plt.show()
