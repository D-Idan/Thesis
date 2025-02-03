# Wiener-Brownian Motion Simulation Documentation

This document explains the Python code for simulating Wiener-Brownian motion (WBM) paths at different time resolutions (`delta_t`) using a shared noise vector. The code demonstrates how discretization affects trajectory construction while preserving statistical properties.

---

## Mathematical Background

### Wiener Process Definition
A Wiener process \( W_t \) is characterized by:
1. **Independent Increments**: \( W_{t} - W_s \sim \mathcal{N}(0, t-s) \) for \( t > s \).
2. **Self-Similarity**: \( W_{at} \overset{d}{=} \sqrt{a}W_t \).

### Discrete Approximation
For time step \( \Delta t \), the increment at step \( k \) is:
\[
\Delta W_k = \epsilon_k \sqrt{\Delta t}, \quad \epsilon_k \sim \mathcal{N}(0,1)
\]
The position at time \( t = n\Delta t \) is:
\[
W_t = \sum_{i=1}^n \Delta W_i = \sqrt{\Delta t} \sum_{i=1}^n \epsilon_i
\]

### Coarse-Graining
When aggregating \( k = \frac{\Delta t_{\text{target}}}{\Delta t_{\text{base}}} \) steps:
\[
\Delta W_{\text{coarse}} = \sqrt{\Delta t_{\text{base}}} \sum_{j=1}^k \epsilon_j \equiv \epsilon \sqrt{\Delta t_{\text{target}}}
\]
This preserves variance \( \mathbb{E}[(\Delta W_{\text{coarse}})^2] = \Delta t_{\text{target}} \).

---

## Code Explanation

### 1. Parameters
```python
np.random.seed(42)  # Reproducibility
T = 1.0             # Total simulation time
delta_ts = [0.001, 0.01, 0.05]  # Time steps to compare
dt_min = min(delta_ts)          # Base resolution (0.001)
```

### 2. Noise Generation
Generate a base noise vector at `dt_min` resolution:
```python
n_steps_base = int(T / dt_min)
epsilon = np.random.randn(n_steps_base)  # ~N(0,1)
```
Each \( \epsilon_i \) represents a unit normal variable scaled later by \( \sqrt{\Delta t} \).

### 3. Brownian Motion Generation Function
```python
def generate_brownian_motion(dt, epsilon_base, dt_base):
    k = int(dt / dt_base)       # Step scaling factor
    n_steps = int(T / dt)
    
    # Reshape noise into (n_steps, k) and sum over k steps
    noise = epsilon_base[:n_steps * k].reshape(n_steps, k)
    increments = np.sum(noise, axis=1) * np.sqrt(dt_base)
    
    # Cumulative sum to build path
    bm = np.cumsum(increments)
    bm = np.concatenate([[0], bm])  # Initial condition W_0 = 0
    
    t = np.linspace(0, T, n_steps + 1)
    return bm, t
```
**Key Operations**:
- **Reshaping**: Groups `k` fine steps into one coarse step.
- **Aggregation**: Summing \( k \) terms and scaling by \( \sqrt{\Delta t_{\text{base}}} \) ensures variance matches \( \Delta t_{\text{target}} \).

### 4. Plotting
```python
plt.figure(figsize=(10, 6))

# Scatter plot of base noise scaled by sqrt(dt_min)
t_scatter = np.linspace(dt_min, T, n_steps_base)
plt.scatter(t_scatter, epsilon * np.sqrt(dt_min), 
            color='gray', alpha=0.3, label=f'Noise (dt={dt_min})')

# Plot trajectories for each delta_t
for dt in delta_ts:
    bm, t = generate_brownian_motion(dt, epsilon, dt_min)
    plt.plot(t, bm, '.-', linewidth=1, markersize=4, 
             label=f'dt={dt} (steps={int(T/dt)})')

plt.xlabel("Time")
plt.ylabel("Position (Brownian Motion)")
plt.gca().invert_yaxis()  # Time starts at bottom (t=0)
plt.title("Wiener-Brownian Motion: Effect of Time Discretization")
plt.legend()
plt.show()
```
**Visual Elements**:
- **Gray Dots**: Individual noise increments at `dt_min` resolution.
- **Lines**: Brownian paths at different `delta_t`, aligned at shared time steps.

---

## Output Interpretation
- **Self-Similarity**: All trajectories originate from the same noise sequence. Differences arise only from how increments are aggregated.
- **Consistency**: Paths overlap at common time points (e.g., \( t=0.01, 0.05 \)) despite different `delta_t`, validating correct scaling.
- **Noise Scaling**: Scatter plot shows \( \epsilon_i \sqrt{\Delta t_{\text{base}}} \), matching the variance of individual increments.
