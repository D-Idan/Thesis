

---

# **Extended Kalman Filter (EKF) for Stochastic Differential Equation**

## **Problem Setup**
We solve the stochastic differential equation (SDE):

$$\dot{X} = -A X + B N$$

where:

* $A = 3$ (Drift coefficient)
* $B = 1$
* $N \sim \mathcal{N}(0,1)$ is white noise (generated once and used for all $\Delta t$)

Initial conditions:

$$X(0) = 0$$

The solution is computed up to $T = 10$ seconds for different $\Delta t$ values and measurement noise levels.

## **State-Space Representation**
Discretizing the system using Euler’s method:

$$X[i] = X[i - 1] \cdot (1 - A \cdot \Delta t) + B \cdot \Delta t \cdot N_n[i]$$

We estimate $X_k$ over time using the Extended Kalman Filter (EKF), incorporating process noise and measurement noise.

## **Implementation in EKF Code**
The **EKF** follows these steps:

1. **Prediction Step**:
$$X_{\text{prior}} = (1 - A \cdot \Delta t) \cdot X_{\text{posterior}} + B \cdot \Delta t \cdot N$$

$$P_{\text{prior}} = F \cdot P_{\text{posterior}} \cdot F^T + Q$$

where:

* $F = 1 - A \cdot \Delta t$ (State transition Jacobian)
* $Q = (B \cdot \Delta t)^2$ (Process noise covariance)
2. **Update Step**:
$$y = Z - H \cdot X_{\text{prior}} \quad \text{(Innovation)}$$

$$S = H \cdot P_{\text{prior}} \cdot H^T + R \quad \text{(Innovation covariance)}$$

$$K = \frac{P_{\text{prior}} \cdot H}{S} \quad \text{(Kalman Gain)}$$

$$X_{\text{posterior}} = X_{\text{prior}} + K \cdot y$$

$$P_{\text{posterior}} = (1 - K \cdot H) \cdot P_{\text{prior}}$$

where:

* $H = 1$ (Measurement Jacobian)
* $R$ is the measurement noise covariance.

## **Code Overview**
The main components of the implementation:

### **EKF Class (ekf.py)**

* Defines **prediction** and **update** functions using the equations above.
* Stores the state and covariance updates.

### **Main Execution (main.py)**

* Simulates the true state trajectory using the given SDE.
* Generates process noise $N_n$ and measurements with noise.
* Runs the **EKF** for different values of $\Delta t$ and $R$.
* Stores and plots results.

## **Results**

* The filter's performance is tested with varying $\Delta t$ and $R$.
* Smaller $\Delta t$ improves accuracy.
* Larger $R$ increases uncertainty in the estimation.




---
---
---
---

# **Extended Kalman Filter (EKF) with Wiener Process Adjustment**

## **Problem Setup**  
We solve the **stochastic differential equation (SDE)**:

\[
\dot{X} = -A X + B N
\]

where:  
- \( A = 3 \) (Drift coefficient)  
- \( B = 1 \)  
- \( N \sim \mathcal{N}(0,1) \) is white noise  

To correctly model the **continuous-time process in discrete time**, we now modify the noise term to reflect a **Wiener process**. Instead of using raw white noise \( N_n[i] \), we scale it by **\( \sqrt{\Delta t} \)**:

\[
N_n[i] = \frac{N[i]}{\sqrt{\Delta t}}
\]

This adjustment ensures that the noise maintains the correct statistical properties when transitioning from continuous to discrete time.

## **Why Adjust the Noise?**  
A **Wiener process** (or Brownian motion) has the property that its **variance scales with time**. This means that over a small time step \( \Delta t \), the total noise effect should have a variance proportional to \( \Delta t \).  

Without this adjustment, if we keep the same noise values regardless of \( \Delta t \), the process noise would **shrink for large \( \Delta t \) and explode for small \( \Delta t \)**. By dividing by \( \sqrt{\Delta t} \), we ensure that the variance is consistent across different time step sizes.

## **State-Space Representation**  
Discretizing the system with the **Wiener process adjustment**, the state equation becomes:

\[
X[i] = X[i - 1] \cdot (1 - A \cdot \Delta t) + B \cdot \sqrt{\Delta t} \cdot N_n[i]
\]

where \( N_n[i] \sim \mathcal{N}(0,1) \) is the white noise term.

## **Extended Kalman Filter (EKF) Implementation**  
The **EKF** follows the same prediction and update steps, with a modified process noise:

### **Prediction Step**  
\[
X_{\text{prior}} = (1 - A \cdot \Delta t) \cdot X_{\text{posterior}} + B \cdot \sqrt{\Delta t} \cdot N
\]
\[
P_{\text{prior}} = F \cdot P_{\text{posterior}} \cdot F^T + Q
\]

where:  
- \( F = 1 - A \cdot \Delta t \) (State transition Jacobian)  
- \( Q = (B \cdot \sqrt{\Delta t})^2 \) (Process noise covariance)  

### **Update Step**  
\[
y = Z - H \cdot X_{\text{prior}}  \quad \text{(Innovation)}
\]
\[
S = H \cdot P_{\text{prior}} \cdot H^T + R  \quad \text{(Innovation covariance)}
\]
\[
K = \frac{P_{\text{prior}} \cdot H}{S}  \quad \text{(Kalman Gain)}
\]
\[
X_{\text{posterior}} = X_{\text{prior}} + K \cdot y
\]
\[
P_{\text{posterior}} = (1 - K \cdot H) \cdot P_{\text{prior}}
\]

where:
- \( H = 1 \) (Measurement Jacobian)  
- \( R \) is the measurement noise covariance.  

## **Experimental Setup**  
We run the **EKF** for different values of \( \Delta t \) and \( R \) with the Wiener noise adjustment.  

- **\( \Delta t \) values:** [0.1, 0.5, 1.0]  
- **\( R \) values:** [0.1, 1.0, 10.0]  
- **Initial state:** \( X_0 = 0 \)  
- **Initial covariance:** \( P_0 = 10 \)  

## **Expected Results & Observations**  
1. **More Accurate Continuous Approximation**  
   - The Wiener process scaling ensures that the noise remains consistent across different \( \Delta t \).  
   - Without the adjustment, smaller \( \Delta t \) would introduce too much noise, and larger \( \Delta t \) would introduce too little.  

2. **Smoother EKF Estimates**  
   - Since the variance of the noise is correctly adjusted, the EKF should perform better at filtering out process noise.  

3. **Comparison with Previous Implementation**  
   - Without the Wiener scaling, the **EKF’s accuracy would degrade for small \( \Delta t \)** because the noise is too strong.  
   - With the correction, the **filter is more stable across different time steps**.  

## **Conclusion**  
The **adjusted EKF** correctly incorporates the **Wiener process noise**, ensuring accurate state estimation. The experiment highlights the importance of correctly scaling noise when transitioning from continuous to discrete models.