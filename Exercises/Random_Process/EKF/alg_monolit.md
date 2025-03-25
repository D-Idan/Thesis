# Extended Kalman Filter with Measurement Averaging: A Step-by-Step Algorithm Description

## Abstract
This document details the algorithm designed for state estimation in a measurement averaging procedure over variable window sizes.
Results transient effects were mitigated by discarding initial results specified by the `transient_steps` parameter.

## 1. Noise Normalization Methodology Note
   - At the begining the noise sequences \( N_{n,\text{true}} \) and \( N_{n,\text{measurements}} \) are normalized by \( \frac{1}{\sqrt{\Delta t}} \) when `norm_noise` is enabled.  
   - This make sure that all the different simulations will use the same noise values.
   - This will be presented in the rest of this paper as \( \eta_k \) and \( \nu_k \) divided by \( \sqrt{\Delta t} \) respectively. 

## 2. System Model and Notation

### 2.1. System Dynamics
The true state evolution is simulated using the following discrete-time linear dynamic model:
\[
X_k = \left(1 - A\,\Delta t\right) X_{k-1} + B\,\Delta t\,\frac{\eta_k}{\sqrt{\Delta t}} \tag{1}
\]
where:
- \( X_k \) is the true state at time step \( k \).
- \( A \) is the drift coefficient (a constant).
- \( B \) is the noise coefficient (a constant).
- \( \Delta t \) is the time step.
- \( \eta_k \) is the process noise, drawn from a standard normal distribution.

*Code Reference:* The simulation loop computing `X_true[i]` implements Equation (1).

### 2.2. Measurement Model
Measurements are generated based on the true state as:
\[
z_k = X_k + {\sqrt{R}}\frac{\nu_k}{\sqrt{\Delta t}} \tag{2}
\]
where:
- \( z_k \) is the measurement at time step \( k \).
- \( R \) is the measurement noise covariance.
- \( \Delta t \) is the time step.
- \( \nu_k \) is measurement noise drawn from a standard normal distribution.

*Code Reference:* The measurement generation uses `measurements = X_true[1:] + Nn_measurements * np.sqrt(R)` as shown in Equation (2).

## 3. Extended Kalman Filter Algorithm

The EKF is implemented with two main steps: prediction and update.

### 3.1. Prediction Step
The prediction step propagates the previous state estimate forward.

#### 3.1.1. Process Noise Covariance Update
The process noise covariance is updated as:
\[
Q_k = B^2\,\Delta t \tag{3}
\]
- \( Q_k \) is the process noise covariance at time step \( k \).

*Code Reference:* In the `predict` method, the code `self.Q = (self.B) ** 2 * delta_t` implements Equation (3).

#### 3.1.2. State Transition Model (Jacobian)
The state transition Jacobian is:
\[
F_k = 1 - A\,\Delta t \tag{4}
\]
- \( F_k \) is the Jacobian of the state transition function.

*Code Reference:* Calculated as `F = 1 - self.A * delta_t` in the `predict` method.

#### 3.1.3. State Prediction
The predicted (a priori) state is computed by:
\[
\hat{x}_k^- = F_k\,\hat{x}_{k-1} + Q_k\,\epsilon \tag{5}
\]
where:
- \( \hat{x}_k^- \) is the predicted state estimate. `(x_prior)`
- \( \hat{x}_{k-1} \) is the previous state estimate.
- \( \epsilon \) is a sample from a standard normal distribution.

*Code Reference:* Implemented via `self.x_prior = (1 - self.A * delta_t) * self.state + self.Q * np.random.randn()`.

#### 3.1.4. Covariance Prediction
The predicted (a priori) error covariance is given by:
\[
P_k^- = F_k\,P_{k-1}\,F_k + Q_k \tag{6}
\]
where:
- \( P_k^- \) is the predicted covariance. `(P_prior)`
- \( P_{k-1} \) is the previous covariance.

*Code Reference:* See `self.P_prior = F * self.covariance * F + self.Q`.

### 3.2. Update Step
The update step refines the prediction using the new measurement.

#### 3.2.1. Innovation (Measurement Residual)
The innovation is defined as:
\[
y_k = z_k - H\,\hat{x}_k^- \tag{7}
\]
where:
- \( y_k \) is the innovation or measurement residual.
- \( H \) is the measurement Jacobian (set to 1 in this implementation).

*Code Reference:* Implemented as `y = z - H * x_prior` in the `update` method.

#### 3.2.2. Innovation Covariance
The covariance of the innovation is:
\[
S_k = H\,P_k^-\,H + R \tag{8}
\]
where:
- \( S_k \) is the innovation covariance.

*Code Reference:* Implemented via `S = H * P_prior * H + self.R`.

#### 3.2.3. Kalman Gain Computation
The Kalman gain is calculated as:
\[
K_k = \frac{P_k^-\,H}{S_k} \tag{9}
\]
where:
- \( K_k \) is the Kalman gain, determining the weight given to the innovation.

*Code Reference:* Seen in `K = (P_prior * H) / S`.

#### 3.2.4. State Update
The updated (posterior) state estimate is:
\[
\hat{x}_k = \hat{x}_k^- + K_k\,y_k \tag{10}
\]
*Code Reference:* Implemented as `x_posterior = x_prior + K * y`.

#### 3.2.5. Covariance Update
The updated (posterior) error covariance is:
\[
P_k = \left(1 - K_k\,H\right)P_k^- \tag{11}
\]
*Code Reference:* Implemented via `P_posterior = (1 - K * H) * P_prior`.

## 4. Measurement Averaging Procedure
To improve robustness against noise, measurements are averaged over a window of \( N \) time steps. Two aspects are adjusted:

#### 4.1. Adjusted Measurement Noise Covariance
For an averaging window of size \( N \) (denoted by `avg_window`), the measurement noise covariance is adjusted as:
\[
R_{\text{kf}} = \frac{R}{N\,\Delta t} \tag{12}
\]
*Code Reference:* The variable `R_kf` is set using this adjustment.

#### 4.2. Averaged Measurement Computation
The measurement used in the update is the average over the window:
\[
z_k = \frac{1}{N} \sum_{i=k}^{k+N-1} z_i \tag{13}
\]
where:
- \( N \) is the number of measurements in the averaging window.

*Code Reference:* This is implemented by `z = np.mean(measurements[idx_start:idx_end])`.

## 5. Simulation Setup
The simulation is configured with the following parameters:
- \( A = 3.0 \) (drift coefficient)
- \( B = 1.0 \) (noise coefficient)
- Total simulation time \( T = 10.0 \) seconds.
- Finest resolution: \( \Delta t = \frac{1}{\text{PRF}} \) with \(\text{PRF} = 2000\).
- \( R = 0.1 \) (base measurement noise covariance)
- Initial state \( \hat{x}_0 = 1.0 \) and initial covariance \( P_0 = 1.0 \).

## 6. Simulation Process with Alternative Prediction Strategies

The simulation process was implemented with two distinct prediction strategies to investigate their impact on filter performance. Both approaches process measurements in windows of size \( N \) (denoted by `avg_window`), but differ in their handling of temporal propagation:

### 6.1. Multi-Step Prediction Strategy
This method follows a discrete-time approach by executing multiple prediction steps corresponding to the window duration:

1. **Temporal Propagation:**  
   For each averaging window, we perform \( N \) consecutive prediction steps with time increment \( \Delta t \):
   \[
   \hat{x}_k^- = F(\Delta t)\,\hat{x}_{k-1} + Q(\Delta t)\,\epsilon \tag{5}
   \]
   where :
   
   \[\ F(\Delta t) = 1 - A\Delta t \tag{4} \]
    \[\ Q(\Delta t) = B^2\Delta t \tag{3}\]
    The final prediction after \( N \) steps becomes:
   ```python
   x_prior, p_prior = [ekf.predict(delta_t) for _ in range(avg_window)][-1]
   ```

2. **Measurement Update:**  
   Uses the average measurement over the window (Equation 13):
   ```python
   z = np.mean(measurements[idx_start:idx_end])
   x_posterior, p_posterior = ekf.update(z)
   ```

### 6.2. Single-Step Prediction Strategy
This alternative method uses continuous-time formulation by scaling the time interval:

1. **Temporal Propagation:**  
   Performs a single prediction step covering the entire window duration \( N\Delta t \):
   \[
   \hat{x}_k^- = F(N\Delta t)\,\hat{x}_{k-1} + Q(N\Delta t)\,\epsilon \tag{5.1}
   \]
      where :
   
   \[\ F(N\Delta t) = 1 - A(N\Delta t) \tag{4.1} \]
    \[\ Q(N\Delta t) = B^2(N\Delta t) \tag{3.1}\]
     Implemented as:
   ```python
   x_prior, p_prior = ekf.predict(avg_window * delta_t)
   ```

2. **Measurement Update:**  
   Identical to the multi-step approach, using the same averaged measurement (Equation 13).

### 6.3. Implementation Rationale
Both strategies were implemented with conditional logic:
```python
if one_prediction_step:
    x_prior, p_prior = ekf.predict(avg_window * delta_t)
else:
    x_prior, p_prior = [ekf.predict(delta_t) for _ in range(avg_window)][-1]
```
- **Multi-step** (`one_prediction_step=False`): Closely approximates discrete-time system evolution
- **Single-step** (`one_prediction_step=True`): Employs continuous-time formulation for computational efficiency

The covariance propagation (Equation 6) and measurement update (Equations 7-11) remain identical between both approaches. 

## 7. List of Symbols and Constants

- **\( A \)**: Drift coefficient (constant, \(3.0\)).
- **\( B \)**: Noise coefficient (constant, \(1.0\)).
- **\( R \)**: Base measurement noise covariance (constant, \(0.1\)).
- **\( Q_k \)**: Process noise covariance at time step \( k \) (computed as in Equation (3)).
- **\( \Delta t \)**: Time step (finest resolution, \(1/\text{PRF}\)).
- **\(\text{PRF}\)**: Pulse repetition frequency (resolution parameter, \(2000\)).
- **\( T \)**: Total simulation time (seconds, \(10.0\)).
- **\( \eta_k \)**: Process noise (random variable from a standard normal distribution).
- **\( \nu_k \)**: Measurement noise (random variable from a standard normal distribution).
- **\( X_k \)**: True state at time step \( k \) (see Equation (1)).
- **\( \hat{x}_k^- \)**: Predicted (a priori) state estimate (Equation (5)).
- **\( \hat{x}_k \)**: Updated (posterior) state estimate (Equation (10)).
- **\( P_k^- \)**: Predicted (a priori) error covariance (Equation (6)).
- **\( P_k \)**: Updated (posterior) error covariance (Equation (11)).
- **\( H \)**: Measurement Jacobian (constant, \(1\)).
- **\( y_k \)**: Innovation or measurement residual (Equation (7)).
- **\( S_k \)**: Innovation covariance (Equation (8)).
- **\( K_k \)**: Kalman gain (Equation (9)).
- **\( z_k \)**: Measurement at time step \( k \) (Equation (2) and (13)).
- **\( N \)**: Averaging window size (number of \(\Delta t\) steps, denoted by `avg_window`).


## 8. Implementation Notes: Noise Covariance Tuning and Performance Observations

### 8.1. Initial Challenges with Measurement Noise Covariance
During initial implementation with \( R = 0.1 \), we observed:
1. **Poor State Tracking**: The posterior estimates exhibited significant lag and failed to follow the true state trajectory
2. **Covariance Mismatch**: The estimated error covariance \( P_{\text{posterior}} \) (~0.2) and MSE (0.25-0.35) at \( N=1 \).
3. **Theoretical-Experimental Discrepancy**: The empirical MSE at \( N=1 \) was actually higher than the theoretical steady-state variance:
   \[
   \sigma_X^2 = \frac{B^2}{2A} = \frac{1^2}{2 \times 3} = \frac{1}{6} \approx 0.1667 \tag{14}
   \]

### 8.2. Diagnosis and Solution
Analysis revealed two key issues:
1. **Overestimated Measurement Noise**: The initial \( R = 0.1 \) were too high for the system dynamics, leading to:
   - Excessive measurement noise weighting
   - Poor state tracking and high MSE
2. **Filter Conservatism**: The large \( R \) value caused the Kalman gain to underweight measurements, leading to:
   - Reduced state update responsiveness
   - Artificially low \( P \) estimates

After reducing to \( R = 0.0001 \):
- **Posterior MSE Improved** to \( 0.0025 \) at \( N=1 \).
- **Covariance Estimates Became Consistent**: \( P_{\text{posterior}} \) become close to measured MSE.
- **Tracking Performance Enhanced**: Posterior estimates closely followed true state dynamics

### 8.3. Key Implications
1. **Covariance Tuning Critical**: Proper \( R \) calibration is essential for both estimation accuracy and covariance consistency.
2. **Measurement Reliability**: The ratio should be \( R/\sigma_X^2 << 1 \)   to ensure measurements inform state updates.
3. **Validation Requirement**: Theoretical MSE bounds should be verified against empirical results during filter tuning.

## 9. Results
### MSE vs Averaging Window Size
The Mean Squared Error (MSE) was computed for varying averaging window sizes \( N \) to assess filter performance. The results are summarized in the image below:
![MSE vs Averaging Window Size](MSE_vs_N.png)

Kalman Filter tracking results for different averaging window sizes are shown below:
- **\( N = 1 \)**: Single-step prediction strategy

![Kalman Filter Tracking Results 1](KF_tracking1.png)

- **\( N = 100 \)**: Multi-step prediction strategy
![Kalman Filter Tracking Results 100](KF_tracking100.png)
