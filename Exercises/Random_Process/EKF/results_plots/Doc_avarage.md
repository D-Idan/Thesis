
---  

# 📌 Improving EKF with a Rolling Average of Measurements  

## **1. Introduction**  
In this exercise, we enhance the Extended Kalman Filter (EKF) by using a **rolling average of measurements** within each time step \( \Delta t \). This modification aims to reduce the impact of high-frequency measurement noise while maintaining an accurate estimate of the system state.  

---

## **2. Problem Setup**  

The given **stochastic differential equation (SDE)** is:  

\[
\dot{X} = -A X + B N
\]

where:  
- \( A = 3.0 \) (Drift coefficient)  
- \( B = 1.0 \)  
- \( N \sim \mathcal{N}(0,1) \) (White noise)  

The system is estimated using an **Extended Kalman Filter (EKF)**, and we apply **rolling average smoothing** to the **measurements** before updating the state.

---

## **3. Why Use a Rolling Average?**  

In the previous implementation, each measurement \( Z_k \) was used directly in the update step. However, sensor measurements can contain **high-frequency noise**, causing fluctuations in the state estimate.  

By computing a rolling average over multiple measurements within each time interval \( \Delta t \), we:  

✅ **Reduce Measurement Noise Before Processing**  
✅ **Improve Stability in State Estimation**  
✅ **Avoid Overreacting to Sensor Noise**  

---

## **4. Updated Equations**  

### **State-Space Representation**  
The system is defined as:  

\[
X[i] = X[i - 1] (1 + \Delta t \cdot A) + \Delta t \cdot B \cdot N_n[i]
\]

### **Rolling Average for Measurements**  

Instead of using a single raw measurement \( Z_k \), we define:  

\[
Z_{\text{avg}}[k] = \frac{1}{N} \sum_{j=1}^{N} Z_j
\]

where:  
- \( N \) is the number of sensor readings collected within \( \Delta t \).  
- \( Z_j \) are the individual measurements within that time window.  

Then, in the **EKF update step**, we use the averaged measurement \( Z_{\text{avg}}[k] \) instead of a single \( Z_k \):

\[
y = Z_{\text{avg}} - H \cdot X_{\text{prior}}
\]

---

## **5. Implementation Details**  

### **Changes to the Code**  
1️⃣ **Collect multiple sensor readings** within each \( \Delta t \).  
2️⃣ **Compute the rolling average** of those readings.  
3️⃣ **Use the averaged measurement** in the EKF update step.  

---

## **7. Expected Results**  

### ✅ **Advantages of Using a Rolling Average**  
1. **Smoother Estimates Without Lag**  
   - High-frequency noise is reduced without distorting the true state.  
2. **Less Impact of Measurement Outliers**  
   - A single bad sensor reading won’t heavily affect the filter.  
3. **Improved Stability in Tracking**  
   - The EKF doesn’t overreact to random fluctuations in sensor data.  

---

## **8. When This Approach Might Fail**  

⚠️ **Rolling Averages Can Lead to Worse Results in These Cases:**  

1️⃣ **Fast-Changing Systems:**  
   - If the system state changes rapidly, averaging measurements over time can introduce **lag**, causing the EKF to react too slowly.  

2️⃣ **Small Measurement Noise (Already Clean Data):**  
   - If the measurement noise is already small, applying a rolling average can **smooth out real variations** rather than just noise.  

3️⃣ **Very Large \( \Delta t \):**  
   - If \( \Delta t \) is large and very few measurements are taken per step, the averaging might **remove too much information**, reducing filter effectiveness.  

---

## **9. Conclusion**  

Using a rolling average on sensor measurements improves the **stability** of the EKF by reducing noise before updating the state estimate. However, it should be used **cautiously in dynamic systems** where real changes might be smoothed out.  

```
---
