To prove that \( \mathbb{E}[W_t^2] = t \) for a Wiener process \( \{W_t\}_{t \geq 0} \), we proceed as follows:

---

### **Step 1: Definition of Wiener Process**
A Wiener process (Brownian motion) satisfies:
1. \( W_0 = 0 \) almost surely.
2. **Independent increments**: For \( 0 \leq s < t \), \( W_t - W_s \) is independent of \( \mathcal{F}_s \) (the filtration up to time \( s \)).
3. **Gaussian increments**: \( W_t - W_s \sim \mathcal{N}(0, t-s) \).
4. **Continuous paths**: \( W_t \) is continuous in \( t \).

---

### **Step 2: Variance of \( W_t \)**
By property (3), over the interval \([0, t]\):  
\[
W_t - W_0 = W_t \sim \mathcal{N}(0, t),
\]
since \( W_0 = 0 \).  

For a Gaussian random variable \( X \sim \mathcal{N}(\mu, \sigma^2) \):  
\[
\mathbb{E}[X^2] = \text{Var}(X) + (\mathbb{E}[X])^2.
\]

For \( W_t \sim \mathcal{N}(0, t) \):  
\[
\mathbb{E}[W_t^2] = \text{Var}(W_t) + (\mathbb{E}[W_t])^2 = t + 0 = t.
\]

