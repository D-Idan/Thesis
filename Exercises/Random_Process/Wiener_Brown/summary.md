**Paper Summary: Wiener-Brownian Motion and Its Key Properties**  
*Prepared for Thesis Research*

---

### **1. Definition of Wiener-Brownian Motion**  
Wiener-Brownian motion (WBM), named after Norbert Wiener and Robert Brown, is a continuous-time stochastic process \( \{W_t\}_{t \geq 0} \) defined by the following properties:  
1. **Initial Condition**: \( W_0 = 0 \) almost surely.  
2. **Independent Increments**: For \( 0 \leq t_1 < t_2 < \dots < t_n \), increments \( W_{t_{i+1}} - W_{t_i} \) are independent.  
3. **Gaussian Distribution**: Increments \( W_t - W_s \) (\( t > s \)) follow \( \mathcal{N}(0, t-s) \).  
4. **Continuous Paths**: \( W_t \) is continuous in \( t \) almost surely.  

**Sources**:  
- Karatzas & Shreve (1991), *Brownian Motion and Stochastic Calculus*.  
- Øksendal (2003), *Stochastic Differential Equations*.  

---

### **2. Key Properties**  
- **Martingale Property**: \( \mathbb{E}[W_t | \mathcal{F}_s] = W_s \) for \( s < t \).  
- **Markov Property**: Future behavior depends only on the current state.  
- **Self-Similarity**: \( W_{at} \overset{d}{=} \sqrt{a} W_t \) for \( a > 0 \).  
- **Non-Differentiability**: Paths are nowhere differentiable with probability 1.  

---

### **3. Independent Increments**  
Increments over disjoint intervals are statistically independent. This property:  
- Implies WBM is a **Lévy process**.  
- Enables decomposing complex stochastic problems into simpler intervals.  

---

### **4. Derivative as White Noise**  
Though \( W_t \) is nowhere differentiable, its generalized derivative \( \xi(t) = \frac{dW_t}{dt} \) is interpreted as **white noise**:  
- **Spectral Density**: Flat (equal intensity across frequencies).  
- **Autocorrelation**: \( \mathbb{E}[\xi(t)\xi(s)] = \delta(t-s) \), where \( \delta \) is the Dirac delta.  

**Source**:  
- Gardiner (2004), *Handbook of Stochastic Methods*.  

---

### **5. Discrete vs. Continuous Approximations**  
- **Discrete**: Scaled random walks \( X_n = \sum_{k=1}^n \epsilon_k \sqrt{\Delta t} \), where \( \epsilon_k \sim \mathcal{N}(0,1) \). As \( \Delta t \to 0 \), \( X_n \to W_t \) (Donsker’s theorem).  
- **Continuous**: Direct construction via Wiener’s measure on the space of continuous functions.  

**Differences**:  
- Discrete approximations have piecewise-constant jumps; continuous WBM is almost surely smooth-free.  
- Discrete models are finite-dimensional, while WBM is inherently infinite-dimensional.  

---

### **6. Fluctuation and Quadratic Variation**  
- **Fluctuation**: WBM paths exhibit **infinite total variation** but **finite quadratic variation**:  
  \[ \lim_{\| \Pi \| \to 0} \sum_{i=1}^n (W_{t_i} - W_{t_{i-1}})^2 = t \quad \text{(a.s.)} \]  
- **Quadratic Variation**:  
  - Critical in Itô calculus for defining integrals like \( \int_0^t f(s) dW_s \).  
  - Contrasts with smooth functions, whose quadratic variation is zero.  

**Source**:  
- Revuz & Yor (1999), *Continuous Martingales and Brownian Motion*.  

---

### **7. Applications and Significance**  
- **Stochastic Calculus**: Foundation for Itô and Stratonovich integrals.  
- **Physics**: Models diffusion processes.  
- **Finance**: Underlies the Black-Scholes equation for option pricing.  

---

### **Conclusion**  
Wiener-Brownian motion is a cornerstone of stochastic processes, characterized by independent increments, continuity, and non-differentiability. Its quadratic variation enables stochastic integration, while discrete approximations bridge theory with numerical applications. Mastery of WBM is essential for advanced topics in stochastic analysis and their applications in fields like mathematical finance and statistical mechanics.

**Further Reading**:  
- Original work: Wiener (1923), *Differential Space*.  
- Historical context: Nelson (1967), *Dynamical Theories of Brownian Motion*.  
