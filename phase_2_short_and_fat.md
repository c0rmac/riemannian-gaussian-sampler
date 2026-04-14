# Efficiency Optimization for "Fat" Stiefel Manifolds: Geometric Duality and Subspace Projections

In global optimization problems on the **Stiefel manifold** $V(n, k)$, the computational cost is often dominated by the ambient dimension $n$. When the system operates under the **"fat" frame condition** ($n < 2k$), the standard algorithmic approach encounters both mathematical singularities and extreme computational inefficiencies[cite: 433, 434]. This article explains how to leverage **geometric duality** and **implicit subspace projections** to make Phase 2 of the exact spectral generative sampling process more efficient[cite: 440, 442].

---

## 1. The Challenge of the "Fat" Frame ($n < 2k$)

When the number of vectors $k$ exceeds half the ambient dimension $n$, the manifold undergoes a "topological lock"[cite: 434]. By the Grassmann dimension theorem, any two $k$-dimensional frames in $\mathbb{R}^n$ must intersect in at least a $(2k-n)$-dimensional subspace [cite: 436, 754-756].

### The Mathematical Singularity
If one attempts to compute a sample directly in the $k$-dimensional space, the stochastic laws of motion encounter a **vector field collapse**[cite: 751]. The Dyson drift governing the boundary interactions includes a term $(n-2k)\cot(\theta_k)$[cite: 762]. Under the fat condition:
* The coefficient $(n-2k)$ becomes strictly negative[cite: 764].
* As the principal angle approaches the forced intersection ($\theta_k \to 0^+$), the drift diverges to $-\infty$[cite: 763, 766].
* This results in an infinite attractive vacuum that causes the Stochastic Differential Equation (SDE) to diverge [cite: 770-774].



---

## 2. The Solution: Geometric Duality

To restore mathematical stability and computational efficiency, we utilize the principle of **geometric duality**[cite: 440]. Optimizing a $k$-frame in $\mathbb{R}^n$ is mathematically equivalent to optimizing its orthogonal complement, which has dimension $k' = n - k$[cite: 440, 442].

Since $n < 2k$, it is strictly guaranteed that $n \ge 2k'$[cite: 433, 443]. By mapping the computation to this "dual" space, we transition back to the "tall and skinny" regime where the spectral sampler is well-behaved and computationally cheaper, as $k' < k$[cite: 441, 443].

---

## 3. Efficient Phase 2 Computation ($O(nk'^2)$)

In the standard Algorithm 4.2, Phase 2 requires $O(n^3)$ operations, such as generating full orthogonal matrices $Q$ and $h_{\text{sub}}$[cite: 1025, 1032]. We can reduce this to $O(nk'^2)$ through the following optimizations:

### A. Implicit Fiber Conjugation via Thin QR
Instead of constructing a full $(n-k') \times (n-k')$ stabilizer matrix $h_{\text{sub}}$, we only need to randomize the orientation of the $k'$-dimensional frame[cite: 1025, 1028].
1. **Thin Draw:** Generate an $(n-k') \times k'$ Gaussian matrix.
2. **Reduced QR:** Perform a thin QR decomposition to obtain an orthogonal $k'$-frame.
3. **Complexity:** This reduces the cost from $O((n-k')^3)$ to $O((n-k')k'^2)$.



### B. Bypassing Full Basis Completion ($Q$)
Step 5 of Algorithm 4.2 usually requires an $n \times n$ orthogonal completion $Q = [\widehat{M}_{k'} \mid \widehat{M}_{k'}^\perp]$[cite: 1032]. This is an $O(n^3)$ operation. We can bypass this by projecting the canonical sample directly[cite: 1031, 1033]:
1. **Consensus Frame:** Let $\widehat{M}_{k'} = \widehat{M}_k^\perp$ be the dual consensus point[cite: 440, 1032].
2. **Projective Noise:** Draw $n \times k'$ Gaussian noise $Z$.
3. **Orthogonal Projection:** Compute $Z^{\perp} = Z - \widehat{M}_{k'}(\widehat{M}_{k'}^\top Z)$. This removes the component sitting in the consensus frame[cite: 1032].
4. **Thin Orthogonalization:** Perform a thin QR on $Z^{\perp}$ to obtain $U \in \mathbb{R}^{n \times k'}$.
5. **Recombination:** Assemble the dual sample $X'$ using only $n \times k'$ matrix multiplications: $X' = \widehat{M}_{k'} \Theta_{11} + U \Theta_{21}$.



---

## 4. Summary of Efficiency Gains

By combining geometric duality with implicit projections, we transform the computational profile for fat Stiefel manifolds:

| Task | Standard Complexity | Optimized Dual Complexity |
| :--- | :--- | :--- |
| **Spectral Sampling** | Divergent ($n < 2k$) | Stable ($n \ge 2k'$) [cite: 443, 751] |
| **Fiber Generation** | $O(n^3)$ | $O(nk'^2)$ |
| **Basis Transport** | $O(n^3)$ | $O(nk'^2)$ |
| **Memory** | $O(n^2)$ | $O(nk')$ |

Once the dual sample $X' \in V(n, k')$ is obtained, the final $k$-dimensional sample is recovered by taking the orthogonal complement $X = (X')^\perp$[cite: 440, 442]. This ensures that the global stationarity of the CBO process is preserved with zero discretization error[cite: 980, 1002].