---
name: state-estimation-expert
description: Guidelines for implementing and refining state estimation algorithms like Kalman Filters.
---

# State Estimation Expert Skill

This skill provides comprehensive guidelines for designing, implementing, and verifying state estimation algorithms (Kalman Filters, EKF, UKF, alpha-beta-gamma).

## 1. Mathematical Rigor & Numerical Stability

- **Joseph Form Update**: For the covariance update in the Kalman Filter, use the Joseph form to ensure the covariance matrix $P$ remains positive semi-definite and symmetric:
  $$P_k = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^T + K_k R_k K_k^T$$
  instead of the simpler but less stable $P_k = (I - K_k H_k) P_{k|k-1}$.
- **Symmetry Correction**: Force symmetry on $P$ after each update: `P = (P + P.T) / 2`.
- **Dimension Validation**: Always validate that $F$ is $n \times n$, $H$ is $m \times n$, $x$ is $n \times 1$, and $z$ is $m \times 1$.

## 2. API Design Principles

- **Modular Base Class**: Always inherit from `BaseFilter`.
- **Time-Step (dt) Awareness**: Use `dt` in `predict()` to dynamically compute the state transition matrix $F$ if it depends on time.
- **Optional Matrices**: Allow filters to be initialized with either fixed matrices or functions (for non-linear cases).

## 3. Implementation Checklist

1. [ ] Check for matrix dimension mismatches.
2. [ ] Use `np.asanyarray` to handle list inputs.
3. [ ] Implement `measure()` to map state to measurement space (useful for residuals).
4. [ ] Ensure all state/covariance matrices are initialized with float types.
5. [ ] Add `**kwargs` to `predict` and `update` for future flexibility.

## 4. Verification Strategies

- **Unit Tests**:
  - Verify state evolves correctly for constant velocity/acceleration models.
  - Verify $P$ decreases after a measurement update.
  - Test edge cases: zero measurement noise, zero process noise.
- **Simulation**: Create a simple 1D or 2D particle simulation to compare ground truth vs. estimated state.
