# Maneuvering Target Tracking

This tutorial demonstrates how to use the Interacting Multiple Model (IMM) filter to track a maneuvering target that switches between constant velocity and constant acceleration dynamics.

---

## Scenario Description

We simulate a target moving in 1D space over 15 seconds:
1. **Constant Velocity (CV)**: For the first 5 seconds, the target moves at a constant velocity of 2.0 m/s.
2. **Constant Acceleration (CA)**: From 5 to 10 seconds, the target accelerates at 1.5 m/s².
3. **Constant Velocity (CV)**: From 10 to 15 seconds, the target stops accelerating and continues at constant velocity.

We compare three tracking strategies:
- A standard **Kalman Filter tuned for CV** (which has low process noise).
- A standard **Kalman Filter tuned for CA** (which has higher process noise to account for acceleration).
- An **IMM Filter** blending both models.

---

## Simulation Code

The full simulation code is located in [tracking_demo.py](file:///Users/minhld/workspace/projects/.research/kalbee/examples/tracking_demo.py).

```python
import numpy as np
from kalbee import KalmanFilter, InteractingMultipleModel
from kalbee.modules.utils.metrics import rmse

# (Set up trajectory and noise)
dt = 0.1
t = np.arange(0, 15, dt)
T = len(t)
# ...

# Initialize IMM blending CV and CA
model_transition = np.array([[0.95, 0.05], [0.05, 0.95]])
model_probabilities = np.array([0.8, 0.2])

imm = InteractingMultipleModel(
    [kf_cv_imm, kf_ca_imm],
    model_transition,
    model_probabilities
)
```

---

## Results and Analysis

When running the simulation, we obtain the following position Root Mean Square Error (RMSE):

| Filter | Position RMSE |
| :--- | :---: |
| Constant Velocity (CV) KF | 3.2481 |
| Constant Acceleration (CA) KF | 0.4712 |
| **IMM Blended Filter** | **0.3755** |

### Discussion

1. **CV Filter Failure**: The CV filter performs poorly (RMSE 3.2481) because its low process noise assumption makes it ignore the measurement deviations during the acceleration phase, lagging far behind the target.
2. **CA Filter Limitations**: The CA filter tracks the maneuver well (RMSE 0.4712) but introduces extra noise and variance during the non-accelerating phases.
3. **IMM Superiority**: The IMM filter achieves the lowest overall error (RMSE 0.3755) by dynamically shifting its belief (model probability) towards the active model.

### IMM Adaptation

The IMM model probabilities adjust dynamically:
- **CV Phase**: Prior CV Probability is high (~0.77).
- **Maneuver Phase**: Prior CA Probability increases to ~0.81 as soon as acceleration is detected.
- **Post-Maneuver**: Prior CV Probability climbs back up once constant velocity resumes.
