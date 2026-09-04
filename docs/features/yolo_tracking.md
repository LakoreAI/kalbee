# YOLO Object Tracking with State Estimation

In real-world computer vision applications, object detection algorithms like YOLO (You Only Look Once) provide bounding boxes of detected objects. However, raw YOLO detections often suffer from:

1. **Jitter**: Bounding boxes fluctuate slightly frame-to-frame due to pixel-level noise.
2. **Occlusions**: Objects may be briefly hidden behind columns, trees, or other objects, causing YOLO to miss detections for several frames.
3. **Maneuvers**: Fast-moving targets can change directions rapidly, introducing tracking lag.

Integrating Kalman Filters and estimators with YOLO output helps smooth the path and predict the object's position during occlusions.

---

## Scenario Setup

We simulate a target moving diagonally on a $640 \times 480$ frame. At $t = 5.0$ seconds, the target executes a sharp accelerating turn. 
- **Noise**: Standard deviation of 8.0 pixels on bounding box coordinates.
- **Occlusion**: From $t = 3.0$s to $4.5$s, the YOLO detector fails to detect the object (no measurements received).

The tracking script compares four filters:
1. **Standard Kalman Filter (KF)**
2. **Square-Root Kalman Filter (SRKF)**
3. **Particle Filter (PF)**
4. **Interacting Multiple Model (IMM) Filter**

The script is available at [yolo_tracking.py](file:///Users/minhld/workspace/projects/.research/kalbee/examples/yolo_tracking.py).

---

## Comparison Results

| Algorithm | Tracking RMSE (px) | Notes |
| :--- | :---: | :--- |
| Standard Kalman Filter | 22.79 | Suffers from lag during maneuvers. |
| Square-Root KF | 22.79 | Mathematically identical to KF, but holds Cholesky factor updates preventing numerical drift. |
| Particle Filter | 56.12 | Extrapolates well but requires tuning of process noise. |
| **IMM Blended Filter** | **6.34** | **Adapts instantly to the maneuver, minimizing tracking error.** |

---

## Analysis of Key Phases

### 1. Occlusion (Steps 30 to 45)
During the occlusion window, the YOLO detector output is missing (`None`).
- Standard filters run in **prediction-only** mode (updating covariance and state projections without measurements).
- The IMM filter maintains stable extrapolation, preserving velocity history.

### 2. Maneuver (Step 50+)
When the target makes a sharp turn:
- Standard KF and SRKF show significant overshoot and lag because they are tuned for constant velocity.
- The IMM filter combines a CV model with a high-acceleration CA model. It detects the mismatch in measurement innovations, shifts its model probability to the CA model, and recovers the target's position with minimal lag (RMSE 6.34 px).
