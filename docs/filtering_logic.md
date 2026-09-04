# Filtering Logic & Design

## Numerical Stability
`kalbee` prioritizes numerical stability. 

### Joseph Form
For covariance updates in Kalman Filters, we use the **Joseph Form**:
$$P = (I - KH)P(I - KH)^T + KRK^T$$

This ensures that the covariance matrix $P$:
1. Remains symmetric.
2. Remains positive semi-definite.

Standard implementations often use $P = (I - KH)P$, which is computationally cheaper but can lead to numerical instability (negative variances) due to floating-point errors.

## Extensibility
The `BaseFilter` class allows you to implement custom filters by simply defining `predict` and `update`.

```python
class MyCustomFilter(BaseFilter):
    def predict(self, dt, **kwargs):
        # ... logic
        return self.state
        
    def update(self, measurement, **kwargs):
        # ... logic
        return self.state
```
