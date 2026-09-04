from typing import Any, Optional, Tuple


class DifferentiableKalmanFilter:
    """
    PyTorch Differentiable Kalman Filter module.

    Enables end-to-end gradient-based training of process models (F, Q)
    and measurement models (H, R) using backpropagation.

    Requires PyTorch (`pip install torch`).
    """

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        F: Optional[Any] = None,
        Q: Optional[Any] = None,
        H: Optional[Any] = None,
        R: Optional[Any] = None,
    ):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for DifferentiableKalmanFilter. Install with: pip install torch")

        self.torch = torch
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        self.F = F if F is not None else torch.eye(state_dim)
        self.Q = Q if Q is not None else torch.eye(state_dim) * 0.01
        self.H = H if H is not None else torch.eye(meas_dim, state_dim)
        self.R = R if R is not None else torch.eye(meas_dim) * 0.1

    def predict(self, state: Any, cov: Any, F: Optional[Any] = None, Q: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Differentiable predict step.
        """
        torch = self.torch
        F = F if F is not None else self.F
        Q = Q if Q is not None else self.Q

        # state: (B, n, 1), cov: (B, n, n)
        state_pred = torch.matmul(F, state)
        cov_pred = torch.matmul(torch.matmul(F, cov), F.transpose(-1, -2)) + Q

        return state_pred, cov_pred

    def update(
        self, state_pred: Any, cov_pred: Any, z: Any, H: Optional[Any] = None, R: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Differentiable update step.
        """
        torch = self.torch
        H = H if H is not None else self.H
        R = R if R is not None else self.R

        # z: (B, m, 1)
        y = z - torch.matmul(H, state_pred)
        S = torch.matmul(torch.matmul(H, cov_pred), H.transpose(-1, -2)) + R
        S_inv = torch.linalg.inv(S)

        K = torch.matmul(torch.matmul(cov_pred, H.transpose(-1, -2)), S_inv)

        state_upd = state_pred + torch.matmul(K, y)
        I_n = torch.eye(self.state_dim, device=state_pred.device)
        I_KH = I_n - torch.matmul(K, H)
        cov_upd = torch.matmul(torch.matmul(I_KH, cov_pred), I_KH.transpose(-1, -2)) + torch.matmul(
            torch.matmul(K, R), K.transpose(-1, -2)
        )

        return state_upd, cov_upd
