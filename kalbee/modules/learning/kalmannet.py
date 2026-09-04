from typing import Any, Tuple


class KalmanNet:
    """
    KalmanNet: Hybrid Model-Based Neural Filter architecture.

    Combines the structural recursion of a Kalman Filter with a Recurrent
    Neural Network (GRU) that learns to predict the optimal Kalman Gain directly
    from measurement sequences.

    Requires PyTorch (`pip install torch`).
    """

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        hidden_dim: int = 32,
    ):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required for KalmanNet. Install with: pip install torch")

        self.torch = torch
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.hidden_dim = hidden_dim

        # Define internal GRU-based gain estimator
        class GainRNN(nn.Module):
            def __init__(self, in_dim, h_dim, out_dim):
                super().__init__()
                self.gru = nn.GRUCell(in_dim, h_dim)
                self.fc = nn.Linear(h_dim, out_dim)

            def forward(self, x, h):
                h_next = self.gru(x, h)
                gain_flat = self.fc(h_next)
                return gain_flat, h_next

        input_dim = meas_dim + state_dim  # innovation + state diff
        output_dim = state_dim * meas_dim

        self.rnn = GainRNN(input_dim, hidden_dim, output_dim)

    def step(
        self,
        state_pred: Any,
        z: Any,
        H: Any,
        h_rnn: Any,
    ) -> Tuple[Any, Any]:
        """
        Single recursive step of KalmanNet.

        Args:
            state_pred: Predicted state (B, n, 1).
            z: Measurement (B, m, 1).
            H: Measurement matrix (m, n) or (B, m, n).
            h_rnn: GRU hidden state (B, hidden_dim).

        Returns:
            Tuple of (state_updated, h_rnn_next).
        """
        torch = self.torch
        B = state_pred.shape[0]

        # Compute innovation: y = z - H @ x_pred
        y = z - torch.matmul(H, state_pred)

        # Feature vector for RNN input
        y_feat = y.squeeze(-1)  # (B, m)
        x_feat = state_pred.squeeze(-1)  # (B, n)
        in_feat = torch.cat([y_feat, x_feat], dim=-1)

        # Predict Kalman Gain matrix K (B, n, m) via RNN
        k_flat, h_next = self.rnn(in_feat, h_rnn)
        K = k_flat.view(B, self.state_dim, self.meas_dim)

        # Apply structural update: x_upd = x_pred + K @ y
        state_upd = state_pred + torch.matmul(K, y)

        return state_upd, h_next

    def forward_sequence(
        self,
        measurements: Any,
        F: Any,
        H: Any,
        x0: Any,
    ) -> Any:
        """
        Filter a full sequence of measurements (T, B, m, 1).
        """
        torch = self.torch
        T, B, _, _ = measurements.shape

        h_rnn = torch.zeros(B, self.hidden_dim, device=measurements.device)
        x_curr = x0
        states_out = []

        for t in range(T):
            # Predict: x_pred = F @ x
            x_pred = torch.matmul(F, x_curr)
            z_t = measurements[t]

            # Update via KalmanNet step
            x_curr, h_rnn = self.step(x_pred, z_t, H, h_rnn)
            states_out.append(x_curr)

        return torch.stack(states_out, dim=0)
