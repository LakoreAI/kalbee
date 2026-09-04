from typing import List
import numpy as np

from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.utils.linalg import safe_inv


class InteractingMultipleModel(BaseFilter):
    """
    Interacting Multiple Model (IMM) Filter.

    Maintains multiple filter hypotheses in parallel and blends their estimates
    based on model transition probabilities and measurement likelihoods.
    """

    def __init__(
        self,
        filters: List[BaseFilter],
        model_transition: np.ndarray,
        model_probabilities: np.ndarray,
    ):
        """
        Initialize the IMM Filter.

        Args:
            filters: List of filters (subclasses of BaseFilter).
            model_transition: Probability matrix of transitioning from model i to j (M x M).
            model_probabilities: Prior probabilities of each model (M).
        """
        self.filters = filters
        self.model_transition = np.asanyarray(model_transition).astype(float)
        self.model_probabilities = (
            np.asanyarray(model_probabilities).astype(float).flatten()
        )

        # Combine states to set initial IMM state and covariance
        self._combine_states()

        super().__init__(
            state=self.state,
            covariance=self.covariance,
        )

    def _combine_states(self):
        """Combine states and covariances from all filters using model probabilities."""
        mu = self.model_probabilities
        M = len(self.filters)
        state_dim = self.filters[0].state.shape[0]

        # Combined state: x = sum(mu_j * x_j)
        self.state = np.zeros((state_dim, 1))
        for j in range(M):
            self.state += mu[j] * self.filters[j].state

        # Combined covariance: P = sum(mu_j * (P_j + (x_j - x)(x_j - x).T))
        self.covariance = np.zeros((state_dim, state_dim))
        for j in range(M):
            diff = self.filters[j].state - self.state
            self.covariance += mu[j] * (self.filters[j].covariance + diff @ diff.T)

    def predict(self, dt: float = 1.0, **kwargs) -> np.ndarray:
        """
        Predict step for IMM:
        1. Mix states and covariances of all filters based on transition matrix.
        2. Perform prediction for each individual filter.
        3. Combine predictions for the global state estimate.
        """
        M = len(self.filters)
        mu = self.model_probabilities
        P_trans = self.model_transition
        state_dim = self.state.shape[0]

        # 1. Compute normalization factor: c_j = sum_i (p_ij * mu_i)
        c = np.dot(P_trans.T, mu)

        # 2. Compute mixing weights: mu_{i|j} = p_ij * mu_i / c_j
        mixing_weights = np.zeros((M, M))
        for j in range(M):
            if c[j] > 0:
                mixing_weights[:, j] = P_trans[:, j] * mu / c[j]
            else:
                mixing_weights[:, j] = P_trans[:, j] * mu

        # 3. Compute mixed states and covariances
        mixed_states = []
        mixed_covs = []
        for j in range(M):
            x_0j = np.zeros((state_dim, 1))
            for i in range(M):
                x_0j += mixing_weights[i, j] * self.filters[i].state
            mixed_states.append(x_0j)

            P_0j = np.zeros((state_dim, state_dim))
            for i in range(M):
                diff = self.filters[i].state - x_0j
                P_0j += mixing_weights[i, j] * (
                    self.filters[i].covariance + diff @ diff.T
                )
            mixed_covs.append(P_0j)

        # 4. Predict each individual filter using the mixed initializations
        for j in range(M):
            self.filters[j].state = mixed_states[j]
            self.filters[j].covariance = mixed_covs[j]
            self.filters[j].predict(dt=dt, **kwargs)

        # Update model probabilities prior
        self.model_probabilities = c

        self._combine_states()
        return self.state

    def update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """
        Update step for IMM:
        1. Update each individual filter with the measurement.
        2. Compute measurement likelihood for each model.
        3. Update model probabilities.
        4. Combine estimates for global state.
        """
        z = measurement
        M = len(self.filters)
        c = self.model_probabilities

        # 1. Update each individual filter and compute likelihoods
        likelihoods = np.zeros(M)
        for j in range(M):
            self.filters[j].update(z, **kwargs)
            likelihoods[j] = self._compute_likelihood(self.filters[j], z)

        # 2. Update model probabilities: mu_j = c_j * L_j / sum(c_k * L_k)
        scaled_likelihoods = c * likelihoods
        sum_likelihoods = np.sum(scaled_likelihoods)

        if sum_likelihoods > 0:
            self.model_probabilities = scaled_likelihoods / sum_likelihoods
        else:
            # Fallback to uniform if all likelihoods are zero/underflow
            self.model_probabilities = np.ones(M) / M

        self._combine_states()
        return self.state

    def _compute_likelihood(self, filter_obj: BaseFilter, z: np.ndarray) -> float:
        """Compute the measurement likelihood of a filter."""
        last_y = getattr(filter_obj, "last_y", None)
        last_S = getattr(filter_obj, "last_S", None)

        if last_y is not None and last_S is not None:
            y = last_y
            S = last_S
        else:
            # Fallback computation if filter doesn't save last_y/last_S
            H = getattr(filter_obj, "measurement_matrix", None)
            R = getattr(filter_obj, "measurement_covariance", None)

            if H is not None and R is not None:
                y = z - H @ filter_obj.state
                S = H @ filter_obj.covariance @ H.T + R
            else:
                h_func = getattr(filter_obj, "measurement_function", None)
                H_func = getattr(filter_obj, "measurement_jacobian", None)
                R = getattr(filter_obj, "measurement_covariance", None)

                if h_func is not None and R is not None:
                    y = z - h_func(filter_obj.state)
                    if callable(H_func):
                        H = H_func(filter_obj.state)
                    elif isinstance(H_func, np.ndarray):
                        H = H_func
                    else:
                        return 1.0
                    S = H @ filter_obj.covariance @ H.T + R
                else:
                    return 1.0

        m = z.shape[0]
        det_S = np.linalg.det(S)
        if det_S <= 0:
            det_S = 1e-10

        S_inv = safe_inv(S)
        exponent = -0.5 * (y.T @ S_inv @ y).item()
        exponent = np.clip(exponent, -700, 700)

        likelihood = (1.0 / np.sqrt((2 * np.pi) ** m * det_S)) * np.exp(exponent)
        return max(likelihood, 1e-100)
