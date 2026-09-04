from kalbee.modules.filters.base import BaseFilter
from kalbee.modules.filters.kf_filter import KalmanFilter
from kalbee.modules.filters.abg_filter import AlphaBetaGammaFilter
from kalbee.modules.filters.ekf_filter import ExtendedKalmanFilter
from kalbee.modules.filters.ukf_filter import UnscentedKalmanFilter
from kalbee.modules.filters.particle_filter import ParticleFilter
from kalbee.modules.filters.enkf_filter import EnsembleKalmanFilter
from kalbee.modules.filters.information_filter import InformationFilter
from kalbee.modules.filters.adaptive_kf import AdaptiveKalmanFilter
from kalbee.modules.filters.square_root_kf import SquareRootKalmanFilter
from kalbee.modules.filters.vectorized_kf import VectorizedKalmanFilter
from kalbee.modules.filters.fading_memory_kf import FadingMemoryKalmanFilter
from kalbee.modules.filters.hinfinity_filter import HInfinityFilter
from kalbee.modules.filters.sigma_point_ukf import SigmaPointUKF


class AutoFilter:
    """
    Factory class for creating filter instances.
    """

    @staticmethod
    def from_filter(*args, mode: str = "kalmanfilter", **kwargs) -> BaseFilter:
        """
        Create a filter instance based on the specified mode.

        Args:
            mode: The type of filter to create.
                  Options:
                  - 'kalmanfilter', 'kf', 'kalman'
                  - 'extendedkalmanfilter', 'ekf'
                  - 'unscentedkalmanfilter', 'ukf'
                  - 'alphabetagamma', 'abg'
                  - 'particlefilter', 'pf', 'particle'
                  - 'ensemblekalmanfilter', 'enkf', 'ensemble'
                  - 'informationfilter', 'if', 'information'
                  - 'adaptivekalmanfilter', 'akf', 'adaptive'
                  - 'squarerootkalmanfilter', 'srkf'
                  - 'vectorizedkalmanfilter', 'vkf'
                  - 'fadingmemorykalmanfilter', 'fmkf', 'fading'
            *args: Positional arguments passed to the filter constructor.
            **kwargs: Keyword arguments passed to the filter constructor.

        Returns:
            An instance of a BaseFilter subclass.

        Raises:
            ValueError: If the mode is unknown.
        """
        mode_clean = mode.lower().replace("_", "").replace("-", "")

        if mode_clean in ["kalmanfilter", "kf", "kalman"]:
            return KalmanFilter(*args, **kwargs)

        elif mode_clean in ["extendedkalmanfilter", "ekf"]:
            return ExtendedKalmanFilter(*args, **kwargs)

        elif mode_clean in ["alphabetagamma", "abg"]:
            return AlphaBetaGammaFilter(*args, **kwargs)

        elif mode_clean in ["unscentedkalmanfilter", "ukf"]:
            return UnscentedKalmanFilter(*args, **kwargs)

        elif mode_clean in ["particlefilter", "pf", "particle"]:
            return ParticleFilter(*args, **kwargs)

        elif mode_clean in ["ensemblekalmanfilter", "enkf", "ensemble"]:
            return EnsembleKalmanFilter(*args, **kwargs)

        elif mode_clean in ["informationfilter", "if", "information"]:
            return InformationFilter(*args, **kwargs)

        elif mode_clean in ["adaptivekalmanfilter", "akf", "adaptive"]:
            return AdaptiveKalmanFilter(*args, **kwargs)

        elif mode_clean in ["squarerootkalmanfilter", "srkf"]:
            return SquareRootKalmanFilter(*args, **kwargs)

        elif mode_clean in ["vectorizedkalmanfilter", "vkf"]:
            return VectorizedKalmanFilter(*args, **kwargs)

        elif mode_clean in ["fadingmemorykalmanfilter", "fmkf", "fading"]:
            return FadingMemoryKalmanFilter(*args, **kwargs)

        elif mode_clean in ["hinfinityfilter", "hinfinity", "hinf"]:
            return HInfinityFilter(*args, **kwargs)

        elif mode_clean in ["sigmapointukf", "spukf", "sigmaukf"]:
            return SigmaPointUKF(*args, **kwargs)

        else:
            raise ValueError(
                f"Unknown filter mode: '{mode}'. Available modes: "
                "kf, ekf, ukf, abg, pf, enkf, if, akf, srkf, vkf, fmkf, hinf, spukf."
            )
