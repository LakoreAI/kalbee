from kalbee.modules.filters import (
    BaseFilter,
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    AlphaBetaGammaFilter,
    ParticleFilter,
    EnsembleKalmanFilter,
    InformationFilter,
    AdaptiveKalmanFilter,
    SquareRootKalmanFilter,
    InteractingMultipleModel,
    VectorizedKalmanFilter,
    FadingMemoryKalmanFilter,
    HInfinityFilter,
    CubatureKalmanFilter,
    FederatedKalmanFilter,
    RaoBlackwellizedParticleFilter,
    CholeskyKalmanFilter,
    SquareRootUKF,
    AsyncKalmanFilter,
    InvariantEKF,
    SO3,
    SE3,
    VariationalBayesKalmanFilter,
    SimplexSigmaPoints,
    MerweScaledSigmaPoints,
    JulierSigmaPoints,
    SigmaPointUKF,
    AutoFilter,
    # Procedural API
    kf_predict,
    kf_update,
    ekf_predict,
    ekf_update,
    compute_kalman_gain,
    compute_nis,
    compute_nees,
    # State classes
    FilterState,
    FilterConfig,
)
from kalbee.modules.smoothers import (
    RTSSmoother,
    ExtendedRTSSmoother,
    FixedLagSmoother,
    UnscentedRTSSmoother,
)
from kalbee.modules.fusion import (
    covariance_intersection,
    sequential_covariance_intersection,
    track_to_track_fusion,
    AsyncSensorBuffer,
)
from kalbee.modules.integration import (
    filter_dataframe,
    filter_series,
    FactorGraphExporter,
)
from kalbee.modules.utils import rmse, nees, nis, log_likelihood
from kalbee.modules.utils.jacobian import (
    numerical_jacobian,
    numerical_transition_jacobian,
    numerical_measurement_jacobian,
)
from kalbee.modules.utils.gating import (
    chi2_gate,
    mahalanobis_distance,
    ellipsoidal_gate,
    gated_update,
)
from kalbee.modules.utils.consistency import (
    nis_test,
    nees_test,
    innovation_whiteness_test,
)
from kalbee.modules.utils.diagnostics import FilterDiagnostics
from kalbee.modules.utils.outlier_detector import Chi2OutlierDetector, DetectionResult
from kalbee.modules.utils.plotting import (
    plot_trajectory,
    plot_covariance,
    plot_innovations,
)
from kalbee.modules.learning.auto_tune import tune_kalman_filter, quick_tune, TuneResult
from kalbee.modules.learning import OnlineEM
from kalbee.experiments import run_experiment
from kalbee.models import (
    constant_velocity,
    constant_acceleration,
    constant_turn,
    imu_velocity_control,
    discrete_white_noise,
    position_measurement_model,
    quaternion_normalize,
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_to_rotation_matrix,
    quaternion_angular_error,
    attitude_transition,
    attitude_transition_jacobian,
    gravity_measurement,
    gravity_measurement_jacobian,
)
from kalbee.tracking import (
    MultiObjectTracker,
    Track,
    iou_matrix,
    mahalanobis_matrix,
    associate,
    JPDAAssociation,
    PMBMTracker,
    BernoulliTarget,
)
from kalbee.learning import em_kalman, EMResult

__version__ = "0.6.0"

__all__ = [
    # Filters
    "BaseFilter",
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "AlphaBetaGammaFilter",
    "ParticleFilter",
    "EnsembleKalmanFilter",
    "InformationFilter",
    "AdaptiveKalmanFilter",
    "SquareRootKalmanFilter",
    "InteractingMultipleModel",
    "VectorizedKalmanFilter",
    "FadingMemoryKalmanFilter",
    "HInfinityFilter",
    "CubatureKalmanFilter",
    "FederatedKalmanFilter",
    "RaoBlackwellizedParticleFilter",
    "CholeskyKalmanFilter",
    "SquareRootUKF",
    "AsyncKalmanFilter",
    "InvariantEKF",
    "SO3",
    "SE3",
    "VariationalBayesKalmanFilter",
    # Sigma Points
    "SimplexSigmaPoints",
    "MerweScaledSigmaPoints",
    "JulierSigmaPoints",
    "SigmaPointUKF",
    # AutoFilter
    "AutoFilter",
    # Procedural API
    "kf_predict",
    "kf_update",
    "ekf_predict",
    "ekf_update",
    "compute_kalman_gain",
    "compute_nis",
    "compute_nees",
    # State Classes
    "FilterState",
    "FilterConfig",
    # Smoothers
    "RTSSmoother",
    "ExtendedRTSSmoother",
    "FixedLagSmoother",
    "UnscentedRTSSmoother",
    # Fusion
    "covariance_intersection",
    "sequential_covariance_intersection",
    "track_to_track_fusion",
    "AsyncSensorBuffer",
    # Polars & pandas Integration
    "filter_dataframe",
    "filter_series",
    "FactorGraphExporter",
    # Metrics
    "rmse",
    "nees",
    "nis",
    "log_likelihood",
    # Numerical Jacobians (EKF helper)
    "numerical_jacobian",
    "numerical_transition_jacobian",
    "numerical_measurement_jacobian",
    # Gating
    "chi2_gate",
    "mahalanobis_distance",
    "ellipsoidal_gate",
    "gated_update",
    # Consistency Tests
    "nis_test",
    "nees_test",
    "innovation_whiteness_test",
    # Diagnostics & Detection
    "FilterDiagnostics",
    "Chi2OutlierDetector",
    "DetectionResult",
    # Plotting
    "plot_trajectory",
    "plot_covariance",
    "plot_innovations",
    # Auto-tuning & Learning
    "tune_kalman_filter",
    "quick_tune",
    "TuneResult",
    "OnlineEM",
    # Experiments
    "run_experiment",
    # Models
    "constant_velocity",
    "constant_acceleration",
    "constant_turn",
    "imu_velocity_control",
    "discrete_white_noise",
    "position_measurement_model",
    # Sensor-fusion cookbook: quaternion attitude EKF
    "quaternion_normalize",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "quaternion_angular_error",
    "attitude_transition",
    "attitude_transition_jacobian",
    "gravity_measurement",
    "gravity_measurement_jacobian",
    # scikit-learn integration (lazy — requires kalbee[sklearn])
    "KalmanEstimator",
    # Tracking
    "MultiObjectTracker",
    "Track",
    "iou_matrix",
    "mahalanobis_matrix",
    "associate",
    "JPDAAssociation",
    "PMBMTracker",
    "BernoulliTarget",
    # Learning
    "em_kalman",
    "EMResult",
]


def __getattr__(name):
    # Lazily wired so importing kalbee doesn't require scikit-learn.
    if name == "KalmanEstimator":
        from kalbee.modules.integration.sklearn_api import KalmanEstimator

        return KalmanEstimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
