"""Ready-made motion and measurement models for building filters."""

from kalbee.models.motion import (
    constant_velocity,
    constant_acceleration,
    constant_turn,
    imu_velocity_control,
)
from kalbee.models.measurement import (
    discrete_white_noise,
    position_measurement_model,
)
from kalbee.models.attitude import (
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

__all__ = [
    "constant_velocity",
    "constant_acceleration",
    "constant_turn",
    "imu_velocity_control",
    "discrete_white_noise",
    "position_measurement_model",
    "quaternion_normalize",
    "quaternion_conjugate",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "quaternion_angular_error",
    "attitude_transition",
    "attitude_transition_jacobian",
    "gravity_measurement",
    "gravity_measurement_jacobian",
]
