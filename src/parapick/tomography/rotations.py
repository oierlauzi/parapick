"""Elementary rotations and Euler-angle conversions.

All matrices are *active* right-handed rotations acting on column vectors, with angles in
degrees. They follow the conventions used by RELION 5 (``Rx``, ``Ry``, ``Rz`` in
``tomography_python_programs/_utils/transformations.py``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "complete_rotation",
    "matrix_to_xyz_euler",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "xyz_euler_to_matrix",
]


def _cos_sin(angles_deg: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    radians = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    return np.cos(radians), np.sin(radians)


def rotation_x(angles_deg: ArrayLike) -> NDArray[np.float64]:
    """Rotation(s) about the X axis. Output shape is ``(*angles.shape, 3, 3)``."""
    c, s = _cos_sin(angles_deg)
    out = np.zeros((*c.shape, 3, 3))
    out[..., 0, 0] = 1.0
    out[..., 1, 1] = c
    out[..., 1, 2] = -s
    out[..., 2, 1] = s
    out[..., 2, 2] = c
    return out


def rotation_y(angles_deg: ArrayLike) -> NDArray[np.float64]:
    """Rotation(s) about the Y axis. Output shape is ``(*angles.shape, 3, 3)``."""
    c, s = _cos_sin(angles_deg)
    out = np.zeros((*c.shape, 3, 3))
    out[..., 0, 0] = c
    out[..., 0, 2] = s
    out[..., 1, 1] = 1.0
    out[..., 2, 0] = -s
    out[..., 2, 2] = c
    return out


def rotation_z(angles_deg: ArrayLike) -> NDArray[np.float64]:
    """Rotation(s) about the Z axis. Output shape is ``(*angles.shape, 3, 3)``."""
    c, s = _cos_sin(angles_deg)
    out = np.zeros((*c.shape, 3, 3))
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    out[..., 2, 2] = 1.0
    return out


def xyz_euler_to_matrix(
    x_tilt: ArrayLike, y_tilt: ArrayLike, z_rot: ArrayLike
) -> NDArray[np.float64]:
    """Compose ``Rz(z_rot) @ Ry(y_tilt) @ Rx(x_tilt)`` (X applied first).

    This is the specimen rotation of RELION 5 (``rlnTomoXTilt``, ``rlnTomoYTilt``,
    ``rlnTomoZRot``).
    """
    return rotation_z(z_rot) @ rotation_y(y_tilt) @ rotation_x(x_tilt)


def matrix_to_xyz_euler(
    matrix: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Inverse of :func:`xyz_euler_to_matrix`.

    Returns ``(x_tilt, y_tilt, z_rot)`` in degrees. In gimbal lock (``|y_tilt| = 90°``) the
    X angle is set to zero.
    """
    r = np.asarray(matrix, dtype=np.float64)
    sin_y = np.clip(-r[..., 2, 0], -1.0, 1.0)
    y = np.arcsin(sin_y)
    locked = np.abs(sin_y) > 1.0 - 1e-12
    x = np.where(locked, 0.0, np.arctan2(r[..., 2, 1], r[..., 2, 2]))
    z = np.where(
        locked,
        np.arctan2(-r[..., 0, 1], r[..., 1, 1]),
        np.arctan2(r[..., 1, 0], r[..., 0, 0]),
    )
    return np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)


def complete_rotation(projection: ArrayLike) -> NDArray[np.float64]:
    """Complete orthonormal 2x3 projection(s) to 3x3 rotation(s) by adding ``r1 x r2``."""
    p = np.asarray(projection, dtype=np.float64)
    third = np.cross(p[..., 0, :], p[..., 1, :])
    return np.concatenate([p, third[..., None, :]], axis=-2)
