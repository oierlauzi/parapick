"""RELION 5 projection convention <-> ParaPick's neutral affine model.

RELION 5 (``Tomogram::setProjectionMatrix``) maps a specimen point to a tilt image as::

    y_px = c_img + shift_Å / p + [Rz(ZRot) Ry(YTilt) Rx(XTilt) (X_px - c_tomo)]_xy

with ``c_img = floor(image_size / 2)`` and ``c_tomo = floor(tomo_size / 2)`` (in pixels). In
contrast, particle coordinates (``rlnCenteredCoordinate*Angst``) are relative to
``tomo_size / 2``. So, with ``X_c`` in centred Å and the offset
``o = (tomo_size / 2 - floor(tomo_size / 2)) p``::

    y_Å = P X_c + t,   P = [Rz Ry Rx]_xy,   t = c_img p + shift_Å + P o
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from parapick.tomography.rotations import (
    complete_rotation,
    matrix_to_xyz_euler,
    xyz_euler_to_matrix,
)

__all__ = ["centre_offset", "from_relion", "image_centre", "to_relion"]


def image_centre(image_size: tuple[int, int]) -> NDArray[np.float64]:
    """Pixel index of the projection centre on a tilt image: ``floor(size / 2)``."""
    return np.floor(np.asarray(image_size, dtype=np.float64) / 2.0)


def centre_offset(tomo_size_px: ArrayLike | None, pixel_size: float) -> NDArray[np.float64]:
    """Offset (Å) from the centred-coordinate origin to RELION's rotation centre."""
    if tomo_size_px is None:
        return np.zeros(3)
    size = np.asarray(tomo_size_px, dtype=np.float64)
    return (size / 2.0 - np.floor(size / 2.0)) * pixel_size


def from_relion(
    x_tilt: ArrayLike,
    y_tilt: ArrayLike,
    z_rot: ArrayLike,
    x_shift_angst: ArrayLike,
    y_shift_angst: ArrayLike,
    *,
    pixel_size: float,
    image_size: tuple[int, int],
    tomo_size_px: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert RELION 5 per-tilt alignment to ``(P, t)`` arrays of shape (N,2,3) and (N,2)."""
    rotations = xyz_euler_to_matrix(x_tilt, y_tilt, z_rot).reshape(-1, 3, 3)
    projections = rotations[:, :2, :]
    shifts = np.stack([np.atleast_1d(x_shift_angst), np.atleast_1d(y_shift_angst)], axis=-1).astype(
        np.float64
    )
    offset = centre_offset(tomo_size_px, pixel_size)
    translations = image_centre(image_size) * pixel_size + shifts + projections @ offset
    return projections, translations


def to_relion(
    projections: ArrayLike,
    translations: ArrayLike,
    *,
    pixel_size: float,
    image_size: tuple[int, int],
    tomo_size_px: ArrayLike | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Inverse of :func:`from_relion`. Projections must have orthonormal rows."""
    p = np.asarray(projections, dtype=np.float64).reshape(-1, 2, 3)
    t = np.asarray(translations, dtype=np.float64).reshape(-1, 2)
    if not np.allclose(p @ p.transpose(0, 2, 1), np.eye(2), atol=1e-6):
        raise ValueError("RELION 5 requires rigid (orthonormal) projections")
    x_tilt, y_tilt, z_rot = matrix_to_xyz_euler(complete_rotation(p))
    offset = centre_offset(tomo_size_px, pixel_size)
    shifts = t - image_centre(image_size) * pixel_size - p @ offset
    return {
        "x_tilt": x_tilt,
        "y_tilt": y_tilt,
        "z_rot": z_rot,
        "x_shift_angst": shifts[:, 0],
        "y_shift_angst": shifts[:, 1],
    }
