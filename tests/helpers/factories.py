"""Factories for test geometry."""

from __future__ import annotations

import numpy as np

from parapick.tomography import TiltSeries, xyz_euler_to_matrix


def make_tilt_series(
    name: str = "TS_01",
    angles: np.ndarray | None = None,
    *,
    pixel_size: float = 10.0,
    image_size: tuple[int, int] = (512, 512),
    volume_size: tuple[float, float, float] | None = (5120.0, 5120.0, 1000.0),
    tilt_axis: float = 85.0,
) -> TiltSeries:
    angles = np.arange(-60, 61, 3.0) if angles is None else angles
    projections = xyz_euler_to_matrix(0.0, angles, tilt_axis)[:, :2]
    translations = np.tile(np.floor(np.array(image_size) / 2) * pixel_size, (len(angles), 1))
    return TiltSeries(
        name=name,
        tilt_ids=tuple(f"tilts/{name}_{i:03d}.mrc" for i in range(len(angles))),
        projections=projections,
        translations=translations,
        pixel_size=pixel_size,
        image_size=image_size,
        volume_size=volume_size,
    )
