"""What template matching needs to know about a tilt-series, inferred from its metadata.

Everything comes from the tilt-series itself, as read from RELION 5 ``tomograms.star``:

* the angle between the beam and the normal of the sample, pre-tilt included, and the
  direction of the defocus gradient on each image, from the projection geometry;
* the acquisition order and the dose of each tilt, from ``rlnMicrographPreExposure``
  (the step to the next tilt; the median step across the gaps left by excluded tilts);
* the CTF of each tilt (``rlnDefocusU``, ``rlnDefocusV``, ``rlnDefocusAngle``,
  ``rlnPhaseShift``) and the optics (``rlnVoltage``, ``rlnSphericalAberration``,
  ``rlnAmplitudeContrast``);
* the thickness of the sample, from the size of the tomogram (``rlnTomoSizeZ``).

Only the particle diameter has to be given. The thickness and the dose can be overridden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from parapick.formats.relion5 import labels as L
from parapick.matching.schedule import Acquisition, TiltImage
from parapick.tomography import TiltSeries

__all__ = ["Optics", "TiltCtf", "TiltSeriesAcquisition", "infer_acquisition"]

logger = logging.getLogger(__name__)

GAP_FACTOR = 1.5
"""Pre-exposure steps above this many times the median one span excluded tilts."""


@dataclass(frozen=True)
class TiltCtf:
    """CTF of a tilt image, as in RELION.

    Defoci in Å (underfocus positive), astigmatism angle and phase shift in degrees.
    """

    defocus_u: float
    defocus_v: float
    astigmatism_angle: float
    phase_shift: float = 0.0


@dataclass(frozen=True)
class Optics:
    """Optics shared by the tilts of a tilt-series."""

    voltage: float
    spherical_aberration: float
    amplitude_contrast: float


@dataclass(frozen=True)
class TiltSeriesAcquisition:
    """Inputs of the schedule for one tilt-series; per-tilt fields in tilt-series order.

    Attributes
    ----------
    acquisition
        Shared acquisition parameters.
    optics
        Shared optics.
    tilts
        Angle, dose and defocus gradient of each tilt.
    ctfs
        CTF of each tilt.
    acquisition_order
        0-based position of each tilt in the acquisition.
    """

    acquisition: Acquisition
    optics: Optics
    tilts: tuple[TiltImage, ...]
    ctfs: tuple[TiltCtf, ...]
    acquisition_order: tuple[int, ...]


def _metadata(ts: TiltSeries) -> tuple[dict[str, Any], pd.DataFrame]:
    row = ts.metadata.get("relion5.global")
    tilts = ts.metadata.get("relion5.tilts")
    if not isinstance(row, dict) or not isinstance(tilts, pd.DataFrame):
        raise ValueError(
            f"Tilt-series {ts.name!r} has no RELION 5 metadata: template matching needs the "
            "CTF and dose of each tilt, read from a RELION 5 tomograms.star"
        )
    if len(tilts) != ts.n_tilts:
        raise ValueError(f"Tilt-series {ts.name!r}: the metadata does not match the tilts")
    return row, tilts


def _number(row: dict[str, Any], label: str, default: float) -> float:
    value = row.get(label)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return float(value)


def _doses(
    ts: TiltSeries, tilts: pd.DataFrame, angles: NDArray[np.float64], dose_per_tilt: float | None
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Pre-exposure, dose and acquisition order of each tilt."""
    n = ts.n_tilts
    pre = (
        tilts[L.PRE_EXPOSURE].to_numpy(dtype=np.float64)
        if L.PRE_EXPOSURE in tilts.columns
        else None
    )
    if pre is not None and (np.isnan(pre).any() or np.ptp(pre) <= 0):
        pre = None  # Unset (e.g. all zero)
    if pre is None:
        if dose_per_tilt is None:
            raise ValueError(
                f"Tilt-series {ts.name!r} has no {L.PRE_EXPOSURE}: give the dose per tilt "
                "(--dose-per-tilt)"
            )
        # Dose-symmetric: from the least tilted image outwards
        nominal = (
            np.abs(tilts[L.NOMINAL_TILT].to_numpy(dtype=np.float64))
            if L.NOMINAL_TILT in tilts.columns
            else angles
        )
        order = np.argsort(nominal, kind="stable")
        logger.warning(
            "%s: no %s, assuming a dose-symmetric acquisition from the least tilted image",
            ts.name,
            L.PRE_EXPOSURE,
        )
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(n)
        return ranks.astype(np.float64) * dose_per_tilt, np.full(n, float(dose_per_tilt)), ranks

    order = np.argsort(pre, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n)
    if dose_per_tilt is not None:
        return pre, np.full(n, dose_per_tilt), ranks
    increments = np.diff(pre[order])
    if (increments <= 0).any():
        raise ValueError(
            f"Tilt-series {ts.name!r}: several tilts have the same {L.PRE_EXPOSURE}, so their "
            "dose is unknown. Give the dose per tilt (--dose-per-tilt)"
        )
    # Each tilt received the dose up to the next one. Steps much larger than usual span
    # excluded tilts, and the last tilt has no next one: those get the median dose.
    typical = float(np.median(increments))
    increments = np.where(increments > GAP_FACTOR * typical, typical, increments)
    doses = np.empty(n)
    doses[order] = np.append(increments, typical)
    return pre, doses, ranks


def infer_acquisition(
    ts: TiltSeries,
    *,
    particle_diameter: float,
    thickness: float | None = None,
    dose_per_tilt: float | None = None,
    energy_filter: bool = True,
) -> TiltSeriesAcquisition:
    """Schedule inputs of a tilt-series read from RELION 5 metadata.

    Parameters
    ----------
    ts
        Tilt-series read from a RELION 5 ``tomograms.star``.
    particle_diameter
        Diameter of the target (Å).
    thickness
        Thickness of the sample (Å). Default: the Z size of the tomogram.
    dose_per_tilt
        Dose of each tilt (e/Å²). Default: the increments of ``rlnMicrographPreExposure``.
    energy_filter
        Whether the images were zero-loss filtered.
    """
    row, tilts = _metadata(ts)
    if ts.image_size is None:
        raise ValueError(f"Tilt-series {ts.name!r} has no image size")
    if thickness is None:
        if ts.volume_size is None:
            raise ValueError(
                f"Tilt-series {ts.name!r} has no tomogram size: give the sample thickness "
                "(--thickness)"
            )
        thickness = ts.volume_size[2]
    missing = [label for label in (L.DEFOCUS_U, L.DEFOCUS_V, L.DEFOCUS_ANGLE) if label not in tilts]
    if missing:
        raise ValueError(
            f"Tilt-series {ts.name!r} has no CTF ({', '.join(missing)} missing): estimate the "
            "CTF of the tilt images first"
        )

    angles = ts.tilt_angles  # beam vs normal of the sample, pre-tilt included
    gradients = ts.projections[:, :, 2]  # the normal of the sample, projected on the image
    pre, doses, ranks = _doses(ts, tilts, angles, dose_per_tilt)
    phase = (
        tilts[L.PHASE_SHIFT].to_numpy(dtype=np.float64)
        if L.PHASE_SHIFT in tilts.columns
        else np.zeros(ts.n_tilts)
    )
    ctfs = tuple(
        TiltCtf(float(u), float(v), float(angle), float(shift))
        for u, v, angle, shift in zip(
            tilts[L.DEFOCUS_U].to_numpy(dtype=np.float64),
            tilts[L.DEFOCUS_V].to_numpy(dtype=np.float64),
            tilts[L.DEFOCUS_ANGLE].to_numpy(dtype=np.float64),
            np.nan_to_num(phase),
            strict=True,
        )
    )
    optics = Optics(
        voltage=_number(row, L.VOLTAGE, 300.0),
        spherical_aberration=_number(row, L.SPHERICAL_ABERRATION, 2.7),
        amplitude_contrast=_number(row, L.AMPLITUDE_CONTRAST, 0.1),
    )
    return TiltSeriesAcquisition(
        acquisition=Acquisition(
            pixel_size=ts.pixel_size,
            image_size=ts.image_size,
            voltage=optics.voltage,
            thickness=float(thickness),
            particle_diameter=particle_diameter,
            energy_filter=energy_filter,
        ),
        optics=optics,
        tilts=tuple(
            TiltImage(
                angle=float(angle),
                pre_exposure=float(p),
                dose=float(d),
                gradient=(float(g[0]), float(g[1])),
            )
            for angle, p, d, g in zip(angles, pre, doses, gradients, strict=True)
        ),
        ctfs=ctfs,
        acquisition_order=tuple(int(r) for r in ranks),
    )
