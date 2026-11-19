"""Per-tilt 2D template matching parameters for the images of a tilt-series.

2D template matching (2DTM) programs such as Leopard-EM are tuned for single, untilted,
high-dose micrographs (30 to 50 e/Å²). A tilt image differs in four ways, and each one sets
a group of parameters:

=============================  ==========================================  ================
Difference                     Consequence                                 Parameters
=============================  ==========================================  ================
Low dose per image             SNR falls as √dose                          tilt ranking
Accumulated pre-exposure       high-resolution signal already destroyed    low-pass cutoff
Lower usable resolution        coarser sampling suffices                   angular, defocus
                                                                           steps
Tilted specimen plane          defocus varies across the field, longer     defocus range,
                               path through the sample                     transmission
=============================  ==========================================  ================

The formulas come from matched-filter theory, the radiation-damage model of Grant &
Grigorieff (2015) and the tilt geometry. They are starting points, not validated settings.
The policy constants live in :class:`ScheduleConstants`; the per-dataset inputs are physical
(:class:`Acquisition`, :class:`TiltImage`).

Units: frequencies ``k`` in 1/Å, doses in e/Å², lengths in Å, angles in degrees. Filter
cutoffs for Leopard-EM are in cycles per pixel (``k * pixel_size``, Nyquist = 0.5).

This module only needs numpy and scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import trapezoid
from scipy.special import erfcinv

__all__ = [
    "Acquisition",
    "ScheduleConstants",
    "TiltImage",
    "TiltParameters",
    "critical_exposure",
    "schedule_tilts",
    "wavelength",
]

FULL_SPHERE_DEG2 = 41252.96
"""Solid angle of the sphere in square degrees, to count the orientations of a search."""


@dataclass(frozen=True)
class ScheduleConstants:
    """Policy constants of the schedule. The defaults are the ones of the specification.

    Attributes
    ----------
    damage_a, damage_b, damage_c
        Critical exposure ``Nc(k) = v (a k^b + c)`` at 300 kV (Grant & Grigorieff, 2015).
    kappa
        The low-pass cuts where the dose reaches ``kappa`` critical exposures, which is where
        the surviving signal falls to about half its average over the band. Larger keeps more
        high frequencies.
    coarsest_lowpass
        Never low-pass to a resolution coarser than this (Å).
    highpass_factor
        The high-pass removes wavelengths longer than this many particle diameters
        (directional thickness ramps of tilted images). 0 disables it.
    falloff
        Soft edge of the band-pass, cycles/px.
    whitening_max_freq
        Highest frequency of the whitening filter, cycles/px.
    theta_reference, psi_reference, resolution_reference, diameter_reference
        Leopard-EM's default angular steps (deg) and the resolution (Å) and particle
        diameter (Å) they apply to. Steps scale with ``resolution / diameter``.
    theta_max, psi_max
        Coarsest angular steps (deg): coarse sampling costs SNR, which tilts lack.
    phase_tolerance
        CTF phase error (rad) allowed at the low-pass edge half a defocus step away.
    defocus_step_min, defocus_step_max
        Bounds of the defocus step (Å).
    defocus_margin
        Allowance for the error of the CTF fit (Å).
    defocus_half_range_min
        Smallest half range of the defocus search (Å), Leopard-EM's default.
    b_factor
        Envelope ``exp(-B k² / 2)`` on the template power (Å²), for the SNR ranking.
    inelastic_mean_free_path
        Inelastic mean free path at 300 kV (Å), for the transmission of the sample.
    reference_dose
        Dose (e/Å²) of the conventional untilted 2DTM image the SNR is relative to.
    """

    damage_a: float = 0.245
    damage_b: float = -1.665
    damage_c: float = 2.81
    kappa: float = 2.5
    coarsest_lowpass: float = 30.0
    highpass_factor: float = 1.6
    falloff: float = 0.02
    whitening_max_freq: float = 0.5
    theta_reference: float = 2.5
    psi_reference: float = 1.5
    resolution_reference: float = 3.4
    diameter_reference: float = 250.0
    theta_max: float = 4.0
    psi_max: float = 3.0
    phase_tolerance: float = math.pi / 4
    defocus_step_min: float = 200.0
    defocus_step_max: float = 500.0
    defocus_margin: float = 300.0
    defocus_half_range_min: float = 1200.0
    b_factor: float = 60.0
    inelastic_mean_free_path: float = 3200.0
    reference_dose: float = 30.0


@dataclass(frozen=True)
class Acquisition:
    """What the schedule needs to know about a tilt-series and the target.

    Attributes
    ----------
    pixel_size
        Å/px of the tilt images.
    image_size
        ``(width, height)`` of the tilt images in pixels.
    voltage
        Acceleration voltage in kV.
    thickness
        Thickness of the sample along its normal (Å).
    particle_diameter
        Diameter of the target (Å).
    energy_filter
        Whether the images were zero-loss filtered.
    """

    pixel_size: float
    image_size: tuple[int, int]
    voltage: float
    thickness: float
    particle_diameter: float
    energy_filter: bool = True


@dataclass(frozen=True)
class TiltImage:
    """One tilt image, in any order.

    Attributes
    ----------
    angle
        Angle between the beam and the normal of the sample (deg), pre-tilt included.
    pre_exposure
        Dose received before this image (e/Å²).
    dose
        Dose of this image (e/Å²).
    gradient
        Direction of the defocus gradient on the image (any length): the projection of the
        normal of the sample. Perpendicular to the tilt axis. Ignored when untilted.
    """

    angle: float
    pre_exposure: float
    dose: float
    gradient: tuple[float, float] = (1.0, 0.0)


@dataclass(frozen=True)
class TiltParameters:
    """Template matching parameters of one tilt image.

    Attributes
    ----------
    resolution
        Effective resolution (Å): the low-pass cutoff, or Nyquist without low-pass.
    lowpass
        Whether the band-pass has a low-pass edge.
    low_freq_cutoff, high_freq_cutoff
        Band-pass edges in cycles/px (high-pass and low-pass), ``None`` if absent.
    theta_step, psi_step
        Angular steps (deg).
    defocus_step, defocus_half_range
        Defocus search around the fitted defocus (Å).
    n_defocus, n_orientations
        Size of the search.
    z_threshold
        Expected detection threshold (Gaussian noise, one false positive per image), only
        to plan; Leopard-EM computes its own.
    relative_snr
        Predicted SNR relative to a conventional untilted image of the same sample. A
        ranking, not a measurement.
    """

    resolution: float
    lowpass: bool
    low_freq_cutoff: float | None
    high_freq_cutoff: float | None
    theta_step: float
    psi_step: float
    defocus_step: float
    defocus_half_range: float
    n_defocus: int
    n_orientations: float
    z_threshold: float
    relative_snr: float

    def leopard(self, *, falloff: float, whitening_max_freq: float) -> dict[str, Any]:
        """Leopard-EM ``match_template`` configuration fragment of this tilt."""
        bandpass = self.low_freq_cutoff is not None or self.high_freq_cutoff is not None
        return {
            "defocus_search_config": {
                "enabled": True,
                "defocus_min": -self.defocus_half_range,
                "defocus_max": self.defocus_half_range,
                "defocus_step": self.defocus_step,
            },
            "orientation_search_config": {
                "base_grid_method": "uniform",
                "theta_step": self.theta_step,
                "psi_step": self.psi_step,
            },
            "preprocessing_filters": {
                "whitening_filter": {
                    "enabled": True,
                    "do_power_spectrum": True,
                    "max_freq": whitening_max_freq,
                    "num_freq_bins": None,
                },
                "bandpass_filter": {
                    "enabled": bandpass,
                    "low_freq_cutoff": self.low_freq_cutoff,
                    "high_freq_cutoff": self.high_freq_cutoff,
                    "falloff": falloff if bandpass else None,
                },
            },
        }


def wavelength(voltage: float) -> float:
    """Relativistic electron wavelength (Å) at ``voltage`` kV."""
    volts = voltage * 1e3
    return 12.2643 / math.sqrt(volts + 0.97845e-6 * volts * volts)


def _voltage_scale(voltage: float) -> float:
    """Critical exposure relative to 300 kV (as in cisTEM and unblur)."""
    return float(np.interp(voltage, [100.0, 200.0, 300.0], [0.57, 0.8, 1.0]))


def critical_exposure(
    k: NDArray[np.float64] | float, voltage: float, constants: ScheduleConstants
) -> NDArray[np.float64]:
    """Dose (e/Å²) at which the signal at frequency ``k`` (1/Å) decays to 1/e."""
    c = constants
    k = np.asarray(k, dtype=np.float64)
    return _voltage_scale(voltage) * (c.damage_a * np.power(k, c.damage_b) + c.damage_c)


def _mean_attenuation(
    k: NDArray[np.float64], start: float, end: float, voltage: float, c: ScheduleConstants
) -> NDArray[np.float64]:
    """Signal amplitude at ``k`` averaged over an exposure from ``start`` to ``end``."""
    twice = 2.0 * critical_exposure(k, voltage, c)
    mean: NDArray[np.float64] = (
        twice / (end - start) * (np.exp(-start / twice) - np.exp(-end / twice))
    )
    return mean


def _band_quality(
    k_low: float, k_high: float, start: float, end: float, voltage: float, c: ScheduleConstants
) -> float:
    """``∫ q s k dk / √(∫ s k dk)`` over the band: the SNR of a step filter over √dose."""
    k = np.linspace(k_low, k_high, 2000)
    power = np.exp(-c.b_factor * k * k / 2.0)
    q = _mean_attenuation(k, start, end, voltage, c)
    return float(trapezoid(q * power * k, k) / math.sqrt(trapezoid(power * k, k)))


def _lowpass_frequency(dose: float, voltage: float, c: ScheduleConstants) -> float:
    """Frequency (1/Å) where ``dose = kappa Nc(k)``; infinite if never reached."""
    x = dose / (c.kappa * _voltage_scale(voltage)) - c.damage_c
    return math.inf if x <= 0 else float((x / c.damage_a) ** (1.0 / c.damage_b))


def _z_threshold(n_tests: float, false_positives: float) -> float:
    return float(math.sqrt(2.0) * erfcinv(2.0 * false_positives / n_tests))


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _transmission(angle: float, acquisition: Acquisition, c: ScheduleConstants) -> float:
    """Fraction of the signal left after the inelastic scattering in the sample."""
    path = acquisition.thickness / math.cos(math.radians(angle))
    factor = 2.0 if acquisition.energy_filter else 1.0
    return math.exp(-path / (factor * c.inelastic_mean_free_path))


def schedule_tilts(
    acquisition: Acquisition,
    tilts: Sequence[TiltImage],
    constants: ScheduleConstants | None = None,
) -> list[TiltParameters]:
    """Template matching parameters of each tilt image (in the order given)."""
    c = constants or ScheduleConstants()
    a = acquisition
    if not a.pixel_size > 0 or not a.particle_diameter > 0 or a.thickness < 0:
        raise ValueError("pixel size and particle diameter must be positive, thickness >= 0")
    lam = wavelength(a.voltage)
    k_nyquist = 0.5 / a.pixel_size
    k_highpass = 1.0 / (c.highpass_factor * a.particle_diameter) if c.highpass_factor else 0.0
    k_floor = 1.0 / c.coarsest_lowpass
    k_band_low = max(k_highpass, 1e-4)
    width, height = (n * a.pixel_size for n in a.image_size)
    reference = (
        math.sqrt(c.reference_dose)
        * _transmission(0.0, a, c)
        * _band_quality(k_band_low, k_nyquist, 0.0, c.reference_dose, a.voltage, c)
    )

    parameters = []
    for tilt in tilts:
        if not tilt.dose > 0:
            raise ValueError("Every tilt image needs a positive dose")
        start, end = tilt.pre_exposure, tilt.pre_exposure + tilt.dose
        alpha = math.radians(tilt.angle)

        # Low-pass where the damage removes more signal than noise (dose at mid-exposure)
        k_cut = _clamp(_lowpass_frequency(0.5 * (start + end), a.voltage, c), k_floor, k_nyquist)
        lowpass = k_cut < 0.999 * k_nyquist
        resolution = 1.0 / k_cut

        # Angular steps scale with resolution / diameter, never finer than the defaults
        scale = (resolution / c.resolution_reference) * (c.diameter_reference / a.particle_diameter)
        theta = round(_clamp(c.theta_reference * scale, c.theta_reference, c.theta_max), 2)
        psi = round(_clamp(c.psi_reference * scale, c.psi_reference, c.psi_max), 2)

        # Defocus step from the depth of field at the cutoff
        step = 2.0 * c.phase_tolerance * resolution**2 / (math.pi * lam)
        step = _clamp(step, c.defocus_step_min, c.defocus_step_max)
        # Defocus range: the tilted plane across the field plus the slab along the beam
        gx, gy = tilt.gradient
        norm = math.hypot(gx, gy)
        extent = 0.5 * (width * abs(gx) + height * abs(gy)) / norm if norm > 0 else 0.0
        half = (
            extent * abs(math.tan(alpha)) + a.thickness / (2.0 * math.cos(alpha)) + c.defocus_margin
        )
        half = math.ceil(max(half, c.defocus_half_range_min) / step) * step
        n_defocus = round(2.0 * half / step) + 1

        n_orientations = (FULL_SPHERE_DEG2 / theta**2) * (360.0 / psi)
        n_tests = a.image_size[0] * a.image_size[1] * n_orientations * n_defocus
        score = (
            math.sqrt(tilt.dose)
            * _transmission(tilt.angle, a, c)
            * _band_quality(k_band_low, k_cut, start, end, a.voltage, c)
        )
        parameters.append(
            TiltParameters(
                resolution=resolution,
                lowpass=lowpass,
                low_freq_cutoff=round(k_highpass * a.pixel_size, 5) if k_highpass > 0 else None,
                high_freq_cutoff=round(k_cut * a.pixel_size, 4) if lowpass else None,
                theta_step=theta,
                psi_step=psi,
                defocus_step=step,
                defocus_half_range=half,
                n_defocus=n_defocus,
                n_orientations=n_orientations,
                z_threshold=_z_threshold(n_tests, 1.0),
                relative_snr=score / reference,
            )
        )
    return parameters
