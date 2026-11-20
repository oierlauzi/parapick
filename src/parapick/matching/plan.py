"""Plan the template matching of a set of tilt-series: one search per tilt image.

The plan does not need Leopard-EM. It holds the schedule of every tilt, which tilts are
searched, and the Leopard-EM ``match_template`` configuration of each one.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from parapick.matching.acquisition import Optics, TiltCtf, infer_acquisition
from parapick.matching.schedule import (
    ScheduleConstants,
    TiltImage,
    TiltParameters,
    schedule_tilts,
)
from parapick.tomography import TomogramSet

__all__ = ["PlannedTilt", "leopard_config", "plan_matching", "schedule_table"]


@dataclass(frozen=True)
class PlannedTilt:
    """Template matching of one tilt image.

    Attributes
    ----------
    tilt_series
        Name of the tilt-series.
    index
        0-based index of the tilt in its tilt-series.
    tilt_id
        Identifier of the tilt (picks refer to it).
    image
        2D MRC file of the tilt image.
    acquisition_order
        0-based position of the tilt in the acquisition.
    pixel_size
        Å/px of the searched image (binned).
    binning
        Binning of the searched image relative to the tilt image.
    optics, ctf, tilt
        What the parameters were derived from.
    parameters
        Template matching parameters.
    selected
        Whether this tilt is searched.
    """

    tilt_series: str
    index: int
    tilt_id: str
    image: str
    acquisition_order: int
    pixel_size: float
    optics: Optics
    ctf: TiltCtf
    tilt: TiltImage
    parameters: TiltParameters
    selected: bool
    binning: int = 1

    @property
    def peak_separation(self) -> float:
        """Smallest distance between two peaks (pixels of the searched image).

        One resolution element of the tilt: the width of a correlation peak after its
        low-pass. Projections of different particles can be much closer than the particle
        size (e.g. at different depths), so the size of the particle is not used.
        """
        return self.parameters.resolution / self.pixel_size

    @property
    def stem(self) -> str:
        """Unique file stem of the tilt: ``<TS>_<NNN>`` (1-based index)."""
        return f"{self.tilt_series}_{self.index + 1:03d}"


def plan_matching(
    tomograms: TomogramSet,
    *,
    particle_diameter: float,
    thickness: float | None = None,
    dose_per_tilt: float | None = None,
    energy_filter: bool = True,
    max_tilts: int | None = None,
    binning: int = 1,
    constants: ScheduleConstants | None = None,
) -> list[PlannedTilt]:
    """Schedule every tilt and select the ones to search.

    With ``max_tilts``, only the tilts of each tilt-series with the highest predicted SNR
    are selected. With ``binning``, the tilts are searched binned by that factor (Fourier
    cropping): the parameters are those of the binned images. The plan is sorted by
    decreasing predicted SNR, the order to search in.
    """
    if max_tilts is not None and max_tilts < 1:
        raise ValueError("max_tilts must be >= 1")
    if binning < 1:
        raise ValueError("binning must be >= 1")
    planned = []
    for ts in tomograms:
        inputs = infer_acquisition(
            ts,
            particle_diameter=particle_diameter,
            thickness=thickness,
            dose_per_tilt=dose_per_tilt,
            energy_filter=energy_filter,
        )
        width, height = inputs.acquisition.image_size
        acquisition = dataclasses.replace(
            inputs.acquisition,
            pixel_size=ts.pixel_size * binning,
            image_size=(width // binning, height // binning),
        )
        parameters = schedule_tilts(acquisition, inputs.tilts, constants)
        ranking = sorted(range(ts.n_tilts), key=lambda i: -parameters[i].relative_snr)
        selected = set(ranking if max_tilts is None else ranking[:max_tilts])
        planned += [
            PlannedTilt(
                tilt_series=ts.name,
                index=i,
                tilt_id=ts.tilt_ids[i],
                image=ts.images[i],
                acquisition_order=inputs.acquisition_order[i],
                pixel_size=acquisition.pixel_size,
                optics=inputs.optics,
                ctf=inputs.ctfs[i],
                tilt=inputs.tilts[i],
                parameters=parameters[i],
                selected=i in selected,
                binning=binning,
            )
            for i in range(ts.n_tilts)
        ]
    return sorted(planned, key=lambda t: -t.parameters.relative_snr)


def leopard_config(
    tilt: PlannedTilt,
    *,
    template: Path,
    result: Path,
    gpu_ids: str | int | list[int] = "all",
    symmetry: str = "C1",
    constants: ScheduleConstants | None = None,
) -> dict[str, Any]:
    """Leopard-EM ``match_template`` configuration (``MatchTemplateManager`` fields)."""
    c = constants or ScheduleConstants()
    config: dict[str, Any] = {
        "micrograph_path": str(Path(tilt.image).resolve()),
        "template_volume_path": str(template.resolve()),
        "optics_group": {
            "label": tilt.stem,
            "pixel_size": tilt.pixel_size,
            "voltage": tilt.optics.voltage,
            "spherical_aberration": tilt.optics.spherical_aberration,
            "amplitude_contrast_ratio": tilt.optics.amplitude_contrast,
            "phase_shift": tilt.ctf.phase_shift,
            "defocus_u": tilt.ctf.defocus_u,
            "defocus_v": tilt.ctf.defocus_v,
            "astigmatism_angle": tilt.ctf.astigmatism_angle,
        },
        **tilt.parameters.leopard(falloff=c.falloff, whitening_max_freq=c.whitening_max_freq),
        "match_template_result": {"hdf5_path": str(result.resolve()), "allow_file_overwrite": True},
        # Batched: the GPU stays busy on small (e.g. binned) images, where the streamed backend
        # spends most of the time launching kernels
        "computational_config": {"gpu_ids": gpu_ids, "backend": "batched"},
    }
    config["orientation_search_config"]["symmetry"] = symmetry
    return config


def schedule_table(plan: Sequence[PlannedTilt]) -> pd.DataFrame:
    """One row per tilt, in acquisition order within each tilt-series."""
    rows = [
        {
            "tilt_series": t.tilt_series,
            "tilt_index": t.index + 1,
            "acquisition_order": t.acquisition_order,
            "micrograph": t.tilt_id,
            "angle": round(t.tilt.angle, 2),
            "pre_exposure": round(t.tilt.pre_exposure, 3),
            "dose": round(t.tilt.dose, 3),
            "defocus_u": t.ctf.defocus_u,
            "defocus_v": t.ctf.defocus_v,
            "resolution": round(t.parameters.resolution, 2),
            "low_freq_cutoff": t.parameters.low_freq_cutoff,
            "high_freq_cutoff": t.parameters.high_freq_cutoff,
            "theta_step": t.parameters.theta_step,
            "psi_step": t.parameters.psi_step,
            "defocus_step": round(t.parameters.defocus_step, 1),
            "defocus_half_range": round(t.parameters.defocus_half_range, 1),
            "n_defocus": t.parameters.n_defocus,
            "n_orientations": round(t.parameters.n_orientations),
            "z_threshold": round(t.parameters.z_threshold, 3),
            "relative_snr": round(t.parameters.relative_snr, 4),
            "selected": t.selected,
        }
        for t in plan
    ]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["tilt_series", "acquisition_order"], ignore_index=True)
