"""Template matching of every tilt image with Leopard-EM, with parameters for tilt-series.

The search itself is done by Leopard-EM (Giammar et al., 2026; see
:data:`parapick.matching.leopard.CITATION`), which must be cited with the results.

Each tilt image is searched on its own (2DTM), with parameters derived from the acquisition
(:mod:`parapick.matching.schedule`): the low-pass follows the dose the tilt had received, the
angular and defocus steps the resolution left, and the defocus range the tilt of the sample.
The peaks found on all tilts are written as ParaPick picks, ready for ``localize``.

Output directory::

    micrographs/<TS>_<NNN>.mrc   tilt images as 2D MRC files (links when possible)
    tomograms.star               geometry pointing at them (use it with localize)
    schedule.csv                 parameters of every tilt and which ones were searched
    configs/<TS>_<NNN>.yaml      Leopard-EM match_template configuration of each tilt
    results/<TS>_<NNN>.h5        Leopard-EM results (MIP, z-scores, orientations, defocus)
    peaks/<TS>_<NNN>.csv         Leopard-EM peaks of each searched tilt
    picks.star                   all the peaks, as ParaPick picks (score: z-score)

Runs resume: tilts whose peaks were written are not searched again unless ``overwrite``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from parapick.formats import AUTO, FormatRegistry, TomogramsReadOptions, default_registry
from parapick.formats.relion5.picks import write_picks
from parapick.matching import leopard
from parapick.matching.plan import PlannedTilt, leopard_config, plan_matching, schedule_table
from parapick.matching.schedule import ScheduleConstants
from parapick.tomography import PickTable
from parapick.workflows.unstack import unstack

__all__ = ["MatchTemplateResult", "collect_picks", "run_match_template"]

logger = logging.getLogger(__name__)

SCHEDULE_FILE = "schedule.csv"
PICKS_FILE = "picks.star"
CONFIGS_DIRECTORY = "configs"
RESULTS_DIRECTORY = "results"
PEAKS_DIRECTORY = "peaks"
SCORE_COLUMN = "scaled_mip"
"""Leopard-EM's z-score of a peak, used as the score of the pick."""
PIXEL_SIZE_TOLERANCE = 0.01


@dataclass(frozen=True)
class MatchTemplateResult:
    """What :func:`run_match_template` produced."""

    plan: tuple[PlannedTilt, ...]
    picks: PickTable | None
    """Peaks of the searched tilts, ``None`` when only the schedule was written."""
    output_directory: Path


def _peaks_path(output: Path, tilt: PlannedTilt) -> Path:
    return output / PEAKS_DIRECTORY / f"{tilt.stem}.csv"


def collect_picks(plan: Sequence[PlannedTilt], output: Path) -> PickTable:
    """Gather the peaks of the searched tilts as ParaPick picks (pixels of the tilt images)."""
    names, tilt_ids, coordinates, scores = [], [], [], []
    for tilt in plan:
        path = _peaks_path(output, tilt)
        if not tilt.selected or not path.exists():
            continue
        peaks = pd.read_csv(path)
        names += [tilt.tilt_series] * len(peaks)
        tilt_ids += [tilt.tilt_id] * len(peaks)
        coordinates.append(peaks[["pos_x_img", "pos_y_img"]].to_numpy(dtype=np.float64))
        scores.append(peaks[SCORE_COLUMN].to_numpy(dtype=np.float64))
    return PickTable(
        tilt_ids=np.array(tilt_ids, dtype=object),
        coordinates=np.concatenate(coordinates) if coordinates else np.empty((0, 2)),
        tilt_series=np.array(names, dtype=object),
        scores=np.concatenate(scores) if scores else np.empty(0),
    )


def _check_template(template: Path, pixel_size: float) -> None:
    size = leopard.template_pixel_size(template)
    if size is None:
        logger.warning("%s has no voxel size; it must be %.4f Å/px", template, pixel_size)
    elif abs(size - pixel_size) > PIXEL_SIZE_TOLERANCE * pixel_size:
        raise ValueError(
            f"The template has {size:.4f} Å/px but the tilt images {pixel_size:.4f} Å/px: "
            "simulate or rescale the template at the pixel size of the tilt-series"
        )


def run_match_template(
    tomograms_path: Path,
    template: Path,
    output_directory: Path,
    *,
    particle_diameter: float,
    thickness: float | None = None,
    dose_per_tilt: float | None = None,
    energy_filter: bool = True,
    max_tilts: int | None = None,
    tilt_series: Sequence[str] | None = None,
    false_positives: float = 1.0,
    symmetry: str = "C1",
    device: str = "auto",
    orientation_batch_size: int = 16,
    schedule_only: bool = False,
    overwrite: bool = False,
    tomograms_format: str = AUTO,
    read_options: TomogramsReadOptions | None = None,
    constants: ScheduleConstants | None = None,
    registry: FormatRegistry | None = None,
) -> MatchTemplateResult:
    """Search every tilt image for ``template`` with Leopard-EM.

    Parameters
    ----------
    tomograms_path
        RELION 5 ``tomograms.star`` with the CTF and the pre-exposure of each tilt.
    template
        Template volume (MRC) at the pixel size of the tilt-series.
    output_directory
        Destination (see the module documentation).
    particle_diameter
        Diameter of the target (Å).
    thickness
        Thickness of the sample (Å). Default: the Z size of the tomogram.
    dose_per_tilt
        Dose of each tilt (e/Å²). Default: from the pre-exposure of the tilts.
    energy_filter
        Whether the images were zero-loss filtered (only changes the SNR ranking).
    max_tilts
        Search only this many tilts per tilt-series, those with the highest predicted SNR.
    tilt_series
        Only these tilt-series (default: all).
    false_positives
        Expected false positives per image when thresholding the peaks.
    symmetry
        Symmetry of the template (e.g. ``C1``, ``D7``), to restrict the search.
    device
        ``auto``, ``gpu`` (all visible GPUs) or ``cpu``.
    orientation_batch_size
        Orientations correlated at once (GPU memory).
    schedule_only
        Write the schedule and the configurations, but do not search.
    overwrite
        Search tilts again even if their peaks exist.
    """
    logger.info(
        "Template matching by Leopard-EM. If you use these results, cite: %s", leopard.CITATION
    )
    registry = registry or default_registry()
    reader = registry.tomograms_reader(tomograms_format, tomograms_path)
    tomograms = reader.read_tomograms(tomograms_path, read_options or TomogramsReadOptions())
    if tilt_series:
        tomograms = tomograms.subset(list(tilt_series))
    if not len(tomograms):
        raise ValueError(f"No tilt-series in {tomograms_path}")
    for ts in tomograms:
        _check_template(template, ts.pixel_size)

    # Leopard-EM reads 2D MRC files: link or extract the tilt images
    output = output_directory
    tomograms, _ = unstack(tomograms, output, anchor=tomograms_path, link=True)
    registry.tomograms_writer("relion5").write_tomograms(tomograms, output / "tomograms.star")

    plan = plan_matching(
        tomograms,
        particle_diameter=particle_diameter,
        thickness=thickness,
        dose_per_tilt=dose_per_tilt,
        energy_filter=energy_filter,
        max_tilts=max_tilts,
        constants=constants,
    )
    schedule_table(plan).to_csv(output / SCHEDULE_FILE, index=False)
    selected = [t for t in plan if t.selected]
    logger.info(
        "Schedule of %d tilts written to %s; %d to search",
        len(plan),
        output / SCHEDULE_FILE,
        len(selected),
    )

    gpu_ids = leopard.gpu_ids_for(device)
    (output / RESULTS_DIRECTORY).mkdir(parents=True, exist_ok=True)
    configs = {}
    for tilt in plan:
        config = leopard_config(
            tilt,
            template=template,
            result=output / RESULTS_DIRECTORY / f"{tilt.stem}.h5",
            gpu_ids=gpu_ids,
            symmetry=symmetry,
            constants=constants,
        )
        configs[tilt.stem] = output / CONFIGS_DIRECTORY / f"{tilt.stem}.yaml"
        leopard.write_config(config, configs[tilt.stem])
    if schedule_only:
        return MatchTemplateResult(tuple(plan), None, output)

    for n, tilt in enumerate(selected, start=1):
        peaks_path = _peaks_path(output, tilt)
        if peaks_path.exists() and not overwrite:
            logger.info("[%d/%d] %s: already searched", n, len(selected), tilt.stem)
            continue
        p = tilt.parameters
        logger.info(
            "[%d/%d] %s: %.0f° tilt, %.1f e/Å² before, %.1f Å, %.2f/%.2f° steps, "
            "%d defoci of %.0f Å, relative SNR %.3f",
            n,
            len(selected),
            tilt.stem,
            tilt.tilt.angle,
            tilt.tilt.pre_exposure,
            p.resolution,
            p.theta_step,
            p.psi_step,
            p.n_defocus,
            p.defocus_step,
            p.relative_snr,
        )
        peaks = leopard.run_match_template(
            configs[tilt.stem],
            orientation_batch_size=orientation_batch_size,
            false_positives=false_positives,
            mask_radius=0.5 * particle_diameter / tilt.pixel_size,
        )
        peaks_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = peaks_path.with_suffix(".csv.tmp")
        peaks.to_csv(temporary, index=False)
        temporary.replace(peaks_path)  # Written last: marks the tilt as searched
        logger.info("%s: %d peaks", tilt.stem, len(peaks))

    picks = collect_picks(plan, output)
    write_picks(picks, output / PICKS_FILE)
    logger.info(
        "Wrote %d picks to %s. Localize them with: parapick localize --tomograms %s "
        "--picks %s -o <dir>",
        len(picks),
        output / PICKS_FILE,
        output / "tomograms.star",
        output / PICKS_FILE,
    )
    return MatchTemplateResult(tuple(plan), picks, output)
