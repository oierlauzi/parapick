"""Template matching of every tilt image with Leopard-EM, with parameters for tilt-series.

The search itself is done by Leopard-EM (Giammar et al., 2026; see
:data:`parapick.matching.leopard.CITATION`), which must be cited with the results.

Each tilt image is searched on its own (2DTM), with parameters derived from the acquisition
(:mod:`parapick.matching.schedule`): the low-pass follows the dose the tilt had received, the
angular and defocus steps the resolution left, and the defocus range the tilt of the sample.
The peaks found on all tilts are written as ParaPick picks, ready for ``localize``.

Output directory::

    micrographs/<TS>_<NNN>.mrc   tilt images as 2D MRC files (links when possible)
    binned/<TS>_<NNN>.mrc        the tilt images binned, if searched binned
    template_bin<N>.mrc          the template binned, if given at the native pixel size
    tomograms.star               geometry pointing at them (use it with localize)
    settings.json                binning and sampling of the run (a resumed run must match)
    schedule.csv                 parameters of every tilt and which ones were searched
    configs/<TS>_<NNN>.yaml      Leopard-EM match_template configuration of each tilt
    results/<TS>_<NNN>.h5        Leopard-EM results (MIP, z-scores, orientations, defocus)
    peaks/<TS>_<NNN>.csv         Leopard-EM peaks of each searched tilt
    picks.star                   all the peaks, as ParaPick picks in native pixels
                                 (score: z-score)

Runs resume: tilts whose peaks were written are not searched again unless ``overwrite``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from parapick.formats import AUTO, FormatRegistry, TomogramsReadOptions, default_registry
from parapick.formats.relion5.picks import write_picks
from parapick.matching import leopard
from parapick.matching.peaks import PickingSettings, find_peaks, gaussian_threshold, score_map
from parapick.matching.plan import PlannedTilt, leopard_config, plan_matching, schedule_table
from parapick.matching.schedule import SAMPLINGS, ScheduleConstants
from parapick.tomography import PickTable
from parapick.workflows.unstack import unstack

__all__ = ["MatchTemplateResult", "collect_picks", "extract_peaks", "run_match_template"]

logger = logging.getLogger(__name__)

SCHEDULE_FILE = "schedule.csv"
PICKS_FILE = "picks.star"
CONFIGS_DIRECTORY = "configs"
RESULTS_DIRECTORY = "results"
PEAKS_DIRECTORY = "peaks"
BINNED_DIRECTORY = "binned"
SETTINGS_FILE = "settings.json"
PICKING_FILE = "picking.json"
SCORE_COLUMN = "score"
"""Score of a peak (see :mod:`parapick.matching.peaks`), used as the score of the pick."""
LEGACY_SCORE_COLUMN = "scaled_mip"
DONE_SUFFIX = ".done"
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
    """Gather the peaks of the searched tilts as ParaPick picks (native pixels of the tilts).

    Peaks found on binned images are scaled back: binned pixel ``i`` is native ``i * binning``.
    """
    names, tilt_ids, coordinates, scores = [], [], [], []
    for tilt in plan:
        path = _peaks_path(output, tilt)
        if not tilt.selected or not path.exists():
            continue
        peaks = pd.read_csv(path)
        names += [tilt.tilt_series] * len(peaks)
        tilt_ids += [tilt.tilt_id] * len(peaks)
        position = peaks[["pos_x_img", "pos_y_img"]].to_numpy(dtype=np.float64)
        coordinates.append(position * tilt.binning)
        column = SCORE_COLUMN if SCORE_COLUMN in peaks else LEGACY_SCORE_COLUMN
        scores.append(peaks[column].to_numpy(dtype=np.float64))
    return PickTable(
        tilt_ids=np.array(tilt_ids, dtype=object),
        coordinates=np.concatenate(coordinates) if coordinates else np.empty((0, 2)),
        tilt_series=np.array(names, dtype=object),
        scores=np.concatenate(scores) if scores else np.empty(0),
    )


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= PIXEL_SIZE_TOLERANCE * b


def _prepare_template(template: Path, pixel_size: float, binning: int, output: Path) -> Path:
    """The template at the searched pixel size: as given, or binned if at the native one."""
    searched = pixel_size * binning
    size = leopard.template_pixel_size(template)
    if size is None:
        logger.warning(
            "%s has no voxel size; assuming %.4f Å/px (the tilt-series)", template, pixel_size
        )
        size = pixel_size
    if _close(size, searched):
        return template
    if binning > 1 and _close(size, pixel_size):
        binned = output / f"template_bin{binning}.mrc"
        leopard.write_binned(template, binned, binning)
        logger.info("Template binned by %d: %s", binning, binned)
        return binned
    expected = f"{pixel_size:.4f}" + (f" or {searched:.4f}" if binning > 1 else "")
    raise ValueError(
        f"The template has {size:.4f} Å/px but must have {expected} Å/px: simulate or rescale "
        "it at the pixel size of the tilt-series"
    )


def _check_settings(output: Path, settings: dict[str, object], overwrite: bool) -> None:
    """Refuse to resume a run made with other settings (its peaks would be mixed in)."""
    path = output / SETTINGS_FILE
    if path.exists() and not overwrite:
        previous = json.loads(path.read_text())
        if previous != settings:
            raise ValueError(
                f"{output} holds a run with {previous}, not {settings}: use another output "
                "directory or --overwrite"
            )
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings))


def _result_path(output: Path, tilt: PlannedTilt) -> Path:
    return output / RESULTS_DIRECTORY / f"{tilt.stem}.h5"


def _searched(output: Path, tilt: PlannedTilt) -> bool:
    """Whether the search of a tilt finished (older runs only left their peaks)."""
    result = _result_path(output, tilt)
    done = result.with_suffix(DONE_SUFFIX).exists() or _peaks_path(output, tilt).exists()
    return done and result.exists()


def _picking_changed(output: Path, picking: PickingSettings) -> bool:
    """Record the picking settings; whether the existing peaks may have been picked otherwise.

    Picking again only reads the results, so runs that did not record them are picked again.
    """
    path = output / PICKING_FILE
    previous = json.loads(path.read_text()) if path.exists() else None
    current = picking.as_dict()
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current))
    return previous != current


def extract_peaks(
    output: Path, tilt: PlannedTilt, picking: PickingSettings, half_box: int
) -> pd.DataFrame:
    """Peaks of a searched tilt, from its results (pixels of the searched image)."""
    tensors, projections = leopard.read_results(_result_path(output, tilt))
    values = score_map(picking.score, mip=tensors["mip"], scaled_mip=tensors["scaled_mip"])
    threshold = (
        gaussian_threshold(values.size * projections, picking.false_positives)
        if picking.peaks_per_tilt is None
        else None
    )
    rows, cols = find_peaks(
        values, separation=tilt.peak_separation, threshold=threshold, count=picking.peaks_per_tilt
    )
    peaks = pd.DataFrame(
        {
            "pos_x_img": cols + half_box,
            "pos_y_img": rows + half_box,
            SCORE_COLUMN: values[rows, cols],
            **{name: tensors[name][rows, cols] for name in leopard.RESULT_TENSORS},
        }
    )
    return peaks


def _write_peaks(peaks: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    peaks.to_csv(temporary, index=False)
    temporary.replace(path)


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
    score: str = "zscore",
    peaks_per_tilt: int | None = None,
    symmetry: str = "C1",
    device: str = "auto",
    orientation_batch_size: int | None = None,
    concurrent_searches: int = 3,
    binning: int = 1,
    sampling: str = "default",
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
        Template volume (MRC) at the pixel size of the tilt-series, or of the binned
        tilt-series.
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
        Expected false positives per image when thresholding the peaks (``zscore``).
    score
        Score of the peaks: ``zscore`` (Leopard-EM's) or ``mip`` (the best correlation,
        normalised over the image), see :mod:`parapick.matching.peaks`.
    peaks_per_tilt
        Keep the best N peaks of each tilt instead of thresholding (required with ``mip``).
        Changing the picking alone re-extracts the peaks from the results, without searching.
    symmetry
        Symmetry of the template (e.g. ``C1``, ``D7``), to restrict the search.
    device
        ``auto``, ``gpu`` (all visible GPUs) or ``cpu``.
    orientation_batch_size
        Orientations correlated at once. ``None``: about 4 GB of GPU memory per search
        (:func:`~parapick.matching.leopard.automatic_batch_size`).
    concurrent_searches
        Searches run at once, each in its own process: Leopard-EM's start-up of each search
        (several seconds) then overlaps with the computation of the others.
    binning
        Search the tilt images binned by this factor (Fourier cropping). The parameters
        follow the binned pixel size; the picks are in native pixels.
    sampling
        Sampling policy, a key of :data:`~parapick.matching.schedule.SAMPLINGS`:
        ``default`` or ``fast`` (steps proportional to resolution, uncapped). Ignored if
        ``constants`` is given.
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
    if sampling not in SAMPLINGS:
        raise ValueError(f"Unknown sampling {sampling!r} (choices: {', '.join(SAMPLINGS)})")
    constants = constants or SAMPLINGS[sampling]
    pixel_size = tomograms.tilt_series[0].pixel_size
    if any(not _close(ts.pixel_size, pixel_size) for ts in tomograms):
        raise ValueError("All tilt-series must have the same pixel size (one template)")
    output = output_directory
    _check_settings(output, {"binning": binning, "sampling": sampling}, overwrite)
    picking = PickingSettings(score, peaks_per_tilt, false_positives)
    template = _prepare_template(template, pixel_size, binning, output)

    # Leopard-EM reads 2D MRC files: link or extract the tilt images
    tomograms, _ = unstack(tomograms, output, anchor=tomograms_path, link=True)
    registry.tomograms_writer("relion5").write_tomograms(tomograms, output / "tomograms.star")

    plan = plan_matching(
        tomograms,
        particle_diameter=particle_diameter,
        thickness=thickness,
        dose_per_tilt=dose_per_tilt,
        energy_filter=energy_filter,
        max_tilts=max_tilts,
        binning=binning,
        constants=constants,
    )
    if binning > 1:
        binned = []
        for tilt in plan:
            target = output / BINNED_DIRECTORY / f"{tilt.stem}.mrc"
            if overwrite or not target.exists():
                leopard.write_binned(Path(tilt.image), target, binning)
            binned.append(dataclasses.replace(tilt, image=str(target)))
        plan = binned
        logger.info("Tilt images binned by %d in %s", binning, output / BINNED_DIRECTORY)
    schedule_table(plan).to_csv(output / SCHEDULE_FILE, index=False)
    selected = [t for t in plan if t.selected]
    logger.info(
        "Schedule of %d tilts written to %s; %d to search",
        len(plan),
        output / SCHEDULE_FILE,
        len(selected),
    )

    searching = not schedule_only and any(
        not _searched(output, tilt) or overwrite for tilt in selected
    )
    # Only a search needs a GPU: writing the schedule or picking again does not
    gpu_ids = leopard.gpu_ids_for(device) if searching else "all"
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

    repick = _picking_changed(output, picking) or overwrite
    half_box = leopard.template_box(template) // 2
    if concurrent_searches < 1:
        raise ValueError("concurrent_searches must be >= 1")

    def finish(n: int, tilt: PlannedTilt, searched: bool) -> None:
        peaks = extract_peaks(output, tilt, picking, half_box)
        _write_peaks(peaks, _peaks_path(output, tilt))
        logger.info(
            "[%d/%d] %s: %d peaks%s",
            n,
            len(selected),
            tilt.stem,
            len(peaks),
            " (picked again)" if searched else "",
        )

    def search(n: int, tilt: PlannedTilt) -> None:
        batch = orientation_batch_size or leopard.automatic_batch_size(
            leopard.image_shape(Path(tilt.image))
        )
        leopard.run_search_process(
            configs[tilt.stem],
            orientation_batch_size=batch,
            log=_result_path(output, tilt).with_suffix(".log"),
        )
        _result_path(output, tilt).with_suffix(DONE_SUFFIX).touch()
        finish(n, tilt, searched=False)

    pending: list[Future[None]] = []
    with ThreadPoolExecutor(max_workers=concurrent_searches) as pool:
        for n, tilt in enumerate(selected, start=1):
            peaks_path = _peaks_path(output, tilt)
            searched = _searched(output, tilt) and not overwrite
            if searched and peaks_path.exists() and not repick:
                logger.info("[%d/%d] %s: already searched", n, len(selected), tilt.stem)
            elif searched:
                finish(n, tilt, searched=True)
            else:
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
                peaks_path.unlink(missing_ok=True)
                pending.append(pool.submit(search, n, tilt))
        done, not_done = wait(pending, return_when=FIRST_EXCEPTION)
        failed = [f for f in done if f.exception() is not None]
        if failed:
            for future in not_done:
                future.cancel()
            raise failed[0].exception()  # type: ignore[misc]

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
