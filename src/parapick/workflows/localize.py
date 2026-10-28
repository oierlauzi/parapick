"""Main use case: localise particles in 3D from 2D picks on tilt images."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from parapick.formats import (
    AUTO,
    FormatRegistry,
    TomogramsReadOptions,
    WriteContext,
    default_registry,
)
from parapick.formats.relion5.assignments import write_assignments as write_assignments_star
from parapick.gmm import EMFitter, Fitter, GMMFit, LocalizeConfig, assign_detections
from parapick.tomography import (
    BoundPicks,
    MatchReport,
    Particles,
    PickAssignments,
    PickTable,
    TiltSeries,
    TomogramSet,
    bind_picks,
)
from parapick.workflows.filter import Selection, select_particles

__all__ = [
    "LocalizationResult",
    "TiltSeriesLocalization",
    "localize",
    "localize_tilt_series",
    "run_localize",
]

logger = logging.getLogger(__name__)

REPORT_NAME = "localize_report.json"
ASSIGNMENTS_NAME = "assignments.star"
DEFAULT_ASSIGNMENT_THRESHOLD = 0.5


@dataclass(frozen=True)
class TiltSeriesLocalization:
    """Fit of one tilt-series."""

    tilt_series: TiltSeries
    fit: GMMFit
    picks_per_tilt: NDArray[np.intp]
    elapsed: float
    assignments: PickAssignments | None = None
    """Picks explained by each particle (see :func:`~parapick.gmm.assign_detections`)."""

    @property
    def name(self) -> str:
        """Tilt-series name."""
        return self.tilt_series.name

    def particles(self) -> Particles:
        """The localised particles of this tilt-series (sorted by decreasing score).

        Particles are numbered ``1..K`` in this order (:attr:`Particles.ids`).
        """
        n = self.fit.n_components
        return Particles(
            tilt_series=np.full(n, self.name, dtype=object),
            coordinates=self.fit.means,
            scores=self.fit.scores,
            support=self.fit.support,
            ids=np.arange(1, n + 1),
        )

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable diagnostics."""
        fit = self.fit
        return {
            "tilt_series": self.name,
            "n_tilts": self.tilt_series.n_tilts,
            "n_picks": int(self.picks_per_tilt.sum()),
            "initial_components": fit.n_initial_components,
            "components": fit.n_components,
            "iterations": fit.n_iterations,
            "converged": fit.converged,
            "log_likelihood": fit.log_likelihood,
            "background_fraction": fit.background_fraction,
            "elapsed_seconds": round(self.elapsed, 3),
            "assigned_picks": None if self.assignments is None else len(self.assignments),
            "tilts": [
                {"tilt_id": tilt_id, "n_picks": int(n), "sigma_angst": float(s)}
                for tilt_id, n, s in zip(
                    self.tilt_series.tilt_ids, self.picks_per_tilt, fit.sigmas, strict=True
                )
            ],
        }


@dataclass(frozen=True)
class LocalizationResult:
    """Fits of every processed tilt-series."""

    results: tuple[TiltSeriesLocalization, ...]
    match_report: MatchReport | None = None
    config: LocalizeConfig = field(default_factory=LocalizeConfig)

    def particles(self, selection: Selection | None = None) -> Particles:
        """All localised particles, optionally filtered."""
        particles = Particles.concatenate([r.particles() for r in self.results])
        return particles if selection is None else select_particles(particles, selection)

    def assignments(self, selection: Selection | None = None) -> PickAssignments:
        """Picks assigned to the particles, optionally only those kept by ``selection``."""
        table = PickAssignments.concatenate(
            [r.assignments for r in self.results if r.assignments is not None]
        )
        if selection is None or selection.is_empty:
            return table
        kept = self.particles(selection)
        assert kept.ids is not None
        keys = set(zip((str(n) for n in kept.tilt_series), kept.ids.tolist(), strict=True))
        mask = np.fromiter(
            (
                (str(n), int(k)) in keys
                for n, k in zip(table.tilt_series, table.particle_ids, strict=True)
            ),
            dtype=bool,
            count=len(table),
        )
        return table.select(mask)

    def report(self) -> dict[str, Any]:
        """JSON-serialisable diagnostics of the whole run."""
        report: dict[str, Any] = {
            "config": dict(vars(self.config)),
            "tilt_series": [r.summary() for r in self.results],
        }
        if self.match_report is not None:
            report["picks"] = {
                "matched": self.match_report.matched,
                "unmatched": self.match_report.unmatched,
            }
        return report


def _resolve_box(
    ts: TiltSeries, detections: Sequence[NDArray[np.float64]], config: LocalizeConfig
) -> tuple[float, float, float]:
    if config.box is not None:
        return config.box
    if ts.volume_size is not None:
        return ts.volume_size
    extent = max((float(np.abs(d).max()) for d in detections if len(d)), default=1.0)
    logger.warning(
        "Tilt-series %r has no volume size; initialising in a %.0f Å box (use --box to set it)",
        ts.name,
        2 * extent,
    )
    return (2 * extent, 2 * extent, extent / 2)


def localize_tilt_series(
    tilt_series: TiltSeries,
    picks: BoundPicks,
    *,
    config: LocalizeConfig | None = None,
    fitter: Fitter | None = None,
    seed: int = 0,
    assignment_threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
) -> TiltSeriesLocalization:
    """Localise the particles of one tilt-series from its bound picks.

    Parameters
    ----------
    tilt_series
        Geometry.
    picks
        Picks of this tilt-series in image-frame Å (see :func:`parapick.tomography.bind_picks`).
    config
        Fit hyper-parameters. Ignored if ``fitter`` is given.
    fitter
        Custom fitter. Defaults to :class:`~parapick.gmm.EMFitter` with ``config``.
    seed
        Random seed of the initialisation.
    assignment_threshold
        Minimum responsibility for a pick to be assigned to a particle.
    """
    config = config or LocalizeConfig()
    fitter = fitter or EMFitter(config)
    per_tilt = picks.per_tilt(tilt_series.n_tilts)
    # Remove each tilt's translation: ŷ_ij = y_ij - t_i.
    detections = [d - t for d, t in zip(per_tilt, tilt_series.translations, strict=True)]
    box = _resolve_box(tilt_series, detections, config)
    start = time.perf_counter()
    fit = fitter.fit(detections, tilt_series.projections, box, seed=seed)
    elapsed = time.perf_counter() - start
    counts = np.array([len(d) for d in per_tilt], dtype=np.intp)
    logger.info(
        "%s: %d picks on %d tilts -> %d particles (%d iterations%s, %.1fs)",
        tilt_series.name,
        counts.sum(),
        tilt_series.n_tilts,
        fit.n_components,
        fit.n_iterations,
        "" if fit.converged else ", NOT converged",
        elapsed,
    )
    assignments = _assign(tilt_series, picks, fit, detections, config, assignment_threshold)
    return TiltSeriesLocalization(tilt_series, fit, counts, elapsed, assignments)


def _assign(
    tilt_series: TiltSeries,
    picks: BoundPicks,
    fit: GMMFit,
    detections: Sequence[NDArray[np.float64]],
    config: LocalizeConfig,
    threshold: float,
) -> PickAssignments:
    """Hard-assign the picks to the particles and trace them back to the input rows."""
    per_tilt = assign_detections(
        fit,
        detections,
        tilt_series.projections,
        threshold=threshold,
        sinkhorn_iterations=config.sinkhorn_iterations,
    )
    # ``detections`` follow ``picks.tilt_order()``.
    order = picks.tilt_order()
    components = np.concatenate([a.components for a in per_tilt]) if per_tilt else np.empty(0)
    assigned = np.flatnonzero(components >= 0)
    if len(assigned) == 0:
        return PickAssignments.empty()
    responsibilities = np.concatenate([a.responsibilities for a in per_tilt])
    residuals = np.concatenate([a.residuals for a in per_tilt])
    source = order[assigned]
    tilt_indices = picks.tilt_indices[source]
    rows = picks.rows[source] if picks.rows is not None else np.full(len(source), -1)
    return PickAssignments(
        tilt_series=np.full(len(source), tilt_series.name, dtype=object),
        particle_ids=components[assigned].astype(np.intp) + 1,
        tilt_ids=np.array([tilt_series.tilt_ids[i] for i in tilt_indices], dtype=object),
        tilt_indices=tilt_indices,
        pick_rows=rows,
        coordinates=picks.coordinates[source],
        responsibilities=responsibilities[assigned],
        residuals=residuals[assigned],
    )


def localize(
    tomograms: TomogramSet,
    picks: PickTable | Mapping[str, BoundPicks],
    *,
    config: LocalizeConfig | None = None,
    fitter: Fitter | None = None,
    pixel_size: float | None = None,
    min_pick_score: float | None = None,
    flip_y: bool = False,
    seed: int = 0,
    progress: Callable[[TiltSeriesLocalization], None] | None = None,
    assignment_threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
) -> LocalizationResult:
    """Localise particles in every tilt-series of ``tomograms``.

    Parameters
    ----------
    tomograms
        Geometry of the tilt-series to process. Tilt-series without picks yield no particles.
    picks
        Raw picks (matched to tilts here), or picks already bound per tilt-series.
    config
        Fit hyper-parameters.
    fitter
        Custom fitter, e.g. with a different initialiser or device.
    pixel_size
        Pixel size (Å/px) of the picked images if it differs from the tilt-series' and is not
        recorded in ``picks``.
    min_pick_score
        Discard 2D picks whose figure of merit is below this value.
    flip_y
        Mirror the picks' y coordinates on the image height (see
        :func:`~parapick.tomography.bind_picks`).
    seed
        Base random seed. Tilt-series ``i`` uses ``seed + i``.
    progress
        Called after each tilt-series is processed.
    assignment_threshold
        Minimum responsibility for a pick to be assigned to a particle. Assignments refer to
        the rows of ``picks`` when it is a :class:`PickTable`.
    """
    config = config or LocalizeConfig()
    fitter = fitter or EMFitter(config)
    match_report = None
    if isinstance(picks, PickTable):
        kept_rows = None
        if min_pick_score is not None:
            if picks.scores is None:
                raise ValueError("The picks have no scores to threshold")
            keep = picks.scores >= min_pick_score
            kept_rows = np.flatnonzero(keep)
            picks = picks.select(keep)
        bound, match_report = bind_picks(picks, tomograms, pixel_size=pixel_size, flip_y=flip_y)
        if kept_rows is not None:
            # Refer to the rows of the table given by the caller, not of the filtered one.
            bound = {
                name: b if b.rows is None else replace(b, rows=kept_rows[b.rows])
                for name, b in bound.items()
            }
    else:
        bound = dict(picks)

    results = []
    for index, ts in enumerate(tomograms):
        ts_picks = bound.get(ts.name)
        if ts_picks is None:
            ts_picks = BoundPicks(ts.name, np.empty(0, dtype=np.intp), np.empty((0, 2)))
        result = localize_tilt_series(
            ts,
            ts_picks,
            config=config,
            fitter=fitter,
            seed=seed + index,
            assignment_threshold=assignment_threshold,
        )
        results.append(result)
        if progress is not None:
            progress(result)
    return LocalizationResult(tuple(results), match_report, config)


def run_localize(
    tomograms_path: Path,
    picks_path: Path,
    output_directory: Path,
    *,
    config: LocalizeConfig | None = None,
    fitter: Fitter | None = None,
    tomograms_format: str = AUTO,
    picks_format: str = AUTO,
    output_format: str = "relion5",
    read_options: TomogramsReadOptions | None = None,
    pixel_size: float | None = None,
    min_pick_score: float | None = None,
    flip_y: bool = False,
    tilt_series: Sequence[str] | None = None,
    selection: Selection | None = None,
    seed: int = 0,
    registry: FormatRegistry | None = None,
    write_assignments: bool = False,
    assignment_threshold: float = DEFAULT_ASSIGNMENT_THRESHOLD,
) -> tuple[LocalizationResult, list[Path]]:
    """File-based :func:`localize`: read, fit, and write particles plus a JSON report.

    With ``write_assignments``, also write ``assignments.star``: the picks explained by each
    selected particle (RELION 5 STAR, see :mod:`parapick.formats.relion5.assignments`).

    Returns the result and the list of files written.
    """
    registry = registry or default_registry()
    reader = registry.tomograms_reader(tomograms_format, tomograms_path)
    tomograms = reader.read_tomograms(tomograms_path, read_options or TomogramsReadOptions())
    if tilt_series:
        tomograms = tomograms.subset(tilt_series)
    picks = registry.picks_reader(picks_format, picks_path).read_picks(picks_path)
    logger.info("Read %d tilt-series (%s) and %d picks", len(tomograms), reader.name, len(picks))

    result = localize(
        tomograms,
        picks,
        config=config,
        fitter=fitter,
        pixel_size=pixel_size,
        min_pick_score=min_pick_score,
        flip_y=flip_y,
        seed=seed,
        assignment_threshold=assignment_threshold,
    )
    particles = result.particles(selection)

    writer = registry.particles_writer(output_format)
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / f"particles{writer.suffixes[0] if writer.suffixes else ''}"
    written = writer.write_particles(
        particles, output, WriteContext(tomograms=tomograms, tomograms_path=tomograms_path)
    )
    if write_assignments:
        assignments = result.assignments(selection)
        written.extend(
            write_assignments_star(assignments, output_directory / ASSIGNMENTS_NAME, picks)
        )
        logger.info("Wrote %d pick assignments", len(assignments))
    report_path = output_directory / REPORT_NAME
    report_path.write_text(json.dumps(result.report(), indent=2, default=str))
    written.append(report_path)
    logger.info("Wrote %d particles to %s", len(particles), output)
    return result, written
