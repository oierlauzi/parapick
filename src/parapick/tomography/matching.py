"""Matching of raw picks to the tilts of a :class:`~parapick.tomography.TomogramSet`."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePath

import numpy as np
from numpy.typing import NDArray

from parapick.tomography.picks import BoundPicks, PickTable
from parapick.tomography.tiltseries import TiltSeries, TomogramSet

__all__ = ["MatchReport", "annotate_picks", "bind_picks", "locate_picks", "tilt_key_variants"]

logger = logging.getLogger(__name__)

_AMBIGUOUS = (-1, -1)


def _strip_index(tilt_id: str) -> str:
    """Remove a RELION ``index@`` stack prefix, if present."""
    head, sep, tail = tilt_id.partition("@")
    return tail if sep and head.isdigit() else tilt_id


def _name(tilt_id: str) -> str:
    return PurePath(_strip_index(tilt_id)).name


_KNOWN_SUFFIXES = frozenset(
    {".mrc", ".mrcs", ".st", ".ali", ".tif", ".tiff", ".eer", ".em", ".spi", ".hdf", ".h5",
     ".png", ".jpg", ".jpeg", ".star", ".txt", ".tsv", ".csv", ".box", ".cbox", ".coords",
     ".pos", ".cs",
     ".gz", ".bz2"}
)  # fmt: skip


def _stem(tilt_id: str) -> str:
    """File name without known image/coordinate extensions (``a.mrc.gz`` -> ``a``).

    Only known extensions are removed, so names such as ``TS_01_-21.00.mrc`` keep their dots.
    """
    path = PurePath(_name(tilt_id))
    while path.suffix.lower() in _KNOWN_SUFFIXES and path.stem:
        path = PurePath(path.stem)
    return path.name


_KEYS: tuple[Callable[[str], str], ...] = (str, _name, _stem)


def tilt_key_variants(tilt_id: str) -> tuple[str, ...]:
    """Keys under which a tilt identifier is matched: exact, file name, then stem."""
    return tuple(key(tilt_id) for key in _KEYS)


@dataclass(frozen=True)
class MatchReport:
    """Summary of how many picks were matched to tilts."""

    matched: int
    unmatched: int
    unmatched_examples: tuple[str, ...]


def _build_indices(
    tomograms: TomogramSet,
) -> tuple[list[dict[str, tuple[int, int]]], dict[str, list[dict[str, tuple[int, int]]]]]:
    """Return global and per-tilt-series lookup tables (one dict per key level)."""
    global_index: list[dict[str, tuple[int, int]]] = [{} for _ in _KEYS]
    local_index: dict[str, list[dict[str, tuple[int, int]]]] = {}
    for s, ts in enumerate(tomograms):
        local: list[dict[str, tuple[int, int]]] = [{} for _ in _KEYS]
        for i, tilt_id in enumerate(ts.tilt_ids):
            for level, key in enumerate(tilt_key_variants(tilt_id)):
                for table in (local[level], global_index[level]):
                    table[key] = _AMBIGUOUS if key in table else (s, i)
        local_index[ts.name] = local
    return global_index, local_index


def _resolve(tilt_id: str, index: list[dict[str, tuple[int, int]]]) -> tuple[int, int] | None:
    for level, key in enumerate(tilt_key_variants(tilt_id)):
        hit = index[level].get(key)
        if hit is None:
            continue
        if hit == _AMBIGUOUS:
            raise ValueError(
                f"Pick tilt identifier {tilt_id!r} matches several tilts; provide the "
                "tilt-series name or unambiguous micrograph names"
            )
        return hit
    return None


def locate_picks(
    picks: PickTable, tomograms: TomogramSet
) -> tuple[NDArray[np.intp], NDArray[np.intp], MatchReport]:
    """Find the tilt-series and tilt index of every pick.

    Tilt identifiers are matched exactly first, then by file name, then by stem (file name
    without image/coordinate extensions). A RELION ``index@`` prefix is ignored. When a pick
    carries a tilt-series name, the search is restricted to that tilt-series.

    Returns
    -------
    tilt_series_index, tilt_index, report
        Per-pick indices into ``tomograms`` and into the tilts of that tilt-series. Picks that
        cannot be matched are ``-1`` in both arrays.
    """
    global_index, local_index = _build_indices(tomograms)
    ts_of_pick = np.full(len(picks), -1, dtype=np.intp)
    tilt_of_pick = np.full(len(picks), -1, dtype=np.intp)
    groups: dict[tuple[object, str], list[int]] = defaultdict(list)
    for row, (ts_name, tilt_id) in enumerate(zip(picks.tilt_series, picks.tilt_ids, strict=True)):
        groups[(ts_name, str(tilt_id))].append(row)

    unknown_series: set[str] = set()
    unmatched_ids: list[str] = []
    for (ts_name, tilt_id), rows in groups.items():
        if ts_name is None or (isinstance(ts_name, float) and np.isnan(ts_name)) or ts_name == "":
            index = global_index
        elif str(ts_name) in local_index:
            index = local_index[str(ts_name)]
        else:
            unknown_series.add(str(ts_name))
            continue
        hit = _resolve(tilt_id, index)
        if hit is None:
            unmatched_ids.append(tilt_id)
            continue
        ts_of_pick[rows], tilt_of_pick[rows] = hit

    if unknown_series:
        logger.warning(
            "Ignoring picks of %d tilt-series absent from the geometry: %s",
            len(unknown_series),
            ", ".join(sorted(unknown_series)[:5]),
        )
    matched = ts_of_pick >= 0
    report = MatchReport(
        matched=int(matched.sum()),
        unmatched=int((~matched).sum()),
        unmatched_examples=tuple(sorted(unmatched_ids)[:5]),
    )
    if unmatched_ids:
        logger.warning(
            "%d picks on %d tilt images could not be matched to any tilt (e.g. %s)",
            report.unmatched,
            len(unmatched_ids),
            ", ".join(report.unmatched_examples),
        )
    return ts_of_pick, tilt_of_pick, report


def _image_height(ts: TiltSeries, pixel_size: float) -> float:
    """Height of the tilt images of ``ts`` in pixels of size ``pixel_size``."""
    if ts.image_size is None:
        raise ValueError(f"Flipping y needs the tilt-image size of {ts.name!r}")
    return ts.image_size[1] * ts.pixel_size / pixel_size


def bind_picks(
    picks: PickTable,
    tomograms: TomogramSet,
    *,
    pixel_size: float | None = None,
    flip_y: bool = False,
) -> tuple[dict[str, BoundPicks], MatchReport]:
    """Assign picks to tilts (see :func:`locate_picks`) and convert them to image-frame Å.

    Parameters
    ----------
    picks
        Raw picks.
    tomograms
        Geometry the picks refer to.
    pixel_size
        Pixel size (Å/px) of the images the picks were made on. It overrides
        ``picks.pixel_size``. If both are ``None``, the native pixel size of each tilt-series is
        used.
    flip_y
        Mirror the y coordinates on the image height (``y ← H − y``). Use it for pickers or
        micrographs whose y axis points the other way.

    Returns
    -------
    dict, MatchReport
        Bound picks for every tilt-series of ``tomograms`` (possibly empty), and a summary.
        Picks that cannot be matched are dropped and reported.
    """
    ts_of_pick, tilt_of_pick, report = locate_picks(picks, tomograms)
    effective_pixel_size = pixel_size if pixel_size is not None else picks.pixel_size
    bound: dict[str, BoundPicks] = {}
    for s, ts in enumerate(tomograms):
        rows = np.flatnonzero(ts_of_pick == s)
        scale = effective_pixel_size if effective_pixel_size is not None else ts.pixel_size
        coordinates = picks.coordinates[rows].copy()
        if flip_y and len(rows):
            coordinates[:, 1] = _image_height(ts, scale) - coordinates[:, 1]
        bound[ts.name] = BoundPicks(
            tilt_series=ts.name,
            tilt_indices=tilt_of_pick[rows],
            coordinates=coordinates * scale,
            scores=None if picks.scores is None else picks.scores[rows],
        )
    return bound, report


def annotate_picks(
    picks: PickTable, tomograms: TomogramSet, *, flip_y: bool = False
) -> tuple[PickTable, MatchReport]:
    """Replace tilt identifiers by the canonical ones of ``tomograms`` and add tilt-series names.

    Coordinates and pixel size are unchanged (unless ``flip_y``, see :func:`bind_picks`);
    unmatched picks are dropped.
    """
    ts_of_pick, tilt_of_pick, report = locate_picks(picks, tomograms)
    keep = ts_of_pick >= 0
    series = tomograms.tilt_series
    kept = picks.select(keep)
    names = [series[s].name for s in ts_of_pick[keep]]
    ids = [series[s].tilt_ids[i] for s, i in zip(ts_of_pick[keep], tilt_of_pick[keep], strict=True)]
    coordinates = kept.coordinates.copy()
    if flip_y:
        for s in np.unique(ts_of_pick[keep]):
            rows = ts_of_pick[keep] == s
            height = _image_height(series[s], kept.pixel_size or series[s].pixel_size)
            coordinates[rows, 1] = height - coordinates[rows, 1]
    return (
        PickTable(
            tilt_ids=np.array(ids, dtype=object),
            coordinates=coordinates,
            tilt_series=np.array(names, dtype=object),
            scores=kept.scores,
            pixel_size=kept.pixel_size,
        ),
        report,
    )
