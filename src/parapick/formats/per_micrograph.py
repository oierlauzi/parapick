"""Shared machinery for pick formats that store one file per micrograph.

Most SPA pickers write one coordinate file per micrograph (crYOLO, EMAN, Xmipp, RELION
autopick...). A format only describes how to read (and optionally write) *one* file.
:class:`PerMicrographPicksFormat` handles directories, single files and assembly into a
:class:`~parapick.tomography.PickTable`, with the tilt identified by the file stem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from parapick.tomography import PickTable
from parapick.tomography.matching import tilt_key_variants

__all__ = [
    "MicrographPicks",
    "PerMicrographPicksFormat",
    "PerMicrographPicksWriter",
    "assemble_picks",
    "group_by_micrograph",
    "micrograph_file_stem",
]


@dataclass(frozen=True)
class MicrographPicks:
    """Picks of one micrograph: ``(M, 2)`` pixel coordinates (x, y) and optional scores."""

    coordinates: NDArray[np.float64]
    scores: NDArray[np.float64] | None = None
    tilt_id: str | None = None
    """Overrides the identifier derived from the file name, if the file records one."""
    tilt_series: str | None = None

    def __post_init__(self) -> None:
        coordinates = np.asarray(self.coordinates, dtype=np.float64).reshape(-1, 2)
        object.__setattr__(self, "coordinates", coordinates)
        if self.scores is not None:
            scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
            if scores.shape[0] != coordinates.shape[0]:
                raise ValueError("One score per pick is required")
            object.__setattr__(self, "scores", scores)


def assemble_picks(
    parts: Iterable[tuple[str, MicrographPicks]], *, pixel_size: float | None = None
) -> PickTable:
    """Concatenate per-micrograph picks into a :class:`PickTable`.

    ``parts`` are ``(tilt_id, picks)`` pairs. Scores are kept only if every non-empty
    micrograph has them.
    """
    parts = list(parts)
    non_empty = [p for _, p in parts if len(p.coordinates)]
    has_scores = bool(non_empty) and all(p.scores is not None for p in non_empty)
    tilt_ids: list[str] = []
    tilt_series: list[str | None] = []
    for tilt_id, picks in parts:
        n = len(picks.coordinates)
        tilt_ids += [picks.tilt_id or tilt_id] * n
        tilt_series += [picks.tilt_series] * n
    coordinates = [p.coordinates for _, p in parts]
    return PickTable(
        tilt_ids=np.array(tilt_ids, dtype=object),
        coordinates=np.concatenate(coordinates) if coordinates else np.empty((0, 2)),
        tilt_series=np.array(tilt_series, dtype=object),
        scores=np.concatenate([p.scores for p in non_empty if p.scores is not None])
        if has_scores
        else None,
        pixel_size=pixel_size,
    )


def group_by_micrograph(picks: PickTable) -> list[tuple[str | None, str, NDArray[np.intp]]]:
    """``(tilt_series, tilt_id, rows)`` for each micrograph, in order of first appearance."""
    groups: dict[tuple[str | None, str], list[int]] = {}
    for i, (ts_value, tilt_value) in enumerate(zip(picks.tilt_series, picks.tilt_ids, strict=True)):
        key = (None if ts_value is None else str(ts_value), str(tilt_value))
        groups.setdefault(key, []).append(i)
    return [(ts, tilt, np.asarray(rows, dtype=np.intp)) for (ts, tilt), rows in groups.items()]


def micrograph_file_stem(tilt_series: str | None, tilt_id: str) -> str:
    """Unique file stem for a micrograph: its stem, prefixed by the tilt-series if absent."""
    stem = tilt_key_variants(tilt_id)[2]
    return f"{tilt_series}_{stem}" if tilt_series and not stem.startswith(tilt_series) else stem


class PerMicrographPicksFormat(ABC):
    """Base class of pick readers with one file per micrograph.

    Subclasses set ``name``, ``description`` and ``suffixes`` and implement :meth:`read_file`.
    :meth:`sniff_file` can add a cheap content check.
    """

    name: str
    description: str
    suffixes: tuple[str, ...]

    @abstractmethod
    def read_file(self, path: Path) -> MicrographPicks:
        """Read the picks of one micrograph."""

    def sniff_file(self, path: Path) -> bool:
        """Content check for a single file with a matching suffix (default: accept)."""
        return True

    def tilt_id(self, path: Path) -> str:
        """Identifier of the micrograph a file belongs to.

        It is the file name without this format's suffix, then without known image extensions
        (``a.mrc.box`` -> ``a``).
        """
        name = path.name
        if path.suffix.lower() in self.suffixes:
            name = name[: -len(path.suffix)]
        return tilt_key_variants(name)[2]

    def files(self, path: Path) -> list[Path]:
        """The coordinate files of a directory (sorted), or the file itself."""
        if path.is_dir():
            return sorted(
                f for f in path.iterdir() if f.is_file() and f.suffix.lower() in self.suffixes
            )
        return [path]

    def sniff(self, path: Path) -> bool:
        """A directory holding files of this format, or a matching file."""
        if path.is_dir():
            return bool(self.files(path))
        return path.is_file() and path.suffix.lower() in self.suffixes and self.sniff_file(path)

    def read_picks(self, path: Path) -> PickTable:
        """Read a directory of coordinate files, or a single one."""
        if not path.exists():
            raise FileNotFoundError(path)
        return assemble_picks((self.tilt_id(f), self.read_file(f)) for f in self.files(path))


class PerMicrographPicksWriter(ABC):
    """Mixin giving a :class:`PerMicrographPicksFormat` a ``write_picks`` method.

    ``path`` is the output directory. Each micrograph is written to
    ``<path>/<stem><suffix>``, where ``<stem>`` comes from :func:`micrograph_file_stem`.
    """

    suffixes: tuple[str, ...]

    @abstractmethod
    def write_file(self, picks: MicrographPicks, path: Path) -> None:
        """Write the picks of one micrograph."""

    def write_picks(self, picks: PickTable, path: Path) -> list[Path]:
        """Write one file per micrograph into the directory ``path``."""
        path.mkdir(parents=True, exist_ok=True)
        written = []
        for ts_name, tilt_id, rows in group_by_micrograph(picks):
            target = path / f"{micrograph_file_stem(ts_name, tilt_id)}{self.suffixes[0]}"
            self.write_file(
                MicrographPicks(
                    picks.coordinates[rows],
                    None if picks.scores is None else picks.scores[rows],
                ),
                target,
            )
            written.append(target)
        return written
