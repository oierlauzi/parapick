"""Per-micrograph RELION coordinate files, as written by RELION autopick, crYOLO, Topaz, etc.

Accepted inputs:

* an *index* STAR file with ``rlnMicrographName`` and ``rlnMicrographCoordinates`` (e.g.
  RELION's ``autopick.star``), optionally with ``rlnTomoName``;
* a *directory* of per-micrograph coordinate STAR files; the tilt is the file stem;
* a single per-micrograph coordinate STAR file.

Each coordinate file holds ``rlnCoordinateX``/``rlnCoordinateY`` in pixels, optionally
``rlnAutopickFigureOfMerit``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from parapick.formats.relion5 import labels as L
from parapick.formats.star import (
    find_loop,
    peek_labels,
    read_blocks,
    resolve_relative_path,
    write_blocks,
)
from parapick.tomography import PickTable
from parapick.tomography.matching import tilt_key_variants

__all__ = ["RelionCoordinatesFormat"]

COORDINATES_DIRECTORY = "coordinates"


def _read_coordinate_file(
    path: Path, tilt_id: str, tilt_series: str | None
) -> tuple[pd.DataFrame, str, str | None]:
    blocks = read_blocks(path)
    if not blocks:
        empty = pd.DataFrame({L.COORDINATE_X: [], L.COORDINATE_Y: []})
        return empty, tilt_id, tilt_series
    table = find_loop(blocks, [L.COORDINATE_X, L.COORDINATE_Y], path=path)
    return table, tilt_id, tilt_series


def _to_table(parts: list[tuple[pd.DataFrame, str, str | None]]) -> PickTable:
    tilt_ids: list[str] = []
    tilt_series: list[str | None] = []
    coordinates: list[NDArray[np.float64]] = []
    scores: list[NDArray[np.float64]] = []
    has_scores = all(L.AUTOPICK_FOM in table.columns for table, _, _ in parts if len(table))
    for table, tilt_id, ts_name in parts:
        n = len(table)
        tilt_ids += [tilt_id] * n
        tilt_series += [ts_name] * n
        coordinates.append(table[[L.COORDINATE_X, L.COORDINATE_Y]].to_numpy(dtype=np.float64))
        if has_scores and n:
            scores.append(table[L.AUTOPICK_FOM].to_numpy(dtype=np.float64))
    return PickTable(
        tilt_ids=np.array(tilt_ids, dtype=object),
        coordinates=np.concatenate(coordinates) if coordinates else np.empty((0, 2)),
        tilt_series=np.array(tilt_series, dtype=object),
        scores=(np.concatenate(scores) if scores else np.empty(0)) if has_scores else None,
    )


class RelionCoordinatesFormat:
    """RELION single-particle coordinate files, one per tilt image."""

    name = "relion-coords"
    description = "Per-micrograph RELION coordinate STAR files (directory or autopick index)"
    suffixes = (".star",)

    def sniff(self, path: Path) -> bool:
        """True for coordinate directories, coordinate indices and single coordinate files."""
        if path.is_dir():
            return any(path.glob("*.star"))
        found = peek_labels(path)
        return L.MICROGRAPH_COORDINATES in found or (
            L.COORDINATE_X in found and L.MICROGRAPH_NAME not in found and L.TOMO_NAME not in found
        )

    def read_picks(self, path: Path) -> PickTable:
        """Read an index, a directory or a single coordinate file."""
        if path.is_dir():
            parts = [
                _read_coordinate_file(f, tilt_key_variants(f.name)[2], None)
                for f in sorted(path.glob("*.star"))
            ]
            return _to_table(parts)
        blocks = read_blocks(path)
        try:
            index = find_loop(blocks, [L.MICROGRAPH_NAME, L.MICROGRAPH_COORDINATES], path=path)
        except ValueError:
            return _to_table([_read_coordinate_file(path, tilt_key_variants(path.name)[2], None)])
        parts = []
        for _, row in index.iterrows():
            ts_name = str(row[L.TOMO_NAME]) if L.TOMO_NAME in index.columns else None
            coords_path = resolve_relative_path(str(row[L.MICROGRAPH_COORDINATES]), path)
            if coords_path.exists():
                parts.append(
                    _read_coordinate_file(coords_path, str(row[L.MICROGRAPH_NAME]), ts_name)
                )
        return _to_table(parts)

    def write_picks(self, picks: PickTable, path: Path) -> list[Path]:
        """Write ``path`` as an index and one coordinate file per tilt image next to it."""
        directory = path.parent / COORDINATES_DIRECTORY
        written: list[Path] = []
        index_rows: list[dict[str, str]] = []
        groups: dict[tuple[str | None, str], list[int]] = {}
        for i, (ts_value, tilt_value) in enumerate(
            zip(picks.tilt_series, picks.tilt_ids, strict=True)
        ):
            key = (None if ts_value is None else str(ts_value), str(tilt_value))
            groups.setdefault(key, []).append(i)
        for (ts_name, tilt_id), group in groups.items():
            stem = tilt_key_variants(tilt_id)[2]
            prefix = f"{ts_name}_" if ts_name and not stem.startswith(ts_name) else ""
            coords_path = directory / f"{prefix}{stem}.star"
            table = {
                L.COORDINATE_X: picks.coordinates[group, 0],
                L.COORDINATE_Y: picks.coordinates[group, 1],
            }
            if picks.scores is not None:
                table[L.AUTOPICK_FOM] = picks.scores[group]
            written.append(write_blocks({"": pd.DataFrame(table)}, coords_path))
            entry = {L.MICROGRAPH_NAME: tilt_id, L.MICROGRAPH_COORDINATES: str(coords_path)}
            if ts_name is not None:
                entry = {L.TOMO_NAME: ts_name, **entry}
            index_rows.append(entry)
        written.insert(0, write_blocks({"coordinate_files": pd.DataFrame(index_rows)}, path))
        return written
