"""RELION 5 ``tomograms.star`` (+ per tilt-series STAR files) reader and writer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from parapick.formats.base import TomogramsReadOptions
from parapick.formats.mrc import read_image_size, strip_stack_index
from parapick.formats.relion5 import labels as L
from parapick.formats.relion5.conventions import from_relion, to_relion
from parapick.formats.star import (
    find_loop,
    read_blocks,
    require_columns,
    resolve_relative_path,
    write_blocks,
)
from parapick.tomography import TiltSeries, TomogramSet

__all__ = ["read_tomograms", "write_tomograms"]

logger = logging.getLogger(__name__)

_DEFAULT_OPTICS: dict[str, Any] = {
    L.VOLTAGE: 300.0,
    L.SPHERICAL_ABERRATION: 2.7,
    L.AMPLITUDE_CONTRAST: 0.1,
    L.TOMO_HAND: -1.0,
}


def _column(frame: pd.DataFrame, label: str, default: float = 0.0) -> np.ndarray[Any, Any]:
    if label in frame.columns:
        return frame[label].to_numpy(dtype=np.float64)
    return np.full(len(frame), default)


def _image_size(
    row: pd.Series[Any],
    tilts: pd.DataFrame,
    anchor: Path,
    options: TomogramsReadOptions,
) -> tuple[int, int]:
    if options.tilt_image_size is not None:
        return options.tilt_image_size
    if all(label in row.index and pd.notna(row[label]) for label in L.TILT_IMAGE_SIZE):
        return int(row[L.TILT_IMAGE_SIZE[0]]), int(row[L.TILT_IMAGE_SIZE[1]])
    image = resolve_relative_path(strip_stack_index(str(tilts[L.MICROGRAPH_NAME].iloc[0])), anchor)
    try:
        return read_image_size(image)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Cannot determine the tilt-image size of {row[L.TOMO_NAME]!r}: failed to read the "
            f"header of {image} ({error}). Run from the RELION project directory, add the "
            f"{L.TILT_IMAGE_SIZE[0]}/{L.TILT_IMAGE_SIZE[1]} labels, or pass the size "
            "explicitly (--tilt-image-size)."
        ) from error


def read_tomograms(path: Path, options: TomogramsReadOptions) -> TomogramSet:
    """Read a RELION 5 tomogram set (``tomograms.star`` and its tilt-series STAR files)."""
    blocks = read_blocks(path)
    table = find_loop(
        blocks, [L.TOMO_NAME, L.TILT_SERIES_STAR], preferred=L.GLOBAL_BLOCK, path=path
    )
    require_columns(table, [L.TOMO_NAME, L.TILT_SERIES_STAR, L.TILT_SERIES_PIXEL_SIZE], path)
    if any(label in table.columns for label in L.DEFORMATION_GRID):
        logger.warning("%s contains deformation grids; ParaPick ignores them", path)

    series = []
    for _, row in table.iterrows():
        name = str(row[L.TOMO_NAME])
        ts_path = resolve_relative_path(str(row[L.TILT_SERIES_STAR]), path)
        ts_blocks = read_blocks(ts_path)
        tilts = find_loop(ts_blocks, [L.MICROGRAPH_NAME, L.Z_ROT], preferred=name, path=ts_path)
        require_columns(tilts, [L.MICROGRAPH_NAME, L.Y_TILT], ts_path)

        pixel_size = float(row[L.TILT_SERIES_PIXEL_SIZE])
        image_size = _image_size(row, tilts, path, options)
        tomo_size_px = (
            row[list(L.TOMO_SIZE)].to_numpy(dtype=np.float64)
            if all(label in row.index for label in L.TOMO_SIZE)
            else None
        )
        projections, translations = from_relion(
            _column(tilts, L.X_TILT),
            _column(tilts, L.Y_TILT),
            _column(tilts, L.Z_ROT),
            _column(tilts, L.X_SHIFT_ANGST),
            _column(tilts, L.Y_SHIFT_ANGST),
            pixel_size=pixel_size,
            image_size=image_size,
            tomo_size_px=tomo_size_px,
        )
        series.append(
            TiltSeries(
                name=name,
                tilt_ids=tuple(str(m) for m in tilts[L.MICROGRAPH_NAME]),
                projections=projections,
                translations=translations,
                pixel_size=pixel_size,
                image_size=image_size,
                volume_size=None
                if tomo_size_px is None
                else (
                    float(tomo_size_px[0] * pixel_size),
                    float(tomo_size_px[1] * pixel_size),
                    float(tomo_size_px[2] * pixel_size),
                ),
                metadata={"relion5.global": row.to_dict(), "relion5.tilts": tilts},
            )
        )
    return TomogramSet(tuple(series), metadata={"relion5.path": path})


def write_tomograms(tomograms: TomogramSet, path: Path) -> list[Path]:
    """Write a RELION 5 tomogram set: ``path`` plus ``<path dir>/tilt_series/<name>.star``.

    Format-specific metadata read from RELION files (CTF, dose, ...) is preserved.
    """
    written: list[Path] = []
    rows = []
    for ts in tomograms:
        if ts.image_size is None:
            raise ValueError(f"Tilt-series {ts.name!r} needs an image size to be written")
        tomo_size_px = (
            None
            if ts.volume_size is None
            else np.round(np.asarray(ts.volume_size) / ts.pixel_size).astype(int)
        )
        angles = to_relion(
            ts.projections,
            ts.translations,
            pixel_size=ts.pixel_size,
            image_size=ts.image_size,
            tomo_size_px=tomo_size_px,
        )
        previous = ts.metadata.get("relion5.tilts")
        tilts = (
            previous.copy()
            if isinstance(previous, pd.DataFrame) and len(previous) == ts.n_tilts
            else pd.DataFrame({L.NOMINAL_TILT: np.round(ts.tilt_angles, 2)})
        )
        tilts[L.MICROGRAPH_NAME] = list(ts.tilt_ids)
        tilts[L.X_TILT] = angles["x_tilt"]
        tilts[L.Y_TILT] = angles["y_tilt"]
        tilts[L.Z_ROT] = angles["z_rot"]
        tilts[L.X_SHIFT_ANGST] = angles["x_shift_angst"]
        tilts[L.Y_SHIFT_ANGST] = angles["y_shift_angst"]
        ts_path = path.parent / "tilt_series" / f"{ts.name}.star"
        written.append(write_blocks({ts.name: tilts}, ts_path))

        row: dict[str, Any] = dict(_DEFAULT_OPTICS)
        row.update(ts.metadata.get("relion5.global", {}))
        row[L.TOMO_NAME] = ts.name
        row[L.TILT_SERIES_STAR] = str(ts_path)
        row[L.TILT_SERIES_PIXEL_SIZE] = ts.pixel_size
        if tomo_size_px is not None:
            row.update(dict(zip(L.TOMO_SIZE, (int(v) for v in tomo_size_px), strict=True)))
        row[L.TILT_IMAGE_SIZE[0]], row[L.TILT_IMAGE_SIZE[1]] = ts.image_size
        rows.append(row)

    frame = pd.DataFrame(rows)
    leading = [L.TOMO_NAME, L.TILT_SERIES_STAR]
    frame = frame[leading + [c for c in frame.columns if c not in leading]]
    written.insert(0, write_blocks({L.GLOBAL_BLOCK: frame}, path))
    return written
