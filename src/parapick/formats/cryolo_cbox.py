"""crYOLO CBOX files (``.cbox``).

CBOX files keep *every* detection with its confidence, including those below the picking
threshold. That makes them the best crYOLO output for ParaPick, which benefits from
permissive picks: threshold them later with ``--min-pick-score``.

Two layouts exist, and both are read:

* crYOLO ≥ 1.8: a STAR file whose ``cryolo`` block has ``_CoordinateX``/``_CoordinateY`` (the
  *lower-left corner* of the box), ``_Width``, ``_Height`` and ``_Confidence``
  (``CoordsIO.write_cbox_file``);
* older crYOLO: whitespace-separated ``x y w h confidence [est_w est_h]``, with the same
  corner convention.

Centres are ``corner + size / 2``, in RELION's pixel frame (see :mod:`.eman_box`).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import starfile

from parapick.formats.per_micrograph import MicrographPicks, PerMicrographPicksFormat

__all__ = ["CryoloCboxFormat"]

_STAR_COLUMNS = ("CoordinateX", "CoordinateY", "Width", "Height", "Confidence")


def _is_star(path: Path) -> bool:
    with path.open("r", errors="replace") as handle:
        return "data_" in handle.read(4096)


class CryoloCboxFormat(PerMicrographPicksFormat):
    """crYOLO CBOX files, one per micrograph, with a confidence per detection."""

    name = "cryolo-cbox"
    description = "crYOLO CBOX files (all detections with confidence; STAR or legacy text)"
    suffixes = (".cbox",)

    def read_file(self, path: Path) -> MicrographPicks:
        """Read one CBOX file and return box centres and confidences."""
        if _is_star(path):
            return self._read_star(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # empty file
            data = np.loadtxt(path, ndmin=2, usecols=(0, 1, 2, 3, 4))
        if data.size == 0:
            return MicrographPicks(np.empty((0, 2)), np.empty(0))
        return MicrographPicks(data[:, :2] + data[:, 2:4] / 2.0, data[:, 4])

    @staticmethod
    def _read_star(path: Path) -> MicrographPicks:
        blocks = starfile.read(path, always_dict=True)
        table = blocks.get("cryolo")
        if not isinstance(table, pd.DataFrame):
            table = next(
                (
                    b
                    for b in blocks.values()
                    if isinstance(b, pd.DataFrame) and "CoordinateX" in b.columns
                ),
                None,
            )
        if table is None or table.empty:
            return MicrographPicks(np.empty((0, 2)), np.empty(0))
        missing = [c for c in _STAR_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"{path} is missing CBOX column(s): {', '.join(missing)}")
        corners = table[["CoordinateX", "CoordinateY"]].to_numpy(dtype=np.float64)
        sizes = table[["Width", "Height"]].to_numpy(dtype=np.float64)
        return MicrographPicks(
            corners + sizes / 2.0, table["Confidence"].to_numpy(dtype=np.float64)
        )
