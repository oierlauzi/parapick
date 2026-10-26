"""Topaz particle coordinate tables (``topaz extract`` output)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from parapick.tomography import PickTable
from parapick.tomography.matching import tilt_key_variants

__all__ = ["TopazFormat"]

_COLUMNS = ("image_name", "x_coord", "y_coord")
_SCORE = "score"


class TopazFormat:
    """Tab-separated ``image_name x_coord y_coord [score]`` in pixels of the picked images."""

    name = "topaz"
    description = "Topaz coordinate table (image_name, x_coord, y_coord, score)"
    suffixes = (".txt", ".tsv")

    def sniff(self, path: Path) -> bool:
        """True if the header line has Topaz's column names."""
        if not path.is_file():
            return False
        with path.open("r", errors="replace") as handle:
            header = handle.readline().split()
        return all(column in header for column in _COLUMNS)

    def read_picks(self, path: Path) -> PickTable:
        """Read a Topaz table. Image names are matched to tilts by stem."""
        table = pd.read_csv(path, sep="\t", dtype={"image_name": str})
        missing = [c for c in _COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"{path} is missing Topaz column(s): {', '.join(missing)}")
        return PickTable(
            tilt_ids=table["image_name"].to_numpy(dtype=object),
            coordinates=table[["x_coord", "y_coord"]].to_numpy(dtype=np.float64),
            scores=table[_SCORE].to_numpy(dtype=np.float64) if _SCORE in table.columns else None,
        )

    def write_picks(self, picks: PickTable, path: Path) -> list[Path]:
        """Write a Topaz table. Image names are the stems of the tilt identifiers."""
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(
            {
                "image_name": [tilt_key_variants(str(t))[2] for t in picks.tilt_ids],
                "x_coord": np.rint(picks.coordinates[:, 0]).astype(int),
                "y_coord": np.rint(picks.coordinates[:, 1]).astype(int),
            }
        )
        if picks.scores is not None:
            table[_SCORE] = picks.scores
        table.to_csv(path, sep="\t", index=False)
        return [path]
