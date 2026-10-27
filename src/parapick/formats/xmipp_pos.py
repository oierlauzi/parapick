"""Xmipp/Scipion coordinate files (``.pos``).

Xmipp metadata in STAR syntax (``# XMIPP_STAR_1 *``), with a ``data_header`` block and a
particles block holding ``_xcoor``/``_ycoor``. These are particle *centres* in pixels, in the
same frame as RELION coordinates (Scipion converts between the two without flipping). An
optional ``_cost`` column is the picker's score.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import starfile

from parapick.formats.per_micrograph import (
    MicrographPicks,
    PerMicrographPicksFormat,
    PerMicrographPicksWriter,
)

__all__ = ["XmippPosFormat"]


class XmippPosFormat(PerMicrographPicksWriter, PerMicrographPicksFormat):
    """Xmipp ``.pos`` coordinate files, one per micrograph."""

    name = "xmipp-pos"
    description = "Xmipp/Scipion .pos coordinate files (_xcoor, _ycoor, optional _cost)"
    suffixes = (".pos",)

    def read_file(self, path: Path) -> MicrographPicks:
        """Read the particles block of one ``.pos`` file."""
        blocks = starfile.read(path, always_dict=True)
        table = next(
            (
                b
                for b in blocks.values()
                if isinstance(b, pd.DataFrame) and {"xcoor", "ycoor"} <= set(b.columns)
            ),
            None,
        )
        if table is None:
            return MicrographPicks(np.empty((0, 2)))
        return MicrographPicks(
            table[["xcoor", "ycoor"]].to_numpy(dtype=np.float64),
            table["cost"].to_numpy(dtype=np.float64) if "cost" in table.columns else None,
        )

    def write_file(self, picks: MicrographPicks, path: Path) -> None:
        """Write one ``.pos`` file in Xmipp's layout (integer pixel centres)."""
        lines = [
            "# XMIPP_STAR_1 *",
            "#",
            "data_header",
            "loop_",
            " _pickingMicrographState",
            "Manual",
            "data_particles",
            "loop_",
            " _xcoor",
            " _ycoor",
        ]
        if picks.scores is not None:
            lines.append(" _cost")
        rounded = np.rint(picks.coordinates).astype(int)
        for i, (x, y) in enumerate(rounded):
            row = f"{x:>7d} {y:>7d}"
            if picks.scores is not None:
                row += f" {picks.scores[i]:.6f}"
            lines.append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
