"""EMAN box files (``.box``): crYOLO ``EMAN/``, EMAN2 e2boxer, Gautomatch, and others.

Each line is ``x y width height`` in pixels, with ``(x, y)`` the *lower-left corner* of the
box. The particle centre is ``(x + width/2, y + height/2)``, in the same pixel frame as
RELION coordinates: crYOLO writes its RELION STAR files exactly this way
(``CoordsIO.write_star_file``). Extra columns (EMAN1 filament markers) are ignored.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from parapick.formats.per_micrograph import MicrographPicks, PerMicrographPicksFormat

__all__ = ["EmanBoxFormat"]


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


class EmanBoxFormat(PerMicrographPicksFormat):
    """EMAN box files, one per micrograph (read only: writing needs a box size)."""

    name = "eman-box"
    description = "EMAN box files (x y w h, lower-left corner), e.g. crYOLO EMAN/, e2boxer"
    suffixes = (".box",)

    def sniff_file(self, path: Path) -> bool:
        """At least four numeric columns on the first non-empty line (or an empty file)."""
        with path.open("r", errors="replace") as handle:
            for line in handle:
                tokens = line.split()
                if tokens:
                    return len(tokens) >= 4 and all(_is_number(t) for t in tokens[:4])
        return True

    def read_file(self, path: Path) -> MicrographPicks:
        """Read one box file and return box centres."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # empty file
            data = np.loadtxt(path, ndmin=2, usecols=(0, 1, 2, 3))
        if data.size == 0:
            return MicrographPicks(np.empty((0, 2)))
        return MicrographPicks(data[:, :2] + data[:, 2:4] / 2.0)
