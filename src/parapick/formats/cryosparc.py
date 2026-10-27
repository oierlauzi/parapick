"""cryoSPARC particle files (``.cs``) from its pickers (blob, template, Topaz, deep picker...).

A ``.cs`` file is a NumPy structured array. Particle locations are stored as fractions of the
micrograph (``location/center_x_frac``, ``location/center_y_frac``) together with
``location/micrograph_shape`` (``[height, width]``) and ``location/micrograph_path``.
Following pyem's ``csparc2star`` defaults (cryoSPARC's origin is bottom-left):

    x = center_x_frac × width,   y = (1 − center_y_frac) × height

This holds for micrographs imported into cryoSPARC, e.g. the output of ``parapick unstack``.
If micrographs were flipped elsewhere, add ``--flip-y`` to ``localize``/``convert-picks``.

cryoSPARC prefixes imported file names with a numeric uid (``<uid>_TS_01_001.mrc``). It is
removed, so the names match the tilts. Read only.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePath

import numpy as np

from parapick.tomography import PickTable

__all__ = ["CryosparcFormat"]

_MAGIC = b"\x93NUMPY"
_REQUIRED = (
    "location/micrograph_path",
    "location/center_x_frac",
    "location/center_y_frac",
    "location/micrograph_shape",
)
_SCORE_FIELDS = ("pick_stats/ncc_score",)
_PIXEL_SIZE = "location/micrograph_psize_A"
_UID_PREFIX = re.compile(r"^\d{10,}_")


def _strip_uid(path: str) -> str:
    name = PurePath(path).name
    return _UID_PREFIX.sub("", name)


class CryosparcFormat:
    """cryoSPARC ``.cs`` particle files with pick locations."""

    name = "cryosparc"
    description = "cryoSPARC .cs particle files (picker output with location/* fields)"
    suffixes = (".cs",)

    def sniff(self, path: Path) -> bool:
        """A ``.cs`` file in NumPy format."""
        if not (path.is_file() and path.suffix.lower() in self.suffixes):
            return False
        with path.open("rb") as handle:
            return handle.read(len(_MAGIC)) == _MAGIC

    def read_picks(self, path: Path) -> PickTable:
        """Read pick locations in micrograph pixels (RELION frame)."""
        data = np.load(path, allow_pickle=False)
        names = data.dtype.names or ()
        missing = [f for f in _REQUIRED if f not in names]
        if missing:
            raise ValueError(
                f"{path} has no particle locations (missing {', '.join(missing)}); "
                "export the picker's particles output"
            )
        shape = np.asarray(data["location/micrograph_shape"], dtype=np.float64).reshape(-1, 2)
        height, width = shape[:, 0], shape[:, 1]
        x = np.asarray(data["location/center_x_frac"], dtype=np.float64) * width
        y = (1.0 - np.asarray(data["location/center_y_frac"], dtype=np.float64)) * height
        paths = [
            _strip_uid(p.decode() if isinstance(p, bytes) else str(p))
            for p in data["location/micrograph_path"]
        ]
        score_field = next((f for f in _SCORE_FIELDS if f in names), None)
        pixel_size = None
        if _PIXEL_SIZE in names:
            sizes = np.unique(np.asarray(data[_PIXEL_SIZE], dtype=np.float64))
            sizes = sizes[sizes > 0]
            if len(sizes) > 1:
                raise ValueError(f"{path}: micrographs have different pixel sizes {sizes}")
            pixel_size = float(sizes[0]) if len(sizes) else None
        return PickTable(
            tilt_ids=np.array(paths, dtype=object),
            coordinates=np.stack([x, y], axis=-1),
            scores=None if score_field is None else np.asarray(data[score_field], dtype=np.float64),
            pixel_size=pixel_size,
        )
