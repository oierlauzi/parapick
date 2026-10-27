"""Writers that reproduce the layouts of third-party picking tools, for tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import starfile

from parapick.formats.per_micrograph import group_by_micrograph
from parapick.tomography import PickTable
from parapick.tomography.matching import tilt_key_variants

BOX = 48.0


def _per_micrograph(
    picks: PickTable, directory: Path, suffix: str, write: Callable[..., None]
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for _, tilt_id, rows in group_by_micrograph(picks):
        scores = np.ones(len(rows)) if picks.scores is None else picks.scores[rows]
        write(
            directory / f"{tilt_key_variants(tilt_id)[2]}{suffix}", picks.coordinates[rows], scores
        )
    return directory


def write_eman_boxes(picks: PickTable, directory: Path) -> Path:
    """EMAN1 box files as crYOLO writes them: lower-left corner, width, height."""

    def write(path: Path, centres: np.ndarray, _: np.ndarray) -> None:
        corners = centres - BOX / 2
        np.savetxt(path, np.c_[corners, np.full((len(centres), 2), BOX)], delimiter="\t")

    return _per_micrograph(picks, directory, ".box", write)


def write_cbox_star(picks: PickTable, directory: Path) -> Path:
    """crYOLO >= 1.8 CBOX: STAR with 'global' and 'cryolo' blocks, corner coordinates."""

    def write(path: Path, centres: np.ndarray, scores: np.ndarray) -> None:
        n = len(centres)
        cryolo = pd.DataFrame(
            {
                "CoordinateX": centres[:, 0] - BOX / 2,
                "CoordinateY": centres[:, 1] - BOX / 2,
                "CoordinateZ": np.full(n, np.nan),
                "Width": np.full(n, BOX),
                "Height": np.full(n, BOX),
                "Depth": np.full(n, np.nan),
                "EstWidth": np.full(n, BOX),
                "EstHeight": np.full(n, BOX),
                "Confidence": scores,
                "NumBoxes": np.full(n, np.nan),
                "Angle": np.full(n, np.nan),
            }
        )
        starfile.write(
            {"global": pd.DataFrame({"cbox_format_version": ["1.0"]}), "cryolo": cryolo}, path
        )

    return _per_micrograph(picks, directory, ".cbox", write)


def write_cbox_legacy(picks: PickTable, directory: Path) -> Path:
    """crYOLO < 1.8 CBOX: x y w h confidence est_w est_h (corner coordinates)."""

    def write(path: Path, centres: np.ndarray, scores: np.ndarray) -> None:
        n = len(centres)
        sizes = np.full((n, 2), BOX)
        np.savetxt(path, np.c_[centres - BOX / 2, sizes, scores, sizes], delimiter="\t")

    return _per_micrograph(picks, directory, ".cbox", write)


def write_xmipp_pos(picks: PickTable, directory: Path) -> Path:
    """Xmipp .pos files as Xmipp writes them (header block, particles block)."""

    def write(path: Path, centres: np.ndarray, _: np.ndarray) -> None:
        rows = "\n".join(f"{x:>7.0f} {y:>7.0f}" for x, y in centres)
        path.write_text(
            "# XMIPP_STAR_1 * \n#\ndata_header\nloop_\n _pickingMicrographState\nAuto\n"
            f"data_particles\nloop_\n _xcoor\n _ycoor\n{rows}\n"
        )

    return _per_micrograph(picks, directory, ".pos", write)


def write_cryosparc(
    picks: PickTable,
    path: Path,
    image_size: tuple[int, int],
    *,
    flip_y: bool = True,
    pixel_size: float | None = None,
) -> Path:
    """A cryoSPARC picker .cs file (bottom-left origin, uid-prefixed micrograph paths)."""
    width, height = image_size
    n = len(picks)
    fields = [
        ("uid", "<u8"),
        ("location/micrograph_uid", "<u8"),
        ("location/micrograph_path", "S256"),
        ("location/micrograph_shape", "<u4", (2,)),
        ("location/center_x_frac", "<f4"),
        ("location/center_y_frac", "<f4"),
        ("pick_stats/ncc_score", "<f4"),
    ]
    if pixel_size is not None:
        fields.append(("location/micrograph_psize_A", "<f4"))
    data = np.zeros(n, dtype=fields)
    data["uid"] = np.arange(n)
    names = [f"J3/imported/{10**18 + i}_{Path(str(t)).name}" for i, t in enumerate(picks.tilt_ids)]
    data["location/micrograph_path"] = [name.encode() for name in names]
    data["location/micrograph_shape"] = (height, width)
    data["location/center_x_frac"] = picks.coordinates[:, 0] / width
    y_frac = picks.coordinates[:, 1] / height
    data["location/center_y_frac"] = 1.0 - y_frac if flip_y else y_frac
    data["pick_stats/ncc_score"] = 1.0 if picks.scores is None else picks.scores
    if pixel_size is not None:
        data["location/micrograph_psize_A"] = pixel_size
    with path.open("wb") as handle:
        np.save(handle, data)
    return path
