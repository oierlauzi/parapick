"""Thin helpers around :mod:`starfile` shared by STAR-based formats."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import starfile

__all__ = [
    "STRING_LABELS",
    "find_loop",
    "peek_labels",
    "read_blocks",
    "require_columns",
    "resolve_relative_path",
    "write_blocks",
]

STRING_LABELS: tuple[str, ...] = (
    "rlnTomoName",
    "rlnMicrographName",
    "rlnMicrographMovieName",
    "rlnMicrographCoordinates",
    "rlnTomoTiltSeriesStarFile",
    "rlnTomoTiltSeriesName",
    "rlnTomoParticlesFile",
    "rlnTomoTomogramsFile",
    "rlnImageName",
)
"""Labels that must never be parsed as numbers (e.g. tomogram names like ``001``)."""

_LABEL = re.compile(r"^\s*_([A-Za-z][\w]*)", re.MULTILINE)


def read_blocks(path: Path) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Read every data block of a STAR file, keeping name-like labels as strings."""
    blocks = starfile.read(path, always_dict=True, parse_as_string=list(STRING_LABELS))
    return {str(name): block for name, block in blocks.items()}


def find_loop(
    blocks: Mapping[str, pd.DataFrame | dict[str, Any]],
    required: Iterable[str],
    *,
    preferred: str | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """Return the ``preferred`` block, or else the first block holding ``required`` labels.

    Simple (key-value) blocks are promoted to single-row tables.
    """
    required = tuple(required)

    def as_frame(block: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
        return block if isinstance(block, pd.DataFrame) else pd.DataFrame([block])

    if preferred is not None and preferred in blocks:
        return as_frame(blocks[preferred])
    for block in blocks.values():
        frame = as_frame(block)
        if all(label in frame.columns for label in required):
            return frame
    where = f" in {path}" if path else ""
    raise ValueError(f"No STAR block{where} contains the labels {', '.join(required)}")


def require_columns(frame: pd.DataFrame, labels: Iterable[str], path: Path | None = None) -> None:
    """Raise a helpful error if ``frame`` lacks any of ``labels``."""
    missing = [label for label in labels if label not in frame.columns]
    if missing:
        where = f" in {path}" if path else ""
        raise ValueError(f"Missing STAR label(s){where}: {', '.join(missing)}")


def write_blocks(blocks: Mapping[str, pd.DataFrame | dict[str, Any]], path: Path) -> Path:
    """Write data blocks to ``path``, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    starfile.write(dict(blocks), path, float_format="%.6f")
    return path


def peek_labels(path: Path, max_bytes: int = 1 << 16) -> set[str]:
    """Labels declared in the first ``max_bytes`` of a STAR file (cheap, for sniffing)."""
    if not path.is_file():
        return set()
    with path.open("r", errors="replace") as handle:
        head = handle.read(max_bytes)
    if "data_" not in head:
        return set()
    return set(_LABEL.findall(head))


def resolve_relative_path(name: str, anchor: Path) -> Path:
    """Resolve a path stored in a STAR file.

    RELION stores paths relative to the project directory. We try the path as given (absolute
    or relative to the working directory), then relative to ``anchor``'s directory and each
    of its parents. If nothing exists, the path is returned unchanged.
    """
    candidate = Path(name)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    for base in (anchor.parent, *anchor.parent.parents):
        if (base / candidate).exists():
            return base / candidate
    return candidate
