"""Header-only access to image dimensions."""

from __future__ import annotations

from pathlib import Path

import mrcfile

__all__ = ["read_image_size", "strip_stack_index"]


def strip_stack_index(name: str) -> str:
    """Remove a RELION ``index@`` prefix from an image reference."""
    head, sep, tail = name.partition("@")
    return tail if sep and head.isdigit() else name


def read_image_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` in pixels of an MRC image or stack, reading only its header."""
    with mrcfile.open(path, mode="r", header_only=True, permissive=True) as mrc:
        header = mrc.header
        return int(header.nx), int(header.ny)
