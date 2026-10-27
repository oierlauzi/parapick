"""References to tilt images and minimal MRC input/output.

Tilt images are referenced the way RELION does it:

* ``path.mrc``: a file holding one 2D image;
* ``N@path.mrcs``: section ``N`` (1-based) of a stack;
* ``path.st:mrc``: a RELION *type suffix* forcing the MRC reader for unusual extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mrcfile
import numpy as np
from numpy.typing import NDArray

from parapick.formats.star import resolve_relative_path

__all__ = [
    "MRC_SUFFIXES",
    "ImageReference",
    "read_image_size",
    "read_section",
    "strip_stack_index",
    "write_mrc",
]

MRC_SUFFIXES = frozenset({".mrc", ".mrcs", ".st", ".ali", ".preali", ".map", ".rec"})
"""File extensions read with the MRC reader (IMOD uses MRC for ``.st`` and ``.ali``)."""


@dataclass(frozen=True)
class ImageReference:
    """A 2D image: a whole file, or one section of a stack.

    Parameters
    ----------
    path
        File path, without any RELION type suffix (``:mrc``).
    index
        0-based section within a stack, or ``None`` for the whole file.
    type_suffix
        RELION type suffix (e.g. ``"mrc"``) if one was given; kept for round-tripping.
    """

    path: str
    index: int | None = None
    type_suffix: str | None = None

    @classmethod
    def parse(cls, text: str) -> ImageReference:
        """Parse ``[N@]path[:type]``; ``N`` is 1-based, as in RELION."""
        index = None
        head, sep, tail = text.partition("@")
        if sep and head.isdigit():
            index, text = int(head) - 1, tail
            if index < 0:
                raise ValueError(f"Stack indices are 1-based: {head}@...")
        type_suffix = None
        stem, colon, suffix = text.rpartition(":")
        # A type suffix is short and alphabetic; this keeps Windows drive letters intact.
        if colon and suffix.isalpha() and len(suffix) <= 4 and len(stem) > 1:
            text, type_suffix = stem, suffix
        return cls(text, index, type_suffix)

    def __str__(self) -> str:
        prefix = "" if self.index is None else f"{self.index + 1}@"
        suffix = "" if self.type_suffix is None else f":{self.type_suffix}"
        return f"{prefix}{self.path}{suffix}"

    def resolve(self, anchor: Path | None = None) -> Path:
        """The file on disk, resolved like RELION paths (see ``resolve_relative_path``)."""
        return resolve_relative_path(self.path, anchor) if anchor else Path(self.path)

    def is_mrc(self) -> bool:
        """Whether the MRC reader applies (by extension or ``:mrc`` type suffix)."""
        return self.type_suffix == "mrc" or Path(self.path).suffix.lower() in MRC_SUFFIXES


def strip_stack_index(name: str) -> str:
    """The file path of an image reference (drops ``N@`` and ``:type``)."""
    return ImageReference.parse(name).path


def _require_mrc(reference: ImageReference) -> None:
    if not reference.is_mrc():
        raise ValueError(
            f"Unsupported image format for {reference}: only MRC files are supported "
            f"({', '.join(sorted(MRC_SUFFIXES))}, or a ':mrc' type suffix)"
        )


def read_image_size(reference: ImageReference | str, anchor: Path | None = None) -> tuple[int, int]:
    """``(width, height)`` in pixels of an MRC image or stack, reading only its header."""
    ref = ImageReference.parse(reference) if isinstance(reference, str) else reference
    with mrcfile.open(ref.resolve(anchor), mode="r", header_only=True, permissive=True) as mrc:
        return int(mrc.header.nx), int(mrc.header.ny)


def read_section(
    reference: ImageReference | str, anchor: Path | None = None
) -> NDArray[np.float32]:
    """Read one 2D image as ``(height, width)`` float32 without loading whole stacks.

    A reference without an index must point to a file holding a single section.
    """
    ref = ImageReference.parse(reference) if isinstance(reference, str) else reference
    _require_mrc(ref)
    path = ref.resolve(anchor)
    with mrcfile.mmap(path, mode="r", permissive=True) as mrc:
        data = mrc.data
        if data is None:
            raise ValueError(f"Could not read image data from {path}")
        if data.ndim == 2:
            sections = data[None]
        elif data.ndim == 3:
            sections = data
        else:
            raise ValueError(f"{path} has {data.ndim} dimensions; expected an image or a stack")
        if ref.index is None:
            if sections.shape[0] != 1:
                raise ValueError(
                    f"{path} holds {sections.shape[0]} sections; reference one as 'N@{ref.path}'"
                )
            index = 0
        else:
            index = ref.index
        if not 0 <= index < sections.shape[0]:
            raise IndexError(f"Section {index + 1} is out of range for {path}")
        return np.array(sections[index], dtype=np.float32)


def write_mrc(
    path: Path, image: NDArray[np.floating], pixel_size: float, *, overwrite: bool = False
) -> Path:
    """Write a 2D image as a single-section float32 MRC file with the given voxel size."""
    if image.ndim != 2:
        raise ValueError("write_mrc expects a 2D image")
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(np.ascontiguousarray(image, dtype=np.float32))
        mrc.voxel_size = pixel_size
    return path
