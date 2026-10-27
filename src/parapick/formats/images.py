"""Tilt-image references and the image input/output interface.

Tilt images are referenced the way RELION does it:

* ``path.mrc``: a file holding one 2D image;
* ``N@path.mrcs``: section ``N`` (1-based) of a stack;
* ``path.st:mrc``: a RELION *type suffix*, naming the format of an unusual extension.

Pixel data goes through an :class:`ImageIO`. Its operations return a :class:`Pending` result
instead of blocking, so callers can keep several reads, writes and file finishes in flight and
overlap them. The bundled implementation
(:class:`~parapick.formats.vitrio_images.VitrioImageIO`) runs them on vitrio's asynchronous
interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from parapick.formats.star import resolve_relative_path

__all__ = [
    "ImageIO",
    "ImageIOError",
    "ImageReference",
    "Pending",
    "completed",
    "default_image_io",
    "failed",
    "read_image_size",
    "read_section",
    "strip_stack_index",
]

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")


class ImageIOError(OSError):
    """An image could not be read or written."""


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

    def resolved(self, anchor: Path | None) -> ImageReference:
        """The same reference with its path resolved like RELION paths, relative to ``anchor``."""
        if anchor is None:
            return self
        return replace(self, path=str(resolve_relative_path(self.path, anchor)))


def strip_stack_index(name: str) -> str:
    """The file path of an image reference (drops ``N@`` and ``:type``)."""
    return ImageReference.parse(name).path


@runtime_checkable
class Pending(Protocol[T_co]):
    """The result of an operation that may still be running (e.g. a ``Future``)."""

    def done(self) -> bool:
        """Whether the operation has finished."""
        ...

    def result(self) -> T_co:
        """Wait for the operation and return its result, raising its error if it failed."""
        ...


def completed(value: T) -> Future[T]:
    """A :class:`Pending` that has already finished with ``value``."""
    future: Future[T] = Future()
    future.set_result(value)
    return future


def failed(error: BaseException) -> Future[T]:
    """A :class:`Pending` that has already failed with ``error``."""
    future: Future[T] = Future()
    future.set_exception(error)
    return future


@runtime_checkable
class ImageIO(Protocol):
    """Reads and writes 2D images in batches.

    Operations may return before they finish. Callers wait on the returned :class:`Pending`
    and must keep the arrays they pass unchanged until it is done. A file written with
    :meth:`write_batch` is complete only once :meth:`finish` has finished it.
    """

    def image_size(self, reference: ImageReference) -> tuple[int, int]:
        """``(width, height)`` in pixels of the referenced image, from its header."""
        ...

    def read_batch(self, references: Sequence[ImageReference]) -> Pending[NDArray[np.float32]]:
        """Read equally sized images into a ``(B, height, width)`` float32 array."""
        ...

    def write_batch(self, images: NDArray[np.float32], paths: Sequence[Path]) -> Pending[None]:
        """Write each image of a ``(B, height, width)`` array to its own single-image file.

        Once the writes are done, each file still has to be finished with :meth:`finish`. If
        they fail, the files are left finished (possibly incomplete) and need no
        :meth:`finish`.
        """
        ...

    def finish(self, path: Path) -> Pending[None]:
        """Finish a file written with :meth:`write_batch`: flush it to storage and close it."""
        ...


_DEFAULT_IO: ImageIO | None = None


def default_image_io() -> ImageIO:
    """The process-wide :class:`ImageIO` (vitrio-based)."""
    global _DEFAULT_IO  # noqa: PLW0603 - lazily created singleton
    if _DEFAULT_IO is None:
        from parapick.formats.vitrio_images import VitrioImageIO

        _DEFAULT_IO = VitrioImageIO()
    return _DEFAULT_IO


def read_image_size(
    reference: ImageReference | str, anchor: Path | None = None, *, io: ImageIO | None = None
) -> tuple[int, int]:
    """``(width, height)`` in pixels of an image or stack, reading only its header."""
    ref = ImageReference.parse(reference) if isinstance(reference, str) else reference
    return (io or default_image_io()).image_size(ref.resolved(anchor))


def read_section(
    reference: ImageReference | str, anchor: Path | None = None, *, io: ImageIO | None = None
) -> NDArray[np.float32]:
    """Read one 2D image as a ``(height, width)`` float32 array (blocking)."""
    ref = ImageReference.parse(reference) if isinstance(reference, str) else reference
    batch: NDArray[np.float32] = (
        (io or default_image_io()).read_batch([ref.resolved(anchor)]).result()
    )
    return np.asarray(batch[0], dtype=np.float32)
