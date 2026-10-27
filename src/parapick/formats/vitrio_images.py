"""Image input/output with `vitrio <https://pypi.org/project/vitrio/>`_.

vitrio reads MRC (including IMOD ``.st``/``.ali``), TIFF, EER and DM files, and writes MRC
and TIFF. This implementation runs on its asynchronous interface:

* a batch is read with ``vio.read_batch_async`` straight into one preallocated
  ``(B, H, W)`` float32 array;
* a batch is written with ``vio.write_batch_async``, one single-image file per image, after
  declaring each file on a writer provider;
* a written file is finished (flushed and closed) with ``close_async``.

Each returns a vitrio ``Completion``, which is wrapped as a
:class:`~parapick.formats.images.Pending`. Reads and writes share one thread pool.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import vitrio as vio
from numpy.typing import NDArray

from parapick.formats.images import ImageIOError, ImageReference, Pending, completed, failed

__all__ = ["VitrioImageIO"]

T = TypeVar("T")

_VITRIO_ERRORS = (RuntimeError, IndexError, ValueError, OSError)
_FLOAT32 = vio.NumericalType.float32


class _Completions(Generic[T]):
    """A :class:`~parapick.formats.images.Pending` over vitrio completions.

    Its result is ``value`` once every completion is ready. If any failed, the first failure
    is raised as an :class:`~parapick.formats.images.ImageIOError`, after ``on_failure`` has
    run (with every completion ready).
    """

    def __init__(
        self,
        completions: Sequence[vio.Completion],
        value: T,
        context: str,
        on_failure: Callable[[], None] | None = None,
    ) -> None:
        self._completions = list(completions)
        self._value = value
        self._context = context
        self._on_failure = on_failure
        self._error: ImageIOError | None = None
        self._settled = False

    def done(self) -> bool:
        """Whether every completion is ready."""
        return self._settled or all(c.is_ready for c in self._completions)

    def result(self) -> T:
        """Wait for every completion, then return the value or raise the first failure."""
        if not self._settled:
            for completion in self._completions:
                completion.wait()
            try:
                for completion in self._completions:
                    completion.get()
            except _VITRIO_ERRORS as error:
                self._error = ImageIOError(f"{self._context}: {error}")
                self._error.__cause__ = error
                if self._on_failure is not None:
                    self._on_failure()
            self._settled = True
            self._completions = []
        if self._error is not None:
            raise self._error
        return self._value


class VitrioImageIO:
    """:class:`~parapick.formats.images.ImageIO` on vitrio's asynchronous interface.

    Parameters
    ----------
    workers
        Threads that run the reads and writes (default: what the machine reports).
    open_files
        How many input files the reader keeps open between batches.

    The thread pool and the reader and writer are created on first use. Call the methods
    from one thread at a time.
    """

    def __init__(self, workers: int | None = None, open_files: int = 8) -> None:
        self._workers = workers
        self._open_files = open_files
        self._executor: vio.ThreadPoolExecutor | None = None
        self._loader: vio.ImageLoader | None = None
        self._writers: vio.FileImageWriterProvider | None = None
        self._saver: vio.ImageSaver | None = None

    def _pool(self) -> vio.ThreadPoolExecutor:
        if self._executor is None:
            self._executor = vio.ThreadPoolExecutor(self._workers or os.cpu_count() or 1)
        return self._executor

    def _reader(self) -> vio.ImageLoader:
        if self._loader is None:
            self._loader = vio.loader(cache=self._open_files, executor=self._pool())
        return self._loader

    def _writer(self) -> tuple[vio.FileImageWriterProvider, vio.ImageSaver]:
        if self._writers is None or self._saver is None:
            self._writers = vio.writer_provider()
            self._saver = vio.saver(self._writers, executor=self._pool())
        return self._writers, self._saver

    @staticmethod
    def _descriptor(path: str) -> vio.ImageDescriptor:
        try:
            return vio.query_descriptor(path)
        except _VITRIO_ERRORS as error:
            raise ImageIOError(f"Cannot read the header of {path}: {error}") from error

    @staticmethod
    def _planar_shape(descriptor: vio.ImageDescriptor) -> tuple[int, int]:
        extents = descriptor.extents
        if len(extents) < 2 or len(extents) > 3:
            raise ImageIOError(f"Expected an image or a stack of images, got extents {extents}")
        return int(extents[-2]), int(extents[-1])

    def image_size(self, reference: ImageReference) -> tuple[int, int]:
        """``(width, height)`` from the file header."""
        height, width = self._planar_shape(self._descriptor(reference.path))
        return width, height

    def read_batch(self, references: Sequence[ImageReference]) -> Pending[NDArray[np.float32]]:
        """Start reading equally sized images into one ``(B, H, W)`` float32 array.

        The headers are checked before this returns. The pixel data is read in the background.
        """
        try:
            batch, locations = self._plan_read(references)
        except ImageIOError as error:
            return failed(error)
        if not locations:
            return completed(batch)
        context = f"Cannot read {references[0]}" + (
            f" and {len(references) - 1} more" if len(references) > 1 else ""
        )
        # vitrio cannot mix whole files and stack sections in one call: issue one per run.
        completions = []
        try:
            start = 0
            for end in range(1, len(locations) + 1):
                if (
                    end == len(locations)
                    or locations[end].has_index_in_stack != locations[start].has_index_in_stack
                ):
                    completions.append(
                        vio.read_batch_async(self._reader(), batch[start:end], locations[start:end])
                    )
                    start = end
        except _VITRIO_ERRORS as error:
            for completion in completions:
                completion.wait()
            return failed(ImageIOError(f"{context}: {error}"))
        return _Completions(completions, batch, context)

    def _plan_read(
        self, references: Sequence[ImageReference]
    ) -> tuple[NDArray[np.float32], list[vio.ImageLocation]]:
        """The destination array and the vitrio location of each reference (headers only)."""
        descriptors: dict[str, vio.ImageDescriptor] = {}
        locations: list[vio.ImageLocation] = []
        shape: tuple[int, int] | None = None
        for reference in references:
            path = reference.path
            if path not in descriptors:
                descriptors[path] = self._descriptor(path)
            descriptor = descriptors[path]
            planar = self._planar_shape(descriptor)
            if shape is None:
                shape = planar
            elif planar != shape:
                raise ImageIOError(
                    f"{reference} is {planar[1]}x{planar[0]} but the batch is {shape[1]}x{shape[0]}"
                )
            index = self._section_index(reference, descriptor)
            locations.append(
                vio.ImageLocation(path)
                if len(descriptor.extents) == 2
                else vio.ImageLocation(path, index)
            )
        if shape is None:
            return np.empty((0, 0, 0), dtype=np.float32), []
        return np.empty((len(references), *shape), dtype=np.float32), locations

    @staticmethod
    def _section_index(reference: ImageReference, descriptor: vio.ImageDescriptor) -> int:
        extents = descriptor.extents
        sections = int(extents[0]) if len(extents) == 3 else 1
        if reference.index is None:
            if sections != 1:
                raise ImageIOError(
                    f"{reference.path} holds {sections} sections; reference one as "
                    f"'N@{reference.path}'"
                )
            return 0
        if not 0 <= reference.index < sections:
            raise ImageIOError(f"Section {reference.index + 1} is out of range for {reference}")
        return reference.index

    def write_batch(self, images: NDArray[np.float32], paths: Sequence[Path]) -> Pending[None]:
        """Start writing each image as a single-image file (format chosen by its extension).

        Each file must then be finished with :meth:`finish`. If the writes fail, the files are
        finished here instead.
        """
        if images.ndim != 3 or len(images) != len(paths):
            return failed(ValueError("write_batch expects (B, H, W) images and B paths"))
        if not len(paths):
            return failed(ValueError("write_batch expects at least one image"))
        writers, saver = self._writer()
        source = np.ascontiguousarray(images, dtype=np.float32)
        descriptor = vio.ImageDescriptor(source.shape[1:], 2, _FLOAT32)
        keys = [str(path) for path in paths]
        declared: list[str] = []

        def close_all() -> None:
            for key in declared:
                with contextlib.suppress(*_VITRIO_ERRORS):
                    writers.close(key)

        try:
            for path, key in zip(paths, keys, strict=True):
                path.parent.mkdir(parents=True, exist_ok=True)
                writers.declare(key, descriptor)
                declared.append(key)
            writing = vio.write_batch_async(saver, source, [vio.ImageLocation(key) for key in keys])
        except _VITRIO_ERRORS as error:
            close_all()
            culprit = keys[len(declared)] if len(declared) < len(keys) else keys[0]
            return failed(ImageIOError(f"Cannot write {culprit}: {error}"))
        context = f"Cannot write {keys[0]}" + (
            f" and {len(keys) - 1} more" if len(keys) > 1 else ""
        )
        return _Completions([writing], None, context, on_failure=close_all)

    def finish(self, path: Path) -> Pending[None]:
        """Start flushing and closing a file written with :meth:`write_batch`."""
        writers, _ = self._writer()
        try:
            closing = writers.close_async(str(path))
        except _VITRIO_ERRORS as error:
            return failed(ImageIOError(f"Cannot finish {path}: {error}"))
        return _Completions([closing], None, f"Cannot finish {path}")
