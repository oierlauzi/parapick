"""Image input/output with `vitrio <https://pypi.org/project/vitrio/>`_.

vitrio reads MRC (including IMOD ``.st``/``.ali``), TIFF, EER and DM files, and writes MRC
and TIFF. This implementation uses its synchronous ``vio.read`` and ``vio.write_single``.
Each batch is read straight into one preallocated array and returned as an already
completed :class:`~parapick.formats.images.Pending`.

Moving to asynchronous I/O for throughput only needs this module to change. Batches would be
dispatched with ``vio.read_batch_async(loader, destination, locations)`` and
``vio.write_batch_async(saver, source, locations)``, and the returned ``Completion`` wrapped
as the :class:`~parapick.formats.images.Pending`. The memory layout (one contiguous
``(B, H, W)`` float32 batch) is already the one those calls take.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import vitrio as vio
from numpy.typing import NDArray

from parapick.formats.images import (
    ImageIOError,
    ImageReference,
    Pending,
    completed,
    failed,
)

__all__ = ["VitrioImageIO"]

_VITRIO_ERRORS = (RuntimeError, IndexError, ValueError, OSError)


class VitrioImageIO:
    """:class:`~parapick.formats.images.ImageIO` backed by vitrio (synchronous for now)."""

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
        """Read equally sized images into one ``(B, H, W)`` float32 array."""
        try:
            return completed(self._read(references))
        except ImageIOError as error:
            return failed(error)

    def _read(self, references: Sequence[ImageReference]) -> NDArray[np.float32]:
        descriptors: dict[str, vio.ImageDescriptor] = {}
        volumes: dict[str, NDArray[np.float32]] = {}
        batch: NDArray[np.float32] | None = None
        for slot, reference in enumerate(references):
            path = reference.path
            if path not in descriptors:
                descriptors[path] = self._descriptor(path)
            descriptor = descriptors[path]
            shape = self._planar_shape(descriptor)
            if batch is None:
                batch = np.empty((len(references), *shape), dtype=np.float32)
            elif batch.shape[1:] != shape:
                raise ImageIOError(
                    f"{reference} is {shape[1]}x{shape[0]} but the batch is "
                    f"{batch.shape[2]}x{batch.shape[1]}"
                )
            index = self._section_index(reference, descriptor)
            try:
                if descriptor.core_rank == 2:
                    # A 2D image, or one image of an image stack: read in place.
                    location = (
                        vio.ImageLocation(path)
                        if len(descriptor.extents) == 2
                        else vio.ImageLocation(path, index)
                    )
                    vio.read(location, out=batch[slot])
                else:
                    # A stack flagged as a volume (MRC space group 1, common for tilt-series
                    # written by some tools): sections are not addressable, so read it once.
                    if path not in volumes:
                        volumes[path] = np.asarray(vio.read(path, data_type=np.float32))
                    batch[slot] = volumes[path][index]
            except _VITRIO_ERRORS as error:
                raise ImageIOError(f"Cannot read {reference}: {error}") from error
        return batch if batch is not None else np.empty((0, 0, 0), dtype=np.float32)

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
        """Write each image as a single-image file (format chosen by its extension)."""
        if images.ndim != 3 or len(images) != len(paths):
            return failed(ValueError("write_batch expects (B, H, W) images and B paths"))
        try:
            for image, path in zip(images, paths, strict=True):
                path.parent.mkdir(parents=True, exist_ok=True)
                vio.write_single(np.ascontiguousarray(image, dtype=np.float32), str(path))
        except _VITRIO_ERRORS as error:
            return failed(ImageIOError(f"Cannot write {path}: {error}"))
        return completed(None)
