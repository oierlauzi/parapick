"""Write every tilt image as its own 2D MRC file, for single-particle pickers.

The pixel size is not recorded in the MRC headers: the pickers read only the image data and
shape, and ParaPick takes the pixel size from the geometry.

Most SPA pickers expect one micrograph per file: crYOLO rejects MRC files with more than one
section, and Topaz in 2D mode treats each file as one micrograph keyed by its stem. Tilt-series
are commonly stored as one stack per tilt-series. :func:`unstack` writes one ``.mrc`` per tilt
and returns the same geometry with its tilts pointing at the new files. Picks made on those
files therefore match the tilts when they are localised with the new geometry.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from parapick.formats import AUTO, FormatRegistry, TomogramsReadOptions, default_registry
from parapick.formats.images import ImageIO, ImageReference, Pending, default_image_io
from parapick.tomography import TomogramSet

__all__ = ["FILES_FINISHING", "MICROGRAPHS_DIRECTORY", "run_unstack", "unstack", "unstacked_name"]

logger = logging.getLogger(__name__)

MICROGRAPHS_DIRECTORY = "micrographs"


def unstacked_name(tilt_series: str, index: int) -> str:
    """File name of tilt ``index`` (0-based) of a tilt-series: ``<TS>_<NNN>.mrc``, 1-based."""
    return f"{tilt_series}_{index + 1:03d}.mrc"


def _single_mrc_file(reference: ImageReference) -> bool:
    return reference.index is None and Path(reference.path).suffix.lower() == ".mrc"


@dataclass(frozen=True)
class _Batch:
    """Images of one tilt-series to copy together (all of the same size)."""

    sources: tuple[ImageReference, ...]
    targets: tuple[Path, ...]


def _batches(copies: list[tuple[ImageReference, Path]], batch_size: int) -> Iterator[_Batch]:
    for start in range(0, len(copies), batch_size):
        chunk = copies[start : start + batch_size]
        yield _Batch(tuple(s for s, _ in chunk), tuple(t for _, t in chunk))


def _replace(target: Path) -> None:
    """Remove an existing output (never write through a symlink into its source)."""
    if target.exists() or target.is_symlink():
        target.unlink()


FILES_FINISHING = 16
"""How many written files are finished (flushed and closed) at once.

Finishing a file waits for the storage, so it runs while the next ones are read and written.
"""


class _Pipeline:
    """Copies batches with reads, writes and file finishes in flight at once.

    Up to ``in_flight`` batches are being read ahead of the one being written, and up to
    ``in_flight`` are being written. A file starts finishing as soon as its batch is written.
    Once a batch has started finishing, the oldest files are waited for until at most
    :data:`FILES_FINISHING` are left finishing.
    """

    def __init__(self, io: ImageIO, in_flight: int) -> None:
        self._io = io
        self._in_flight = in_flight
        self._reads: deque[tuple[_Batch, Pending[NDArray[np.float32]]]] = deque()
        self._writes: deque[tuple[_Batch, Pending[None]]] = deque()
        self._finishing: deque[tuple[Path, Pending[None]]] = deque()
        self._incomplete: dict[Path, None] = {}  # started, not yet finished (in order)
        self.written: list[Path] = []

    def run(self, batches: Iterable[_Batch]) -> list[Path]:
        try:
            for batch in batches:
                self._reads.append((batch, self._io.read_batch(batch.sources)))
                if len(self._reads) == self._in_flight:
                    self._start_write()
            while self._reads:
                self._start_write()
            while self._writes:
                self._complete_write()
            while self._finishing:
                self._complete_finish()
        except BaseException:
            self._abandon()
            raise
        return self.written

    def _start_write(self) -> None:
        """Wait for the oldest read, then start writing its images."""
        batch, reading = self._reads.popleft()
        images = reading.result()
        for target in batch.targets:
            _replace(target)
            self._incomplete[target] = None
        self._writes.append((batch, self._io.write_batch(images, batch.targets)))
        if len(self._writes) == self._in_flight:
            self._complete_write()

    def _complete_write(self) -> None:
        """Wait for the oldest write, then start finishing its files."""
        batch, writing = self._writes.popleft()
        writing.result()
        # Start them all before waiting on any, so a failure never leaves one unfinished.
        self._finishing.extend((target, self._io.finish(target)) for target in batch.targets)
        while len(self._finishing) > FILES_FINISHING:
            self._complete_finish()

    def _complete_finish(self) -> None:
        target, finishing = self._finishing.popleft()
        finishing.result()
        del self._incomplete[target]
        self.written.append(target)

    def _abandon(self) -> None:
        """After a failure: let everything in flight settle, then delete incomplete files.

        Files that were fully written are kept. Deleting the others keeps a re-run, which skips
        existing files, from keeping a broken one.
        """
        for _, reading in self._reads:
            _settle(reading)
        for batch, writing in self._writes:
            if _settle(writing):
                self._finishing.extend((t, self._io.finish(t)) for t in batch.targets)
        for target, finishing in self._finishing:
            if _settle(finishing):
                del self._incomplete[target]
        for target in self._incomplete:
            target.unlink(missing_ok=True)


def _settle(pending: Pending[object]) -> bool:
    """Wait for ``pending``, and say whether it succeeded."""
    try:
        pending.result()
    except Exception:
        return False
    return True


def unstack(
    tomograms: TomogramSet,
    output_directory: Path,
    *,
    anchor: Path | None = None,
    link: bool = False,
    overwrite: bool = False,
    io: ImageIO | None = None,
    batch_size: int = 4,
    in_flight: int = 4,
) -> tuple[TomogramSet, list[Path]]:
    """Write one 2D MRC file per tilt into ``output_directory/micrographs``.

    Parameters
    ----------
    tomograms
        Geometry whose ``images`` reference the tilt images (files or ``N@stack`` sections).
    output_directory
        Destination. Files are named ``<TS>_<NNN>.mrc`` (1-based, in tilt order), so their stems
        are unique across tilt-series.
    anchor
        File that relative image paths are resolved against (e.g. the ``tomograms.star`` they
        were read from), as RELION does.
    link
        Symlink tilts that already are single 2D ``.mrc`` files instead of copying them.
    overwrite
        Replace existing output files (by default they are kept, so runs can resume).
    io
        Image reader/writer (default: :func:`~parapick.formats.images.default_image_io`).
    batch_size
        Images per read/write batch.
    in_flight
        Batches read ahead, and batches written, at once. Reads, writes and file finishes
        overlap. Memory is bounded by about ``2 × in_flight × batch_size`` images.

    Returns
    -------
    TomogramSet, list[Path]
        The same geometry with tilt identifiers and images pointing at the new files, and the
        image files created in this call.
    """
    if batch_size < 1 or in_flight < 1:
        raise ValueError("batch_size and in_flight must be >= 1")
    io = io or default_image_io()
    directory = output_directory / MICROGRAPHS_DIRECTORY
    series, batches, written = [], [], []
    for ts in tomograms:
        names: list[str] = []
        copies: list[tuple[ImageReference, Path]] = []
        for i, image in enumerate(ts.images):
            target = directory / unstacked_name(ts.name, i)
            names.append(str(target))
            if (target.exists() or target.is_symlink()) and not overwrite:
                logger.debug("Keeping existing %s", target)
                continue
            source = ImageReference.parse(image).resolved(anchor)
            if link and _single_mrc_file(source):
                _replace(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(Path(source.path).resolve())
                written.append(target)
            else:
                copies.append((source, target))
        batches += list(_batches(copies, batch_size))
        series.append(dataclasses.replace(ts, tilt_ids=tuple(names), images=tuple(names)))
        logger.info("%s: %d tilt images -> %s", ts.name, ts.n_tilts, directory)
    written += _Pipeline(io, in_flight).run(batches)
    return TomogramSet(tuple(series), tomograms.metadata), written


def run_unstack(
    tomograms_path: Path,
    output_directory: Path,
    *,
    tomograms_format: str = AUTO,
    output_format: str = "relion5",
    read_options: TomogramsReadOptions | None = None,
    link: bool = False,
    overwrite: bool = False,
    io: ImageIO | None = None,
    batch_size: int = 4,
    in_flight: int = 4,
    registry: FormatRegistry | None = None,
) -> tuple[TomogramSet, list[Path]]:
    """File-based :func:`unstack`: also writes the new geometry (e.g. ``tomograms.star``).

    Pick on ``output_directory/micrographs/*.mrc``, then use the written geometry for
    ``convert-picks`` and ``localize``.
    """
    registry = registry or default_registry()
    reader = registry.tomograms_reader(tomograms_format, tomograms_path)
    tomograms = reader.read_tomograms(tomograms_path, read_options or TomogramsReadOptions())
    unstacked, images = unstack(
        tomograms,
        output_directory,
        anchor=tomograms_path,
        link=link,
        overwrite=overwrite,
        io=io,
        batch_size=batch_size,
        in_flight=in_flight,
    )
    writer = registry.tomograms_writer(output_format)
    geometry_path = output_directory / f"tomograms{writer.suffixes[0] if writer.suffixes else ''}"
    written = writer.write_tomograms(unstacked, geometry_path)
    logger.info(
        "Wrote %d tilt images and %s. Pick on %s, then pass --tomograms %s to "
        "convert-picks and localize.",
        len(images),
        geometry_path,
        output_directory / MICROGRAPHS_DIRECTORY,
        geometry_path,
    )
    return unstacked, written + images
