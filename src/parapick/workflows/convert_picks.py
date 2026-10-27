"""Convert 2D picks between formats, optionally resolving them against tilt-series."""

from __future__ import annotations

import logging
from pathlib import Path

from parapick.formats import AUTO, FormatRegistry, TomogramsReadOptions, default_registry
from parapick.tomography import PickTable, annotate_picks

__all__ = ["convert_picks"]

logger = logging.getLogger(__name__)


def convert_picks(
    input_path: Path,
    output_path: Path,
    *,
    input_format: str = AUTO,
    output_format: str = "relion5",
    tomograms_path: Path | None = None,
    tomograms_format: str = AUTO,
    read_options: TomogramsReadOptions | None = None,
    pixel_size: float | None = None,
    registry: FormatRegistry | None = None,
) -> tuple[PickTable, list[Path]]:
    """Read picks in one format and write them in another.

    When ``tomograms_path`` is given, picks are matched to its tilts: tilt-series names are
    added, tilt identifiers become the canonical micrograph names and unmatched picks are
    dropped. This turns e.g. a Topaz table into a self-describing picks STAR file.
    ``pixel_size`` records the pixel size of the picked images in the output.
    """
    registry = registry or default_registry()
    reader = registry.picks_reader(input_format, input_path)
    picks = reader.read_picks(input_path)
    logger.info("Read %d picks from %s (%s)", len(picks), input_path, reader.name)
    if tomograms_path is not None:
        tomograms = registry.tomograms_reader(tomograms_format, tomograms_path).read_tomograms(
            tomograms_path, read_options or TomogramsReadOptions()
        )
        picks, report = annotate_picks(picks, tomograms)
        logger.info("Matched %d picks (%d unmatched)", report.matched, report.unmatched)
    if pixel_size is not None:
        picks = PickTable(
            tilt_ids=picks.tilt_ids,
            coordinates=picks.coordinates,
            tilt_series=picks.tilt_series,
            scores=picks.scores,
            pixel_size=pixel_size,
        )
    written = registry.picks_writer(output_format).write_picks(picks, output_path)
    return picks, written
