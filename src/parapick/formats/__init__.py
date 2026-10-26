"""Readers and writers for tilt-series geometry, 2D picks and 3D particles.

Formats are plug-ins: see :mod:`parapick.formats.base` for the interfaces and
:mod:`parapick.formats.registry` for registration (including third-party entry points).
"""

from parapick.formats.base import (
    Format,
    ParticlesReader,
    ParticlesWriter,
    PicksReader,
    PicksWriter,
    TomogramsReader,
    TomogramsReadOptions,
    TomogramsWriter,
    WriteContext,
)
from parapick.formats.registry import (
    AUTO,
    ENTRY_POINT_GROUP,
    FormatRegistry,
    Role,
    default_registry,
)

__all__ = [
    "AUTO",
    "ENTRY_POINT_GROUP",
    "Format",
    "FormatRegistry",
    "ParticlesReader",
    "ParticlesWriter",
    "PicksReader",
    "PicksWriter",
    "Role",
    "TomogramsReadOptions",
    "TomogramsReader",
    "TomogramsWriter",
    "WriteContext",
    "default_registry",
]
