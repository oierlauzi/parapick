"""Registry of format plug-ins.

Built-in formats are registered on first use. Third-party packages can contribute formats by
declaring an entry point in the ``parapick.formats`` group that points to a format class
(instantiated without arguments) or instance::

    [project.entry-points."parapick.formats"]
    imod = "parapick_imod:ImodFormat"
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from enum import Enum
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, TypeVar, cast

from parapick.formats.base import (
    Format,
    ParticlesReader,
    ParticlesWriter,
    PicksReader,
    PicksWriter,
    TomogramsReader,
    TomogramsWriter,
)

__all__ = [
    "AUTO",
    "ENTRY_POINT_GROUP",
    "FormatRegistry",
    "Role",
    "default_registry",
]

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "parapick.formats"
AUTO = "auto"

T = TypeVar("T")


class Role(Enum):
    """What a format can do."""

    TOMOGRAMS_READER = ("read tilt-series geometry", TomogramsReader)
    TOMOGRAMS_WRITER = ("write tilt-series geometry", TomogramsWriter)
    PICKS_READER = ("read 2D picks", PicksReader)
    PICKS_WRITER = ("write 2D picks", PicksWriter)
    PARTICLES_READER = ("read 3D particles", ParticlesReader)
    PARTICLES_WRITER = ("write 3D particles", ParticlesWriter)

    def __init__(self, label: str, protocol: type) -> None:
        self.label = label
        self.protocol = protocol

    def supported_by(self, fmt: object) -> bool:
        """Whether ``fmt`` implements this role."""
        return isinstance(fmt, self.protocol)


class FormatRegistry:
    """A collection of formats addressable by name and role."""

    def __init__(
        self,
        formats: Iterable[Format] = (),
        *,
        loaders: Iterable[Callable[[FormatRegistry], None]] = (),
    ) -> None:
        self._formats: dict[str, Format] = {}
        self._loaders = list(loaders)
        for fmt in formats:
            self.register(fmt)

    def _ensure_loaded(self) -> None:
        loaders, self._loaders = self._loaders, []
        for loader in loaders:
            loader(self)

    def register(self, fmt: Format | type[Any], *, replace: bool = False) -> Format:
        """Register a format instance (or a class, instantiated without arguments)."""
        instance = cast("Format", fmt() if isinstance(fmt, type) else fmt)
        if not isinstance(instance, Format):
            raise TypeError(f"{instance!r} does not implement the Format interface")
        if not any(role.supported_by(instance) for role in Role):
            raise TypeError(f"Format {instance.name!r} does not implement any role")
        if instance.name in self._formats and not replace:
            raise ValueError(f"A format named {instance.name!r} is already registered")
        self._formats[instance.name] = instance
        return instance

    def load_entry_points(self, group: str = ENTRY_POINT_GROUP) -> None:
        """Register every format advertised through ``group``. Broken plug-ins are skipped."""
        for entry_point in entry_points(group=group):
            try:
                self.register(entry_point.load())
            except Exception:  # pragma: no cover - depends on third-party code
                logger.exception("Could not load format plug-in %r", entry_point.name)

    def __contains__(self, name: object) -> bool:
        self._ensure_loaded()
        return name in self._formats

    def __iter__(self) -> Any:
        self._ensure_loaded()
        return iter(self._formats.values())

    def get(self, name: str) -> Format:
        """Return the format called ``name``."""
        self._ensure_loaded()
        try:
            return self._formats[name]
        except KeyError:
            known = ", ".join(sorted(self._formats))
            raise KeyError(f"Unknown format {name!r}. Known formats: {known}") from None

    def roles(self, fmt: Format) -> list[Role]:
        """Roles implemented by ``fmt``."""
        return [role for role in Role if role.supported_by(fmt)]

    def names(self, role: Role) -> list[str]:
        """Names of the formats that implement ``role``."""
        self._ensure_loaded()
        return [name for name, fmt in self._formats.items() if role.supported_by(fmt)]

    def resolve(self, role: Role, name: str = AUTO, path: Path | None = None) -> Format:
        """Pick a format for ``role``: by ``name``, or (``auto``) by sniffing ``path``.

        Auto-detection tries every candidate's ``sniff``, then falls back to the file suffix.
        """
        self._ensure_loaded()
        if name != AUTO:
            fmt = self.get(name)
            if not role.supported_by(fmt):
                raise ValueError(f"Format {name!r} cannot {role.label}")
            return fmt
        candidates = [fmt for fmt in self._formats.values() if role.supported_by(fmt)]
        if path is None:
            raise ValueError(f"Cannot auto-detect a format to {role.label} without a path")
        if path.exists():
            for fmt in candidates:
                try:
                    if fmt.sniff(path):
                        return fmt
                except Exception:
                    logger.debug("Sniffer of %s failed on %s", fmt.name, path, exc_info=True)
        for fmt in candidates:
            if path.suffix.lower() in fmt.suffixes:
                return fmt
        options = ", ".join(f.name for f in candidates)
        raise ValueError(
            f"Could not detect the format of {path} to {role.label}; choose one of: {options}"
        )

    # Typed conveniences ------------------------------------------------------------------

    def tomograms_reader(self, name: str = AUTO, path: Path | None = None) -> TomogramsReader:
        """Resolve a :class:`TomogramsReader`."""
        return cast("TomogramsReader", self.resolve(Role.TOMOGRAMS_READER, name, path))

    def tomograms_writer(self, name: str = AUTO, path: Path | None = None) -> TomogramsWriter:
        """Resolve a :class:`TomogramsWriter`."""
        return cast("TomogramsWriter", self.resolve(Role.TOMOGRAMS_WRITER, name, path))

    def picks_reader(self, name: str = AUTO, path: Path | None = None) -> PicksReader:
        """Resolve a :class:`PicksReader`."""
        return cast("PicksReader", self.resolve(Role.PICKS_READER, name, path))

    def picks_writer(self, name: str = AUTO, path: Path | None = None) -> PicksWriter:
        """Resolve a :class:`PicksWriter`."""
        return cast("PicksWriter", self.resolve(Role.PICKS_WRITER, name, path))

    def particles_reader(self, name: str = AUTO, path: Path | None = None) -> ParticlesReader:
        """Resolve a :class:`ParticlesReader`."""
        return cast("ParticlesReader", self.resolve(Role.PARTICLES_READER, name, path))

    def particles_writer(self, name: str = AUTO, path: Path | None = None) -> ParticlesWriter:
        """Resolve a :class:`ParticlesWriter`."""
        return cast("ParticlesWriter", self.resolve(Role.PARTICLES_WRITER, name, path))


def _register_builtins(registry: FormatRegistry) -> None:
    from parapick.formats.relion5 import Relion5Format
    from parapick.formats.relion_coords import RelionCoordinatesFormat
    from parapick.formats.topaz import TopazFormat

    for fmt in (Relion5Format, RelionCoordinatesFormat, TopazFormat):
        registry.register(fmt)


def _register_plugins(registry: FormatRegistry) -> None:
    registry.load_entry_points()


_DEFAULT: FormatRegistry | None = None


def default_registry() -> FormatRegistry:
    """The process-wide registry with built-in and installed plug-in formats."""
    global _DEFAULT  # noqa: PLW0603 - lazily created singleton
    if _DEFAULT is None:
        _DEFAULT = FormatRegistry(loaders=(_register_builtins, _register_plugins))
    return _DEFAULT
