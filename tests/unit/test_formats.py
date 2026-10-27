"""Format plug-ins: RELION 5, RELION coordinates, Topaz, and the registry."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile

from parapick.formats import (
    FormatRegistry,
    Role,
    TomogramsReadOptions,
    WriteContext,
    default_registry,
)
from parapick.formats import (
    registry as registry_module,
)
from parapick.formats.relion5 import Relion5Format
from parapick.formats.relion_coords import RelionCoordinatesFormat
from parapick.formats.topaz import TopazFormat
from parapick.tomography import Particles, PickTable, TomogramSet

RELION = Relion5Format()


@pytest.fixture
def picks() -> PickTable:
    return PickTable(
        tilt_ids=np.array(["tilts/TS_01_000.mrc", "tilts/TS_01_000.mrc", "tilts/TS_02_001.mrc"]),
        coordinates=np.array([[10.5, 20.25], [30.0, 40.0], [50.0, 60.0]]),
        tilt_series=np.array(["TS_01", "TS_01", "TS_02"], dtype=object),
        scores=np.array([0.9, 0.5, 0.1]),
    )


# -- RELION 5 tomograms --------------------------------------------------------------------- #


def test_relion_tomograms_round_trip(tmp_path, tomograms):
    path = tmp_path / "tomograms.star"
    written = RELION.write_tomograms(tomograms, path)
    assert path in written and (tmp_path / "tilt_series" / "TS_02.star").exists()
    assert RELION.sniff(path)
    back = RELION.read_tomograms(path, TomogramsReadOptions())
    assert back.names == tomograms.names
    for a, b in zip(tomograms, back, strict=True):
        assert a.tilt_ids == b.tilt_ids
        np.testing.assert_allclose(b.projections, a.projections, atol=1e-6)
        np.testing.assert_allclose(b.translations, a.translations, atol=1e-4)
        assert b.image_size == a.image_size
        assert b.volume_size == pytest.approx(a.volume_size)
    # Writing again from what was read keeps the RELION metadata.
    RELION.write_tomograms(back, tmp_path / "again" / "tomograms.star")
    table = starfile.read(tmp_path / "again" / "tomograms.star")
    assert {"rlnVoltage", "rlnTomoSizeZ", "ppkTiltImageSizeX"} <= set(table.columns)


def test_relion_tomograms_image_size_from_mrc(tmp_path, tomograms):
    path = tmp_path / "tomograms.star"
    RELION.write_tomograms(tomograms.subset(["TS_01"]), path)
    table = starfile.read(path)
    table = table.drop(columns=["ppkTiltImageSizeX", "ppkTiltImageSizeY"])
    starfile.write({"global": table}, path)
    # Without image headers nor labels the size cannot be determined.
    with pytest.raises(ValueError, match="tilt-image size"):
        RELION.read_tomograms(path, TomogramsReadOptions())
    # An explicit size works.
    explicit = RELION.read_tomograms(path, TomogramsReadOptions(tilt_image_size=(512, 512)))
    assert explicit["TS_01"].image_size == (512, 512)
    # Images relative to the project directory (here, the STAR file's directory) are found.
    image = tmp_path / tomograms["TS_01"].tilt_ids[0]
    image.parent.mkdir(parents=True)
    with mrcfile.new(image) as mrc:
        mrc.set_data(np.zeros((512, 512), dtype=np.float32))
    from_header = RELION.read_tomograms(path, TomogramsReadOptions())
    np.testing.assert_allclose(
        from_header["TS_01"].translations, tomograms["TS_01"].translations, atol=1e-4
    )


def test_relion_tomograms_requires_labels(tmp_path):
    path = tmp_path / "bad.star"
    starfile.write({"global": pd.DataFrame({"rlnTomoName": ["a"]})}, path)
    with pytest.raises(ValueError, match="rlnTomoTiltSeriesStarFile"):
        RELION.read_tomograms(path, TomogramsReadOptions())


# -- RELION 5 picks and particles ----------------------------------------------------------- #


def test_relion_picks_round_trip(tmp_path, picks):
    path = tmp_path / "picks.star"
    RELION.write_picks(PickTable(**{**picks.__dict__, "pixel_size": 20.0}), path)
    assert RELION.sniff(path)
    back = RELION.read_picks(path)
    np.testing.assert_allclose(back.coordinates, picks.coordinates)
    np.testing.assert_allclose(back.scores, picks.scores)
    assert list(back.tilt_series) == list(picks.tilt_series)
    assert back.pixel_size == 20.0


def test_relion_picks_without_tomo_name(tmp_path):
    path = tmp_path / "picks.star"
    frame = pd.DataFrame(
        {"rlnMicrographName": ["a.mrc"], "rlnCoordinateX": [1.0], "rlnCoordinateY": [2.0]}
    )
    starfile.write({"picks": frame}, path)
    back = RELION.read_picks(path)
    assert back.tilt_series[0] is None and back.scores is None


def test_relion_particles_round_trip(tmp_path):
    particles = Particles(
        tilt_series=np.array(["TS_01", "TS_02"], dtype=object),
        coordinates=np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]]),
        scores=np.array([0.8, 0.4]),
        support=np.array([32.0, 16.0]),
    )
    path = tmp_path / "out" / "particles.star"
    written = RELION.write_particles(
        particles, path, WriteContext(tomograms_path=Path("tomograms.star"))
    )
    optimisation_set = tmp_path / "out" / "optimisation_set.star"
    assert written == [path, optimisation_set]
    back = RELION.read_particles(path)
    np.testing.assert_allclose(back.coordinates, particles.coordinates)
    np.testing.assert_allclose(back.support, particles.support)
    # The optimisation set can be read as a particles file too.
    via_set = RELION.read_particles(optimisation_set)
    np.testing.assert_allclose(via_set.scores, particles.scores)


def test_relion_legacy_particles(tmp_path, tomograms):
    path = tmp_path / "legacy.star"
    frame = pd.DataFrame(
        {
            "rlnTomoName": ["TS_01"],
            "rlnCoordinateX": [256.0],
            "rlnCoordinateY": [256.0],
            "rlnCoordinateZ": [51.0],
        }
    )
    starfile.write({"particles": frame}, path)
    with pytest.raises(ValueError, match="tomogram set"):
        RELION.read_particles(path)
    particles = RELION.read_particles(path, tomograms)
    np.testing.assert_allclose(particles.coordinates, [[0.0, 0.0, 10.0]])


# -- RELION coordinates and Topaz ----------------------------------------------------------- #


def test_relion_coords_round_trip(tmp_path, picks):
    fmt = RelionCoordinatesFormat()
    index = tmp_path / "autopick.star"
    written = fmt.write_picks(picks, index)
    assert len(written) == 3  # index + 2 micrographs
    assert fmt.sniff(index) and not RELION.sniff(index)
    back = fmt.read_picks(index)
    np.testing.assert_allclose(back.coordinates, picks.coordinates)
    assert list(back.tilt_series) == list(picks.tilt_series)
    assert list(back.tilt_ids) == list(picks.tilt_ids)
    # A directory of coordinate files: tilt identifiers are the file stems.
    directory = tmp_path / "coordinates"
    assert fmt.sniff(directory)
    from_dir = fmt.read_picks(directory)
    assert len(from_dir) == 3
    assert set(from_dir.tilt_ids) == {"TS_01_000", "TS_02_001"}
    # A single coordinate file.
    single = fmt.read_picks(sorted(directory.glob("*.star"))[0])
    assert len(single) == 2


def test_topaz_round_trip(tmp_path, picks):
    fmt = TopazFormat()
    path = tmp_path / "picks.txt"
    fmt.write_picks(picks, path)
    assert fmt.sniff(path)
    back = fmt.read_picks(path)
    assert list(back.tilt_ids) == ["TS_01_000", "TS_01_000", "TS_02_001"]
    np.testing.assert_allclose(back.coordinates, np.rint(picks.coordinates))
    np.testing.assert_allclose(back.scores, picks.scores)
    bad = tmp_path / "bad.txt"
    bad.write_text("image_name\tx\n")
    with pytest.raises(ValueError, match="x_coord"):
        fmt.read_picks(bad)


# -- Registry ------------------------------------------------------------------------------- #


def test_default_registry_resolution(tmp_path, picks, tomograms):
    registry = default_registry()
    assert {"relion5", "relion-coords", "topaz"} <= {f.name for f in registry}
    assert registry.names(Role.TOMOGRAMS_READER) == ["relion5"]
    RELION.write_tomograms(tomograms, tmp_path / "tomograms.star")
    TopazFormat().write_picks(picks, tmp_path / "picks.tsv")
    assert registry.tomograms_reader(path=tmp_path / "tomograms.star").name == "relion5"
    assert registry.picks_reader(path=tmp_path / "picks.tsv").name == "topaz"
    assert registry.picks_writer("relion-coords").name == "relion-coords"
    with pytest.raises(ValueError, match="cannot read tilt-series"):
        registry.tomograms_reader("topaz")
    with pytest.raises(KeyError, match="Unknown format"):
        registry.get("nope")
    with pytest.raises(ValueError, match="without a path"):
        registry.picks_reader()
    with pytest.raises(ValueError, match="Could not detect"):
        registry.picks_reader(path=tmp_path / "unknown.xyz")


class _DummyFormat:
    name = "dummy"
    description = "A third-party picks format"
    suffixes = (".dummy",)

    def sniff(self, path: Path) -> bool:
        return path.suffix == ".dummy"

    def read_picks(self, path: Path) -> PickTable:
        return PickTable(tilt_ids=np.array(["a"]), coordinates=np.zeros((1, 2)))


def test_registry_plugins(monkeypatch):
    registry = FormatRegistry()
    entry_point = EntryPoint(name="dummy", value=f"{__name__}:_DummyFormat", group="x")
    monkeypatch.setattr(registry_module, "entry_points", lambda group: [entry_point])
    registry.load_entry_points()
    assert registry.picks_reader(path=Path("file.dummy")).name == "dummy"
    assert registry.roles(registry.get("dummy")) == [Role.PICKS_READER]
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_DummyFormat)
    registry.register(_DummyFormat(), replace=True)
    with pytest.raises(TypeError):
        registry.register(object())


def test_tomogram_set_metadata_is_preserved(tmp_path, tomograms):
    RELION.write_tomograms(tomograms, tmp_path / "tomograms.star")
    back = RELION.read_tomograms(tmp_path / "tomograms.star", TomogramsReadOptions())
    assert isinstance(back, TomogramSet)
    assert "relion5.tilts" in back["TS_01"].metadata
