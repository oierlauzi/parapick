"""Unstacking tilt-series for SPA pickers, end to end."""

from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile

import parapick as pp
from parapick.cli.main import main
from parapick.formats.relion5 import Relion5Format
from parapick.formats.topaz import TopazFormat
from parapick.workflows import run_unstack, write_dataset

RELION = Relion5Format()


def _section_value(ts_index: int, tilt: int) -> float:
    return 100.0 * ts_index + tilt


@pytest.fixture(scope="module")
def dataset():
    return pp.simulate(
        pp.SimulationConfig(
            n_tilt_series=2, n_particles=12, tilt_step=5.0, image_size=(192, 160),
            false_positives=0, recall=1.0, pick_noise=0.0, seed=5,
        )
    )  # fmt: skip


@pytest.fixture
def stacked(tmp_path, dataset) -> Path:
    """A RELION 5 tomogram set in the IMOD-import layout: one stack per tilt-series."""
    root = tmp_path / "project"
    write_dataset(dataset, root)
    tomograms_path = root / "tomograms.star"
    table = starfile.read(tomograms_path)
    stacks = []
    for s, ts in enumerate(dataset.tomograms):
        width, height = ts.image_size
        data = np.stack(
            [np.full((height, width), _section_value(s, i), np.float32) for i in range(ts.n_tilts)]
        )
        stack = root / "stacks" / f"{ts.name}.st"
        stack.parent.mkdir(exist_ok=True)
        with mrcfile.new(stack) as mrc:
            mrc.set_data(data)
        stacks.append(f"{stack}:mrc")
        ts_path = root / "tilt_series" / f"{ts.name}.star"
        tilts = starfile.read(ts_path).drop(columns=["rlnMicrographName"])
        starfile.write({ts.name: tilts}, ts_path)
    table["rlnTomoTiltSeriesName"] = stacks
    table = table.drop(columns=["ppkTiltImageSizeX", "ppkTiltImageSizeY"])
    starfile.write({"global": table}, tomograms_path)
    return tomograms_path


def test_relion_reader_handles_stacks(tmp_path, stacked, dataset):
    tomograms = RELION.read_tomograms(stacked, pp.formats.TomogramsReadOptions())
    ts = tomograms["TS_01"]
    assert ts.image_size == (192, 160)  # from the stack header
    assert ts.tilt_ids[0].startswith("1@") and ts.tilt_ids[0].endswith("TS_01.st:mrc")
    assert ts.images == ts.tilt_ids
    np.testing.assert_allclose(ts.translations, dataset.tomograms["TS_01"].translations, atol=1e-4)
    # Writing it back keeps the stack layout.
    RELION.write_tomograms(tomograms, tmp_path / "copy" / "tomograms.star")
    table = starfile.read(tmp_path / "copy" / "tomograms.star")
    stacks = stacked.parent / "stacks"
    assert list(table["rlnTomoTiltSeriesName"]) == [f"{stacks / n}.st:mrc" for n in tomograms.names]
    assert list(table["rlnTomoFrameCount"]) == [ts.n_tilts for ts in tomograms]


def test_unstack_writes_single_sections_and_geometry(tmp_path, stacked, dataset):
    out = tmp_path / "unstacked"
    tomograms, written = run_unstack(stacked, out)
    images = sorted((out / "micrographs").glob("*.mrc"))
    assert len(images) == sum(ts.n_tilts for ts in dataset.tomograms)
    assert out / "tomograms.star" in written
    for s, ts in enumerate(tomograms):
        for i in (0, ts.n_tilts - 1):
            path = Path(ts.tilt_ids[i])
            assert path.name == f"{ts.name}_{i + 1:03d}.mrc"
            with mrcfile.open(path) as mrc:
                assert mrc.header.nz == 1  # what crYOLO requires
                assert float(mrc.data.mean()) == _section_value(s, i)
                assert float(mrc.voxel_size.x) == pytest.approx(ts.pixel_size)

    # The written geometry is identical and drops the stack (layout B), so RELION reads the
    # per-tilt images.
    table = starfile.read(out / "tomograms.star")
    assert "rlnTomoTiltSeriesName" not in table.columns
    reread = RELION.read_tomograms(out / "tomograms.star", pp.formats.TomogramsReadOptions())
    for original, new in zip(dataset.tomograms, reread, strict=True):
        np.testing.assert_allclose(new.projections, original.projections, atol=1e-6)
        np.testing.assert_allclose(new.translations, original.translations, atol=1e-4)
        assert new.images == new.tilt_ids

    # Re-running keeps existing files unless asked to overwrite.
    _, again = run_unstack(stacked, out)
    assert not any(p.suffix == ".mrc" for p in again)


def test_picking_on_unstacked_images_localizes(tmp_path, stacked, dataset):
    out = tmp_path / "unstacked"
    assert main(["unstack", "--tomograms", str(stacked), "-o", str(out), "-q"]) == 0
    tomograms = pp.read_tomograms(out / "tomograms.star")
    # Simulate a 2D picker run on micrographs/*.mrc: Topaz names images by stem.
    picks = pp.project_particles(dataset.ground_truth, tomograms)
    TopazFormat().write_picks(picks, tmp_path / "topaz.txt")
    assert set(pd.read_csv(tmp_path / "topaz.txt", sep="\t")["image_name"]).issubset(
        {Path(p).stem for p in (out / "micrographs").iterdir()}
    )
    converted = tmp_path / "picks.star"
    assert main(["convert-picks", str(tmp_path / "topaz.txt"), "-o", str(converted),
                 "--tomograms", str(out / "tomograms.star"), "-q"]) == 0  # fmt: skip
    assert main(["localize", "--tomograms", str(out / "tomograms.star"), "--picks",
                 str(converted), "-o", str(tmp_path / "loc"), "-q"]) == 0  # fmt: skip
    particles = pp.read_particles(tmp_path / "loc" / "particles.star")
    selected = pp.select_particles(particles, pp.Selection(min_score=0.5))
    metrics = pp.evaluate(selected, dataset.ground_truth, cutoff=10.0).overall
    assert metrics.f1 == pytest.approx(1.0)


def test_link_single_images(tmp_path, dataset):
    root = tmp_path / "project"
    write_dataset(dataset, root)
    for ts in dataset.tomograms:
        for i, tilt in enumerate(ts.tilt_ids):
            pp_path = root / tilt
            pp_path.parent.mkdir(parents=True, exist_ok=True)
            with mrcfile.new(pp_path) as mrc:
                mrc.set_data(np.full((160, 192), float(i), np.float32))
    out = tmp_path / "linked"
    tomograms, _ = run_unstack(root / "tomograms.star", out, link=True)
    first = Path(tomograms["TS_02"].tilt_ids[3])
    assert first.is_symlink()
    with mrcfile.open(first) as mrc:
        assert float(mrc.data.mean()) == 3.0
