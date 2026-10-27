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
from parapick.formats.images import ImageIOError, failed
from parapick.formats.relion5 import Relion5Format
from parapick.formats.topaz import TopazFormat
from parapick.formats.vitrio_images import VitrioImageIO
from parapick.workflows import run_unstack, unstack, write_dataset

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


@pytest.fixture(params=["image-stack", "volume-flagged"])
def stacked(request, tmp_path, dataset) -> Path:
    """A RELION 5 tomogram set in the IMOD-import layout: one stack per tilt-series.

    Stacks are written both as MRC2014 image stacks (space group 0, as IMOD does) and flagged
    as volumes (space group 1, as some tools do).
    """
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
            if request.param == "image-stack":
                mrc.set_image_stack()
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


class _RecordingIO:
    """Delegates to vitrio and records the order of the calls; can fail one write."""

    def __init__(self, fail_write: int | None = None, fail_finish: str | None = None) -> None:
        self.inner = VitrioImageIO(workers=2)
        self.calls: list[str] = []
        self.fail_write = fail_write
        self.fail_finish = fail_finish
        self.writes = 0

    def image_size(self, reference):
        return self.inner.image_size(reference)

    def read_batch(self, references):
        self.calls.append("read")
        return self.inner.read_batch(references)

    def write_batch(self, images, paths):
        self.calls.append("write")
        self.writes += 1
        if self.writes == self.fail_write:
            return failed(ImageIOError("disk full"))
        return self.inner.write_batch(images, paths)

    def finish(self, path):
        self.calls.append("finish")
        finishing = self.inner.finish(path)
        if path.name == self.fail_finish:
            finishing.result()
            return failed(ImageIOError("flush failed"))
        return finishing


def test_unstack_pipeline_keeps_batches_in_flight(tmp_path, stacked):
    tomograms = RELION.read_tomograms(stacked, pp.formats.TomogramsReadOptions())
    io = _RecordingIO()
    _, written = unstack(tomograms, tmp_path / "out", anchor=stacked, io=io,
                         batch_size=2, in_flight=3)  # fmt: skip
    n_images = sum(ts.n_tilts for ts in tomograms)
    assert len(written) == n_images
    # Three batches are read before the first is written, and three are written before the
    # first is finished.
    assert io.calls[:4] == ["read", "read", "read", "write"]
    first_finish = io.calls.index("finish")
    assert io.calls[:first_finish].count("write") == 3
    assert io.calls.count("finish") == n_images
    for path in written:
        with mrcfile.open(path) as mrc:
            assert mrc.header.nz == 1


def test_failed_unstack_deletes_incomplete_files_and_resumes(tmp_path, stacked):
    tomograms = RELION.read_tomograms(stacked, pp.formats.TomogramsReadOptions())
    out = tmp_path / "out"
    with pytest.raises(ImageIOError, match="disk full"):
        unstack(tomograms, out, anchor=stacked, io=_RecordingIO(fail_write=3),
                batch_size=2, in_flight=2)  # fmt: skip
    kept = sorted((out / "micrographs").glob("*.mrc"))
    assert 0 < len(kept) < sum(ts.n_tilts for ts in tomograms)
    for path in kept:  # every file left behind is complete
        with mrcfile.open(path) as mrc:
            assert mrc.data.shape == (160, 192)
    names = [p.name for p in kept]
    assert "TS_01_005.mrc" not in names and "TS_01_006.mrc" not in names  # the failed batch

    unstacked, written = unstack(tomograms, out, anchor=stacked)
    assert {p.name for p in written}.isdisjoint(names)
    for s, ts in enumerate(unstacked):
        for i, name in enumerate(ts.tilt_ids):
            with mrcfile.open(name) as mrc:
                assert float(mrc.data.mean()) == _section_value(s, i)


def test_failed_finish_deletes_only_that_file(tmp_path, stacked):
    tomograms = RELION.read_tomograms(stacked, pp.formats.TomogramsReadOptions())
    out = tmp_path / "out"
    io = _RecordingIO(fail_finish="TS_01_005.mrc")
    with pytest.raises(ImageIOError, match="flush failed"):
        unstack(tomograms, out, anchor=stacked, io=io, batch_size=2, in_flight=2)
    names = {p.name for p in (out / "micrographs").glob("*.mrc")}
    assert "TS_01_005.mrc" not in names
    assert {"TS_01_001.mrc", "TS_01_004.mrc", "TS_01_006.mrc"} <= names
    # Nothing is left declared on the writer: the same paths can be written again.
    io.fail_finish = None
    _, written = unstack(tomograms, out, anchor=stacked, io=io, overwrite=True)
    assert len(written) == sum(ts.n_tilts for ts in tomograms)
