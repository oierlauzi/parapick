"""Localising from each supported SPA pick format, end to end through the CLI."""

from __future__ import annotations

import numpy as np
import pytest

import parapick as pp
from parapick.cli.main import main
from parapick.workflows import write_dataset

from helpers.pick_files import (
    write_cbox_legacy,
    write_cbox_star,
    write_cryosparc,
    write_eman_boxes,
    write_xmipp_pos,
)


@pytest.fixture(scope="module")
def dataset():
    return pp.simulate(
        pp.SimulationConfig(
            n_tilt_series=1, n_particles=12, tilt_step=5.0, image_size=(256, 256),
            false_positives=0, recall=1.0, pick_noise=0.0, seed=11,
        )
    )  # fmt: skip


@pytest.fixture
def project(tmp_path, dataset):
    write_dataset(dataset, tmp_path / "project")
    return tmp_path / "project"


def _picks(dataset):
    """Ground-truth projections, rounded to whole pixels as pickers report them."""
    picks = pp.project_particles(dataset.ground_truth, dataset.tomograms)
    return pp.PickTable(
        tilt_ids=picks.tilt_ids,
        coordinates=np.rint(picks.coordinates),
        tilt_series=picks.tilt_series,
        scores=np.full(len(picks), 0.9),
    )


def _localize_f1(project, picks_path, out, dataset, *extra):
    assert main(["localize", "--tomograms", str(project / "tomograms.star"),
                 "--picks", str(picks_path), "-o", str(out), "-q", *extra]) == 0  # fmt: skip
    particles = pp.select_particles(
        pp.read_particles(out / "particles.star"), pp.Selection(min_score=0.5)
    )
    return pp.evaluate(particles, dataset.ground_truth, cutoff=15.0).overall.f1


@pytest.mark.parametrize(
    "writer", [write_eman_boxes, write_cbox_star, write_cbox_legacy, write_xmipp_pos]
)
def test_per_micrograph_formats(tmp_path, project, dataset, writer):
    directory = writer(_picks(dataset), tmp_path / "picks")
    assert _localize_f1(project, directory, tmp_path / "out", dataset) == pytest.approx(1.0)


def test_cbox_confidence_threshold(tmp_path, project, dataset):
    """CBOX keeps sub-threshold detections; --min-pick-score removes them."""
    picks = _picks(dataset)
    rng = np.random.default_rng(0)
    junk = pp.PickTable(
        tilt_ids=picks.tilt_ids,
        coordinates=rng.uniform(0, 256, size=picks.coordinates.shape),
        tilt_series=picks.tilt_series,
        scores=np.full(len(picks), 0.05),
    )
    directory = write_cbox_star(pp.PickTable.concatenate([picks, junk]), tmp_path / "cbox")
    f1 = _localize_f1(project, directory, tmp_path / "out", dataset, "--min-pick-score", "0.3")
    assert f1 == pytest.approx(1.0)


@pytest.mark.parametrize("flipped", [True, False])
def test_cryosparc(tmp_path, project, dataset, flipped):
    ts = dataset.tomograms["TS_01"]
    path = write_cryosparc(_picks(dataset), tmp_path / "picked.cs", ts.image_size, flip_y=flipped)
    extra = [] if flipped else ["--flip-y"]  # unusual handedness: fix it with --flip-y
    assert _localize_f1(project, path, tmp_path / "out", dataset, *extra) == pytest.approx(1.0)
