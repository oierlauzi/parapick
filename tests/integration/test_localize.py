"""End-to-end localisation on simulated data through the Python API."""

from __future__ import annotations

import json

import numpy as np
import pytest

import parapick as pp
from parapick.gmm import UniformBoxInitializer
from parapick.workflows import run_localize, write_dataset


def test_localize_noisy_picks(small_dataset):
    result = pp.localize(small_dataset.tomograms, small_dataset.picks)
    assert result.match_report is not None and result.match_report.unmatched == 0
    assert all(r.fit.converged for r in result.results)
    particles = result.particles(pp.Selection(min_score=0.3))
    metrics = pp.evaluate(particles, small_dataset.ground_truth, cutoff=50.0).overall
    assert metrics.f1 > 0.9
    assert metrics.rmse < 15.0
    report = result.report()
    assert [t["tilt_series"] for t in report["tilt_series"]] == ["TS_01", "TS_02"]
    assert 0 < report["tilt_series"][0]["background_fraction"] < 1


def test_plain_em_on_clean_picks():
    """All robustness components disabled (ablation): clean picks are still recovered."""
    dataset = pp.simulate(
        pp.SimulationConfig(n_tilt_series=1, n_particles=20, recall=0.8, false_positives=0, seed=2)
    )
    config = pp.LocalizeConfig(background_fraction=None, annealing=0.0, merge_distance=0.0)
    fitter = pp.EMFitter(config, initializer=UniformBoxInitializer())
    result = pp.localize(dataset.tomograms, dataset.picks, config=config, fitter=fitter)
    particles = result.particles(pp.Selection(min_score=0.3))
    assert pp.evaluate(particles, dataset.ground_truth, cutoff=50.0).overall.f1 > 0.9


def test_projected_ground_truth_is_recovered(small_dataset):
    picks = pp.project_particles(small_dataset.ground_truth, small_dataset.tomograms)
    result = pp.localize(small_dataset.tomograms, picks)
    particles = result.particles(pp.Selection(min_score=0.5))
    metrics = pp.evaluate(particles, small_dataset.ground_truth, cutoff=5.0).overall
    assert metrics.f1 == pytest.approx(1.0)
    assert metrics.rmse < 0.5


def test_binned_picks_with_pick_threshold(small_dataset):
    binned = pp.PickTable(
        tilt_ids=small_dataset.picks.tilt_ids,
        coordinates=small_dataset.picks.coordinates / 4.0,
        tilt_series=small_dataset.picks.tilt_series,
        scores=np.ones(len(small_dataset.picks)),
        pixel_size=40.0,
    )
    result = pp.localize(small_dataset.tomograms, binned, min_pick_score=0.5)
    particles = result.particles(pp.Selection(min_score=0.3))
    assert pp.evaluate(particles, small_dataset.ground_truth, cutoff=50.0).overall.f1 > 0.9


def test_file_based_run(tmp_path, small_dataset):
    write_dataset(small_dataset, tmp_path / "sim")
    result, written = run_localize(
        tmp_path / "sim" / "tomograms.star",
        tmp_path / "sim" / "picks.star",
        tmp_path / "out",
        tilt_series=["TS_02"],
        selection=pp.Selection(min_score=0.3),
    )
    assert [r.name for r in result.results] == ["TS_02"]
    names = {p.name for p in written}
    assert names == {"particles.star", "optimisation_set.star", "localize_report.json"}
    particles = pp.read_particles(tmp_path / "out" / "particles.star")
    assert set(particles.names) == {"TS_02"}
    report = json.loads((tmp_path / "out" / "localize_report.json").read_text())
    assert report["config"]["alpha"] == pytest.approx(0.5)
