"""Matching, selection, evaluation, projection and simulation."""

from __future__ import annotations

import numpy as np
import pytest

from parapick.tomography import (
    Particles,
    PickTable,
    TiltSeries,
    TomogramSet,
    annotate_picks,
    bind_picks,
)
from parapick.tomography.matching import tilt_key_variants
from parapick.workflows import (
    Selection,
    SimulationConfig,
    evaluate,
    project_particles,
    select_particles,
    simulate,
)


def test_tilt_key_variants():
    assert tilt_key_variants("3@dir/TS_01_-21.00.mrcs") == (
        "3@dir/TS_01_-21.00.mrcs",
        "TS_01_-21.00.mrcs",
        "TS_01_-21.00",
    )


def test_bind_picks_matching_rules(tomograms, caplog):
    ts = tomograms["TS_01"]
    picks = PickTable(
        tilt_ids=np.array(
            [
                ts.tilt_ids[0],  # exact
                "elsewhere/" + ts.tilt_ids[1].split("/")[-1],  # by name
                ts.tilt_ids[2].split("/")[-1].replace(".mrc", ""),  # by stem
                "unknown.mrc",
            ],
            dtype=object,
        ),
        coordinates=np.ones((4, 2)),
        tilt_series=np.array(["TS_01", "TS_01", None, "TS_01"], dtype=object),
    )
    bound, report = bind_picks(picks, tomograms, pixel_size=2.0)
    assert report.matched == 3 and report.unmatched == 1
    assert "unknown.mrc" in report.unmatched_examples
    np.testing.assert_array_equal(bound["TS_01"].tilt_indices, [0, 1, 2])
    np.testing.assert_allclose(bound["TS_01"].coordinates, 2.0)
    assert len(bound["TS_02"]) == 0
    per_tilt = bound["TS_01"].per_tilt(ts.n_tilts)
    assert [len(p) for p in per_tilt[:4]] == [1, 1, 1, 0]


def test_bind_picks_ambiguous_and_unknown_series(tomograms):
    first, second = tomograms
    clash = TomogramSet(
        (first, TiltSeries(
            "TS_02", first.tilt_ids, second.projections, second.translations, second.pixel_size
        ))
    )  # fmt: skip
    picks = PickTable(
        tilt_ids=np.array([first.tilt_ids[0]], dtype=object), coordinates=np.zeros((1, 2))
    )
    with pytest.raises(ValueError, match="several tilts"):
        bind_picks(picks, clash)
    # With the tilt-series name the match is unambiguous.
    named = PickTable(**{**picks.__dict__, "tilt_series": np.array(["TS_02"], dtype=object)})
    bound, report = bind_picks(named, clash)
    assert report.matched == 1 and len(bound["TS_02"]) == 1
    unknown = PickTable(**{**picks.__dict__, "tilt_series": np.array(["TS_99"], dtype=object)})
    _, report = bind_picks(unknown, clash)
    assert report.matched == 0


def test_annotate_picks(tomograms):
    ts = tomograms["TS_02"]
    picks = PickTable(
        tilt_ids=np.array([ts.tilt_ids[3].split("/")[-1], "nope"], dtype=object),
        coordinates=np.array([[1.0, 2.0], [3.0, 4.0]]),
        pixel_size=20.0,
    )
    annotated, report = annotate_picks(picks, tomograms)
    assert report.unmatched == 1
    assert list(annotated.tilt_ids) == [ts.tilt_ids[3]]
    assert list(annotated.tilt_series) == ["TS_02"]
    assert annotated.pixel_size == 20.0


def _particles() -> Particles:
    return Particles(
        tilt_series=np.array(["A", "A", "A", "B"], dtype=object),
        coordinates=np.arange(12.0).reshape(4, 3),
        scores=np.array([0.2, 0.9, 0.5, 0.7]),
        support=np.array([2.0, 9.0, 5.0, 7.0]),
    )


def test_selection():
    particles = _particles()
    assert select_particles(particles, Selection()) is particles
    assert len(select_particles(particles, Selection(min_score=0.5))) == 3
    assert len(select_particles(particles, Selection(min_support=6))) == 2
    top = select_particles(particles, Selection(top_n=1))
    np.testing.assert_allclose(top.scores, [0.9, 0.7])
    with pytest.raises(ValueError, match="non-negative"):
        Selection(top_n=-1)
    bare = Particles(tilt_series=np.array(["A"], dtype=object), coordinates=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="scores"):
        select_particles(bare, Selection(min_score=0.1))


def test_evaluate():
    reference = Particles(
        tilt_series=np.array(["A", "A", "B"], dtype=object),
        coordinates=np.array([[0.0, 0, 0], [100.0, 0, 0], [0.0, 0, 0]]),
    )
    predicted = Particles(
        tilt_series=np.array(["A", "A", "A"], dtype=object),
        coordinates=np.array([[3.0, 4, 0], [100.0, 0, 0], [500.0, 0, 0]]),
    )
    result = evaluate(predicted, reference, cutoff=10.0)
    a, b = result.per_tilt_series
    assert (a.true_positives, a.precision, a.recall) == (2, pytest.approx(2 / 3), 1.0)
    assert a.rmse == pytest.approx(np.sqrt(25 / 2))
    assert b.true_positives == 0 and np.isnan(b.precision)
    assert result.overall.f1 == pytest.approx(2 * 2 / 6)
    assert result.as_dict()["overall"]["true_positives"] == 2
    with pytest.raises(ValueError, match="cutoff"):
        evaluate(predicted, reference, cutoff=0)


def test_project_particles(tomograms):
    particles = Particles(
        tilt_series=np.array(["TS_01", "TS_01", "TS_09"], dtype=object),
        coordinates=np.array([[0.0, 0, 0], [1e5, 0, 0], [0.0, 0, 0]]),
        scores=np.array([1.0, 0.5, 0.2]),
    )
    picks = project_particles(particles, tomograms)
    ts = tomograms["TS_01"]
    assert len(picks) == ts.n_tilts  # far particle is outside every image; TS_09 unknown
    np.testing.assert_allclose(picks.coordinates, ts.translations / ts.pixel_size)
    binned = project_particles(particles, tomograms, pixel_size=20.0, keep_outside=True)
    assert len(binned) == 2 * ts.n_tilts and binned.pixel_size == 20.0


def test_simulation_is_deterministic_and_consistent():
    config = SimulationConfig(n_tilt_series=1, n_particles=10, false_positives=0, recall=1.0,
                              pick_noise=0.0, seed=3)  # fmt: skip
    a, b = simulate(config), simulate(config)
    np.testing.assert_array_equal(a.ground_truth.coordinates, b.ground_truth.coordinates)
    ts = a.tomograms["TS_01"]
    # Noise-free picks are exactly the projections of the ground truth.
    bound, _ = bind_picks(a.picks, a.tomograms)
    projected = ts.project(a.ground_truth.coordinates)
    for i, tilt in enumerate(bound["TS_01"].per_tilt(ts.n_tilts)):
        distances = np.linalg.norm(tilt[:, None] - projected[i][None], axis=-1).min(1)
        np.testing.assert_allclose(distances, 0.0, atol=1e-9)
    with pytest.raises(ValueError, match="recall"):
        SimulationConfig(recall=0.0)
