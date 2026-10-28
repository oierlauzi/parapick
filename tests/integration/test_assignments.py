"""Assignment of the 2D picks to the localised particles."""

from __future__ import annotations

import numpy as np
import pytest

import parapick as pp
from parapick.formats.relion5.assignments import read_assignments
from parapick.gmm.assignment import _capped_sinkhorn
from parapick.gmm.sinkhorn import capped_sinkhorn
from parapick.workflows import run_localize, write_dataset


def _nearest_truth(particles: pp.Particles, truth: pp.Particles) -> dict[tuple[str, int], int]:
    """Map (tilt-series, particle id) to the index of the nearest ground-truth particle."""
    out = {}
    for name, group in particles.by_tilt_series():
        rows = np.flatnonzero(truth.tilt_series == name)
        for pid, xyz in zip(group.ids, group.coordinates, strict=True):
            d = np.linalg.norm(truth.coordinates[rows] - xyz, axis=1)
            out[(name, int(pid))] = int(rows[np.argmin(d)])
    return out


def test_numpy_sinkhorn_matches_jax(rng):
    log_kernel = rng.normal(size=(7, 5)) * 3
    capped = np.arange(5) < 4
    expected = np.asarray(capped_sinkhorn(log_kernel, np.ones(7, dtype=bool), 16, capped))
    np.testing.assert_allclose(_capped_sinkhorn(log_kernel, capped, 16), expected, atol=1e-6)


def test_projected_picks_are_assigned_to_their_particle(small_dataset):
    truth = small_dataset.ground_truth
    picks = pp.project_particles(truth, small_dataset.tomograms, keep_outside=True)
    # Label every projected pick (tilt-major order) with the particle that produced it.
    n_tilts = {ts.name: ts.n_tilts for ts in small_dataset.tomograms}
    source = np.concatenate(
        [np.tile(np.flatnonzero(truth.tilt_series == name), n_tilts[name]) for name in truth.names]
    )
    assert len(source) == len(picks)

    result = pp.localize(small_dataset.tomograms, picks)
    particles = result.particles(pp.Selection(min_score=0.5))
    assignments = result.assignments(pp.Selection(min_score=0.5))
    nearest = _nearest_truth(particles, truth)

    assert len(assignments) > 0.95 * len(picks)
    assert np.all(assignments.pick_rows >= 0)
    for name, pid, row in zip(
        assignments.tilt_series, assignments.particle_ids, assignments.pick_rows, strict=True
    ):
        assert nearest[(str(name), int(pid))] == source[row]
    # At most one pick per particle and tilt.
    keys = list(
        zip(
            assignments.tilt_series, assignments.particle_ids, assignments.tilt_indices, strict=True
        )
    )
    assert len(keys) == len(set(keys))
    assert np.all(np.abs(assignments.residuals) < 1.0)
    assert np.all(assignments.responsibilities >= 0.5)


def test_assignment_rows_survive_pick_threshold(small_dataset):
    picks = small_dataset.picks
    scores = np.where(np.arange(len(picks)) % 3 == 0, 0.0, 1.0)
    scored = pp.PickTable(
        tilt_ids=picks.tilt_ids,
        coordinates=picks.coordinates,
        tilt_series=picks.tilt_series,
        scores=scores,
    )
    result = pp.localize(small_dataset.tomograms, scored, min_pick_score=0.5)
    rows = result.assignments().pick_rows
    assert len(rows) > 0
    assert np.all(scores[rows] == 1.0)


def test_file_based_assignments(tmp_path, small_dataset):
    write_dataset(small_dataset, tmp_path / "sim")
    selection = pp.Selection(min_score=0.3)
    _, written = run_localize(
        tmp_path / "sim" / "tomograms.star",
        tmp_path / "sim" / "picks.star",
        tmp_path / "out",
        selection=selection,
        write_assignments=True,
    )
    assert "assignments.star" in {p.name for p in written}
    particles = pp.read_particles(tmp_path / "out" / "particles.star")
    assignments = read_assignments(tmp_path / "out" / "assignments.star")
    assert particles.ids is not None
    ids = set(zip(particles.tilt_series, particles.ids.tolist(), strict=True))
    assert set(zip(assignments.tilt_series, assignments.particle_ids.tolist(), strict=True)) <= ids
    # Stored coordinates are the input picks, referenced by their row.
    input_picks = pp.read_picks(tmp_path / "sim" / "picks.star")
    np.testing.assert_allclose(
        assignments.coordinates, input_picks.coordinates[assignments.pick_rows]
    )
    assert list(assignments.tilt_ids) == [
        str(t) for t in input_picks.tilt_ids[assignments.pick_rows]
    ]


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_invalid_threshold(small_dataset, threshold):
    with pytest.raises(ValueError, match="threshold"):
        pp.localize(small_dataset.tomograms, small_dataset.picks, assignment_threshold=threshold)
