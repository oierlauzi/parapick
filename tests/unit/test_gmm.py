"""Sinkhorn, EM step parity with the NumPy reference, padding, post-processing, fitter."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp

from parapick.gmm import (
    EMFitter,
    LiftedPicksInitializer,
    LocalizeConfig,
    UniformBoxInitializer,
    merge_duplicates,
)
from parapick.gmm import em as jem
from parapick.gmm.padding import bucket, fine_bucket, pad_detections
from parapick.gmm.sinkhorn import capped_sinkhorn
from parapick.tomography import xyz_euler_to_matrix

from helpers import numpy_em


def _problem(rng, n_particles=12, n_tilts=9, noise=5.0, recall=0.8, false_positives=2):
    truth = rng.uniform([-800, -800, -150], [800, 800, 150], (n_particles, 3))
    projections = xyz_euler_to_matrix(0.0, np.linspace(-60, 60, n_tilts), 85.0)[:, :2]
    detections = []
    for p in projections:
        y = truth @ p.T + rng.normal(0, noise, (n_particles, 2))
        y = y[rng.random(n_particles) < recall]
        detections.append(np.concatenate([y, rng.uniform(-900, 900, (false_positives, 2))]))
    return truth, projections, detections


# -- Sinkhorn ------------------------------------------------------------------------------ #


def test_sinkhorn_marginals(rng):
    # Feasible problem: 17 valid detections, 30 components with capacity one each.
    log_kernel = jnp.asarray(rng.normal(0, 3, (20, 30)))
    mask = jnp.asarray(np.arange(20) < 17)
    r = np.asarray(capped_sinkhorn(log_kernel, mask, 500))
    np.testing.assert_allclose(r[:17].sum(1), 1.0, rtol=1e-10)
    np.testing.assert_array_equal(r[17:], 0.0)
    assert np.all(r.sum(0) <= 1.0 + 1e-6)


def test_sinkhorn_uncapped_column_and_pruned_components(rng):
    log_kernel = np.full((30, 4), -1.0)
    log_kernel[:, 1] = -np.inf  # pruned component
    capped = jnp.asarray([True, True, True, False])
    r = np.asarray(capped_sinkhorn(jnp.asarray(log_kernel), jnp.ones(30, dtype=bool), 200, capped))
    np.testing.assert_allclose(r.sum(1), 1.0)
    np.testing.assert_array_equal(r[:, 1], 0.0)
    assert r[:, 0].sum() <= 1 + 1e-6 and r[:, 2].sum() <= 1 + 1e-6
    np.testing.assert_allclose(r[:, 3].sum(), 30 - r[:, 0].sum() - r[:, 2].sum())


def test_sinkhorn_matches_log_domain_reference(rng):
    log_kernel = rng.normal(0, 4, (15, 10))
    expected = numpy_em.sinkhorn_log(log_kernel, 16)
    actual = capped_sinkhorn(jnp.asarray(log_kernel), jnp.ones(15, dtype=bool), 16)
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-9, atol=1e-12)


# -- EM step ------------------------------------------------------------------------------- #


def test_em_step_matches_numpy_reference(rng):
    _, projections, detections = _problem(rng)
    n_components = 20
    means = rng.uniform([-800, -800, -150], [800, 800, 150], (n_components, 3))
    weights = np.full(n_components, 1 / n_components)
    sigma2 = np.full(len(detections), 60.0**2)
    alpha, reg = 0.5, 1e-6

    expected = numpy_em.em_step(
        detections, projections, means, weights, sigma2,
        alpha=alpha, regularization=reg, n_sinkhorn=16,
    )  # fmt: skip

    padded = pad_detections(detections, projections)
    observations = jem.Observations(
        detections=jnp.asarray(padded.detections),
        mask=jnp.asarray(padded.mask),
        projections=jnp.asarray(padded.projections),
        log_area=jnp.asarray(np.log(padded.areas)),
    )
    state = jem.EMState(
        means=jnp.asarray(means),
        log_weights=jnp.log(jnp.asarray(weights)),
        log_background=jnp.asarray(-jnp.inf),
        sigma2=jnp.asarray(np.r_[sigma2, np.ones(len(padded.areas) - len(sigma2))]),
        sigma2_floor=jnp.asarray(0.0),
        support=jnp.zeros(n_components),
        log_likelihood=jnp.asarray(0.0),
        delta=jnp.asarray(jnp.inf),
        iteration=jnp.asarray(0),
        converged=jnp.asarray(False),
    )
    hyper = jem.Hyperparameters(
        alpha=jnp.asarray(alpha),
        tolerance=jnp.asarray(0.0),
        regularization=jnp.asarray(reg),
        min_sigma2=jnp.asarray(0.0),
        annealing=jnp.asarray(0.0),
    )
    new = jem.em_step(state, observations, hyper, 16)
    exp_means, exp_weights, exp_sigma2, exp_support = expected
    np.testing.assert_allclose(np.asarray(new.means), exp_means, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(np.exp(np.asarray(new.log_weights)), exp_weights, atol=1e-12)
    np.testing.assert_allclose(np.asarray(new.sigma2)[: len(sigma2)], exp_sigma2, rtol=1e-9)
    np.testing.assert_allclose(np.asarray(new.support), exp_support, rtol=1e-9, atol=1e-12)
    assert int(new.iteration) == 1


def test_log_likelihood_matches_mixture_density(rng):
    _, projections, detections = _problem(rng, n_tilts=3)
    means = rng.normal(0, 300, (5, 3))
    log_weights = np.log(np.full(5, 0.2))
    s2 = 40.0**2
    stats = jem.tilt_statistics(
        jnp.asarray(means), jnp.asarray(log_weights), jnp.asarray(-jnp.inf), jnp.asarray(s2),
        jnp.asarray(0.0), jnp.asarray(projections[0]), jnp.asarray(detections[0]),
        jnp.ones(len(detections[0]), dtype=bool), 8,
    )  # fmt: skip
    d2 = ((detections[0][:, None] - (means @ projections[0].T)[None]) ** 2).sum(-1)
    density = logsumexp(log_weights - d2 / (2 * s2) - np.log(2 * np.pi * s2), axis=1)
    np.testing.assert_allclose(float(stats.log_likelihood), density.sum(), rtol=1e-10)


# -- Padding and post-processing ----------------------------------------------------------- #


def test_buckets():
    assert [bucket(n) for n in (0, 1, 8, 9, 100)] == [8, 8, 8, 16, 128]
    assert fine_bucket(100) == 112
    assert all(fine_bucket(n) >= n for n in range(1, 2000, 37))


def test_pad_detections():
    projections = np.zeros((3, 2, 3))
    detections = [np.ones((5, 2)), np.empty((0, 2)), np.array([[0.0, 0.0], [10.0, 20.0]])]
    padded = pad_detections(detections, projections)
    assert padded.detections.shape == (8, 8, 2)
    assert padded.mask.sum(1).tolist() == [5, 0, 2, 0, 0, 0, 0, 0]
    assert padded.areas[2] == pytest.approx(200.0)
    assert padded.n_tilts == 3
    with pytest.raises(ValueError, match="projection"):
        pad_detections(detections, np.zeros((2, 2, 3)))


def test_merge_duplicates():
    means = np.array([[0.0, 0, 0], [1.0, 0, 0], [100.0, 0, 0]])
    support = np.array([2.0, 6.0, 3.0])
    merged, merged_support = merge_duplicates(means, support, 5.0)
    np.testing.assert_allclose(merged_support, [8.0, 3.0])
    np.testing.assert_allclose(merged[0], [0.75, 0, 0])
    _, same_support = merge_duplicates(means, support, 0.0)
    np.testing.assert_allclose(same_support, [6.0, 3.0, 2.0])


# -- Config -------------------------------------------------------------------------------- #


def test_config_validation_and_resolution():
    with pytest.raises(ValueError, match="alpha"):
        LocalizeConfig(alpha=0)
    with pytest.raises(ValueError, match="background_fraction"):
        LocalizeConfig(background_fraction=1.5)
    with pytest.raises(ValueError, match="annealing"):
        LocalizeConfig(annealing=1.0)
    config = LocalizeConfig(overcomplete_factor=2.0, init_sigma_fraction=0.1)
    assert config.resolve_n_components(30) == 60
    assert LocalizeConfig(n_components=7).resolve_n_components(30) == 7
    assert config.resolve_init_sigma((100.0, 200.0, 50.0)) == pytest.approx(20.0)
    assert config.resolve_merge_distance([3.0, 5.0, 4.0]) == pytest.approx(20.0)
    assert LocalizeConfig(merge_distance=0).resolve_merge_distance([3.0]) == 0


# -- Fitter -------------------------------------------------------------------------------- #


@pytest.mark.parametrize("initializer", [UniformBoxInitializer(), LiftedPicksInitializer()])
def test_initializers_stay_in_box(rng, initializer):
    _, projections, detections = _problem(rng)
    box = np.array([2000.0, 2000.0, 400.0])
    centres = initializer(rng, 50, box, detections, projections)
    assert centres.shape == (50, 3)
    assert np.all(np.abs(centres) <= box / 2 + 1e-9)


@pytest.mark.parametrize("background", [None, 0.1])
def test_fitter_recovers_clean_particles(rng, background):
    truth, projections, detections = _problem(
        rng, n_tilts=21, noise=3.0, recall=0.9, false_positives=0
    )
    config = LocalizeConfig(background_fraction=background)
    fit = EMFitter(config).fit(detections, projections, (2000.0, 2000.0, 400.0), seed=3)
    assert fit.converged
    strong = fit.means[fit.scores > 0.5]
    assert len(strong) == len(truth)
    nearest = np.linalg.norm(strong[:, None] - truth[None], axis=-1).min(1)
    assert np.all(nearest < 10.0)
    assert np.all(np.diff(fit.support) <= 0)  # sorted by decreasing support
    np.testing.assert_allclose(fit.scores, fit.support / len(projections))
    assert np.median(fit.sigmas) == pytest.approx(3.0, rel=0.5)


def test_fitter_background_absorbs_false_positives(rng):
    truth, projections, detections = _problem(rng, n_tilts=21, noise=3.0, false_positives=10)
    fit = EMFitter(LocalizeConfig(background_fraction=0.1)).fit(
        detections, projections, (2000.0, 2000.0, 400.0), seed=0
    )
    expected_fraction = 10 / (10 + 0.8 * len(truth))
    assert fit.background_fraction == pytest.approx(expected_fraction, abs=0.1)
    assert np.median(fit.sigmas) < 6.0
    assert (fit.scores > 0.4).sum() == len(truth)

    paper_config = LocalizeConfig(background_fraction=None, annealing=0.0, merge_distance=0.0)
    paper = EMFitter(paper_config, initializer=UniformBoxInitializer()).fit(
        detections, projections, (2000.0, 2000.0, 400.0), seed=0
    )
    assert paper.background_fraction == 0.0
    assert np.median(paper.sigmas) > np.median(fit.sigmas)


def test_fitter_handles_empty_input():
    fit = EMFitter().fit([np.empty((0, 2))] * 3, np.zeros((3, 2, 3)), (1.0, 1.0, 1.0))
    assert fit.n_components == 0
    assert fit.sigmas.shape == (3,)


def test_fitter_is_deterministic(rng):
    _, projections, detections = _problem(rng)
    fitter = EMFitter()
    a = fitter.fit(detections, projections, (2000.0, 2000.0, 400.0), seed=11)
    b = fitter.fit(detections, projections, (2000.0, 2000.0, 400.0), seed=11)
    np.testing.assert_array_equal(a.means, b.means)
