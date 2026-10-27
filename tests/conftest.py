"""Shared fixtures. Tests run on CPU in double precision, for determinism."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from parapick.tomography import TiltSeries, TomogramSet  # noqa: E402
from parapick.workflows import SimulatedDataset, SimulationConfig, simulate  # noqa: E402

from helpers.factories import make_tilt_series  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def tilt_series() -> TiltSeries:
    return make_tilt_series()


@pytest.fixture
def tomograms() -> TomogramSet:
    return TomogramSet((make_tilt_series("TS_01"), make_tilt_series("TS_02", tilt_axis=-80.0)))


@pytest.fixture(scope="session")
def small_dataset() -> SimulatedDataset:
    """Two tilt-series with noisy picks and false positives."""
    return simulate(
        SimulationConfig(
            n_tilt_series=2,
            n_particles=25,
            tilt_step=4.0,
            image_size=(384, 384),
            pick_noise=10.0,
            recall=0.7,
            false_positives=8,
            seed=7,
        )
    )
