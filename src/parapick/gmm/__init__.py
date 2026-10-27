"""Projected 3D Gaussian-mixture fitting with EM, implemented in JAX."""

from parapick.gmm.config import LocalizeConfig
from parapick.gmm.fitter import EMFitter, Fitter, GMMFit
from parapick.gmm.initialization import (
    INITIALIZERS,
    Initializer,
    LiftedPicksInitializer,
    UniformBoxInitializer,
)
from parapick.gmm.postprocess import merge_duplicates

__all__ = [
    "INITIALIZERS",
    "EMFitter",
    "Fitter",
    "GMMFit",
    "Initializer",
    "LiftedPicksInitializer",
    "LocalizeConfig",
    "UniformBoxInitializer",
    "merge_duplicates",
]
