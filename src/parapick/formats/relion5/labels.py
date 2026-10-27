"""STAR labels used by the RELION 5 formats.

Labels prefixed with ``ppk`` are ParaPick-specific. RELION keeps unknown labels when it reads
and writes STAR files, so they are safe to carry along.
"""

from __future__ import annotations

# -- tomograms.star (data_global) -------------------------------------------------------- #
TOMO_NAME = "rlnTomoName"
TILT_SERIES_STAR = "rlnTomoTiltSeriesStarFile"
TILT_SERIES_PIXEL_SIZE = "rlnTomoTiltSeriesPixelSize"
TILT_SERIES_NAME = "rlnTomoTiltSeriesName"
"""Stack holding all tilts (IMOD-import layout); takes precedence over per-tilt images."""
FRAME_COUNT = "rlnTomoFrameCount"
TOMO_SIZE = ("rlnTomoSizeX", "rlnTomoSizeY", "rlnTomoSizeZ")
TOMO_HAND = "rlnTomoHand"
VOLTAGE = "rlnVoltage"
SPHERICAL_ABERRATION = "rlnSphericalAberration"
AMPLITUDE_CONTRAST = "rlnAmplitudeContrast"
DEFORMATION_GRID = ("rlnTomoDeformationGridSizeX", "rlnTomoDeformationGridSizeY")

# -- per tilt-series star ---------------------------------------------------------------- #
MICROGRAPH_NAME = "rlnMicrographName"
X_TILT = "rlnTomoXTilt"
Y_TILT = "rlnTomoYTilt"
Z_ROT = "rlnTomoZRot"
X_SHIFT_ANGST = "rlnTomoXShiftAngst"
Y_SHIFT_ANGST = "rlnTomoYShiftAngst"
NOMINAL_TILT = "rlnTomoNominalStageTiltAngle"

# -- 2D picks (SPA coordinate convention) ------------------------------------------------ #
COORDINATE_X = "rlnCoordinateX"
COORDINATE_Y = "rlnCoordinateY"
COORDINATE_Z = "rlnCoordinateZ"
AUTOPICK_FOM = "rlnAutopickFigureOfMerit"
MICROGRAPH_PIXEL_SIZE = "rlnMicrographPixelSize"
MICROGRAPH_COORDINATES = "rlnMicrographCoordinates"

# -- 3D particles -------------------------------------------------------------------------- #
CENTERED_COORDINATE = (
    "rlnCenteredCoordinateXAngst",
    "rlnCenteredCoordinateYAngst",
    "rlnCenteredCoordinateZAngst",
)
PARTICLES_FILE = "rlnTomoParticlesFile"
TOMOGRAMS_FILE = "rlnTomoTomogramsFile"

# -- ParaPick-specific (ad hoc) ------------------------------------------------------------ #
TILT_IMAGE_SIZE = ("ppkTiltImageSizeX", "ppkTiltImageSizeY")
"""Tilt-image dimensions in native pixels (avoids opening image headers)."""
EFFECTIVE_COUNT = "ppkEffectiveCount"
"""Effective number of supporting detections ``n_k``."""

# -- block names -------------------------------------------------------------------------- #
GLOBAL_BLOCK = "global"
PICKS_BLOCK = "picks"
PARTICLES_BLOCK = "particles"
OPTIMISATION_SET_BLOCK = "optimisation_set"
