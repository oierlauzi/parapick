"""2D template matching on the images of tilt-series.

:mod:`~parapick.matching.schedule` derives per-tilt search parameters from the acquisition
(numpy only); :mod:`~parapick.matching.acquisition` infers its inputs from the metadata of a
tilt-series; :mod:`~parapick.matching.plan` plans the searches; and
:mod:`~parapick.matching.leopard` runs them with Leopard-EM, an optional dependency
(``pip install parapick[leopard]``).
"""

from parapick.matching.acquisition import Optics, TiltCtf, TiltSeriesAcquisition, infer_acquisition
from parapick.matching.plan import PlannedTilt, leopard_config, plan_matching, schedule_table
from parapick.matching.schedule import (
    Acquisition,
    ScheduleConstants,
    TiltImage,
    TiltParameters,
    critical_exposure,
    schedule_tilts,
    wavelength,
)

__all__ = [
    "Acquisition",
    "Optics",
    "PlannedTilt",
    "ScheduleConstants",
    "TiltCtf",
    "TiltImage",
    "TiltParameters",
    "TiltSeriesAcquisition",
    "critical_exposure",
    "infer_acquisition",
    "leopard_config",
    "plan_matching",
    "schedule_table",
    "schedule_tilts",
    "wavelength",
]
