from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import pandas as pd
import pytest

from parapick.formats.relion5 import labels as L
from parapick.matching import (
    Acquisition,
    TiltImage,
    infer_acquisition,
    leopard,
    leopard_config,
    plan_matching,
    schedule_table,
    schedule_tilts,
    wavelength,
)
from parapick.tomography import TiltSeries, TomogramSet, xyz_euler_to_matrix
from parapick.workflows.match_template import collect_picks

from helpers.factories import make_tilt_series

# Reference case of the specification: 70S ribosome, 1.7 Å/px, ±60° in 3° steps,
# dose-symmetric, 133 e/Å² in total, 300 kV.
REFERENCE = Acquisition(
    pixel_size=1.7, image_size=(3838, 3710), voltage=300.0, thickness=2000.0,
    particle_diameter=250.0,
)  # fmt: skip


def _dose_symmetric(step: float = 3.0, limit: float = 60.0) -> list[float]:
    order, n = [0.0], int(limit / step)
    for i in range(1, n + 1):
        order += [i * step, -i * step]
    return order


def _reference_tilts() -> list[TiltImage]:
    order = _dose_symmetric()
    dose = 133.0 / len(order)
    return [TiltImage(angle=a, pre_exposure=i * dose, dose=dose) for i, a in enumerate(order)]


def test_wavelength_at_300kv():
    assert wavelength(300.0) == pytest.approx(0.01969, abs=1e-5)


def test_schedule_reproduces_the_reference_table():
    # Section 8.3 of the specification (acquisition index -> expected values)
    expected = {
        0: (None, 2.5, 1.5, 294, 1468, 0.571),
        1: (None, 2.5, 1.5, 294, 1762, 0.481),
        4: (0.3754, 3.33, 2.0, 500, 2000, 0.303),
        6: (0.2589, 4.0, 2.9, 500, 2000, 0.240),
        12: (0.1536, 4.0, 3.0, 500, 2500, 0.157),
        40: (0.0699, 4.0, 3.0, 500, 8000, 0.056),
    }
    schedule = schedule_tilts(REFERENCE, _reference_tilts())
    for i, (cutoff, theta, psi, step, half, snr) in expected.items():
        p = schedule[i]
        assert p.high_freq_cutoff == cutoff
        assert p.lowpass == (cutoff is not None)
        assert (p.theta_step, p.psi_step) == (theta, psi)
        assert p.defocus_step == pytest.approx(step, abs=1)
        assert p.defocus_half_range == pytest.approx(half, abs=1)
        assert p.relative_snr == pytest.approx(snr, abs=1e-3)
        assert p.low_freq_cutoff == pytest.approx(0.00425, abs=1e-5)  # 400 Å high-pass
    # Coarser with dose, wider with tilt, worse SNR down the acquisition
    assert all(a.relative_snr > b.relative_snr for a, b in itertools.pairwise(schedule))


def test_defocus_range_follows_the_gradient_direction():
    tilt = TiltImage(angle=45.0, pre_exposure=0.0, dose=3.0)
    wide = Acquisition(1.0, (4000, 1000), 300.0, 0.0, 250.0)
    along_x, along_y = (
        schedule_tilts(wide, [dataclasses.replace(tilt, gradient=g)])[0]
        for g in ((1.0, 0.0), (0.0, 1.0))
    )
    # Half-extent along the gradient x tan(45°) + margin, rounded up to the step
    assert along_x.defocus_half_range >= 2000.0 + 300.0
    assert along_y.defocus_half_range < along_x.defocus_half_range


def test_leopard_fragment():
    p = schedule_tilts(REFERENCE, _reference_tilts())[4]
    fragment = p.leopard(falloff=0.02, whitening_max_freq=0.5)
    assert fragment["defocus_search_config"]["defocus_min"] == -p.defocus_half_range
    bandpass = fragment["preprocessing_filters"]["bandpass_filter"]
    assert bandpass == {
        "enabled": True, "low_freq_cutoff": p.low_freq_cutoff,
        "high_freq_cutoff": 0.3754, "falloff": 0.02,
    }  # fmt: skip


# --------------------------------------------------------------------- inferred inputs
def _relion_tilt_series(
    *, pre_exposure: bool = True, ctf: bool = True, x_tilt: float = 0.0, name: str = "TS_01"
) -> TiltSeries:
    order = _dose_symmetric(step=10.0, limit=60.0)
    angles = np.sort(np.array(order))  # stored by angle, acquired dose-symmetrically
    ts = make_tilt_series(name, angles, pixel_size=2.0, image_size=(1000, 800), tilt_axis=85.0)
    if x_tilt:
        projections = xyz_euler_to_matrix(x_tilt, angles, 85.0)[:, :2]
        ts = dataclasses.replace(ts, projections=projections)
    tilts = pd.DataFrame({L.NOMINAL_TILT: angles})
    if pre_exposure:
        tilts[L.PRE_EXPOSURE] = [3.0 * order.index(a) for a in angles]
    if ctf:
        tilts[L.DEFOCUS_U] = 30000.0 + angles
        tilts[L.DEFOCUS_V] = 29000.0 + angles
        tilts[L.DEFOCUS_ANGLE] = 45.0
    row = {L.VOLTAGE: 200.0, L.SPHERICAL_ABERRATION: 2.7, L.AMPLITUDE_CONTRAST: 0.07}
    return dataclasses.replace(ts, metadata={"relion5.global": row, "relion5.tilts": tilts})


def test_infer_acquisition_from_relion_metadata():
    ts = _relion_tilt_series()
    inputs = infer_acquisition(ts, particle_diameter=250.0)
    angles = ts.metadata["relion5.tilts"][L.NOMINAL_TILT].to_numpy()
    assert inputs.acquisition.thickness == 1000.0  # Z size of the tomogram
    assert inputs.acquisition.voltage == 200.0
    assert inputs.optics.amplitude_contrast == 0.07
    np.testing.assert_allclose([t.angle for t in inputs.tilts], np.abs(angles), atol=1e-9)
    np.testing.assert_allclose([t.dose for t in inputs.tilts], 3.0)  # increments (last: median)
    assert inputs.acquisition_order[int(np.argmin(np.abs(angles)))] == 0
    assert inputs.ctfs[0].defocus_u == 30000.0 + angles[0]
    # The defocus gradient is perpendicular to the tilt axis: Y, rotated by ZRot = 85°
    gx, gy = inputs.tilts[0].gradient
    axis = (-math.sin(math.radians(85.0)), math.cos(math.radians(85.0)))
    assert gx * axis[0] + gy * axis[1] == pytest.approx(0.0, abs=1e-9)


def test_excluded_tilts_do_not_inflate_the_dose():
    ts = _relion_tilt_series()
    tilts = ts.metadata["relion5.tilts"]
    keep = (tilts[L.PRE_EXPOSURE] != 6.0).to_numpy()  # third acquired tilt excluded
    ts = dataclasses.replace(
        ts,
        tilt_ids=tuple(np.array(ts.tilt_ids)[keep]),
        images=(),
        projections=ts.projections[keep],
        translations=ts.translations[keep],
        metadata={**ts.metadata, "relion5.tilts": tilts[keep].reset_index(drop=True)},
    )
    inputs = infer_acquisition(ts, particle_diameter=250.0)
    np.testing.assert_allclose([t.dose for t in inputs.tilts], 3.0)


def test_pretilt_is_included_in_the_angle():
    ts = _relion_tilt_series(x_tilt=10.0)
    inputs = infer_acquisition(ts, particle_diameter=250.0)
    untilted = int(np.argmin(np.abs(ts.metadata["relion5.tilts"][L.NOMINAL_TILT])))
    assert inputs.tilts[untilted].angle == pytest.approx(10.0)


def test_missing_pre_exposure_needs_the_dose():
    ts = _relion_tilt_series(pre_exposure=False)
    with pytest.raises(ValueError, match="dose-per-tilt"):
        infer_acquisition(ts, particle_diameter=250.0)
    inputs = infer_acquisition(ts, particle_diameter=250.0, dose_per_tilt=2.0)
    pre = {abs(round(t.angle)): t.pre_exposure for t in inputs.tilts}
    assert pre[0] == 0.0 and pre[60] > pre[10]


def test_missing_ctf_is_an_error():
    with pytest.raises(ValueError, match="CTF"):
        infer_acquisition(_relion_tilt_series(ctf=False), particle_diameter=250.0)


def test_metadata_is_required():
    with pytest.raises(ValueError, match="RELION 5"):
        infer_acquisition(make_tilt_series(), particle_diameter=250.0)


# --------------------------------------------------------------------------- planning
def test_plan_selects_the_best_tilts_of_each_tilt_series(tmp_path):
    tomograms = TomogramSet.of(
        [_relion_tilt_series(name="TS_01"), _relion_tilt_series(name="TS_02")]
    )
    plan = plan_matching(tomograms, particle_diameter=250.0, max_tilts=3)
    snr = [t.parameters.relative_snr for t in plan]
    assert snr == sorted(snr, reverse=True)
    for name in ("TS_01", "TS_02"):
        chosen = [t for t in plan if t.tilt_series == name and t.selected]
        assert sorted(t.acquisition_order for t in chosen) == [0, 1, 2]

    table = schedule_table(plan)
    assert len(table) == len(plan) and table["selected"].sum() == 6
    assert list(table["acquisition_order"][:13]) == list(range(13))

    tilt = plan[0]
    config = leopard_config(tilt, template=tmp_path / "t.mrc", result=tmp_path / "r.h5")
    assert config["optics_group"]["defocus_u"] == tilt.ctf.defocus_u
    assert config["optics_group"]["voltage"] == 200.0
    assert config["orientation_search_config"]["symmetry"] == "C1"
    assert config["match_template_result"]["hdf5_path"].endswith("r.h5")


def test_collect_picks_from_peaks(tmp_path):
    plan = plan_matching(
        TomogramSet.of([_relion_tilt_series()]), particle_diameter=250.0, max_tilts=2
    )
    (tmp_path / "peaks").mkdir()
    for tilt in plan[:3]:  # the third one is not selected: ignored
        pd.DataFrame(
            {"pos_x_img": [10.0, 20.0], "pos_y_img": [5.0, 6.0], "scaled_mip": [9.0, 8.5]}
        ).to_csv(tmp_path / "peaks" / f"{tilt.stem}.csv", index=False)
    picks = collect_picks(plan, tmp_path)
    assert len(picks) == 4
    assert set(picks.tilt_ids) == {plan[0].tilt_id, plan[1].tilt_id}
    np.testing.assert_allclose(picks.scores, [9.0, 8.5, 9.0, 8.5])


def test_command_only_registered_with_leopard(monkeypatch):
    from parapick.cli import commands

    monkeypatch.setattr(leopard, "available", lambda: False)
    assert commands._optional_commands() == ()
    monkeypatch.setattr(leopard, "available", lambda: True)
    assert [c.name for c in commands._optional_commands()] == ["match-template"]
