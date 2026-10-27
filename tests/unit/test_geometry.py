"""Rotations, RELION 5 projection convention and the neutral tilt-series model."""

from __future__ import annotations

import numpy as np
import pytest

from parapick.formats.relion5.conventions import from_relion, to_relion
from parapick.tomography import (
    TiltSeries,
    TomogramSet,
    complete_rotation,
    matrix_to_xyz_euler,
    rotation_x,
    rotation_y,
    rotation_z,
    xyz_euler_to_matrix,
)


def _relion_affine(x_tilt, y_tilt, z_rot, shift_angst, pixel_size, image_size, tomo_size):
    """Transcription of RELION 5 ``Tomogram::setProjectionMatrix`` (4x4, pixel units)."""

    def translation(v):
        m = np.eye(4)
        m[:3, 3] = v
        return m

    def rot(r):
        m = np.eye(4)
        m[:3, :3] = r
        return m

    s0 = translation(-np.array([int(t / 2) for t in tomo_size], dtype=float))
    s1 = translation([shift_angst[0] / pixel_size, shift_angst[1] / pixel_size, 0])
    s2 = translation([int(image_size[0] / 2), int(image_size[1] / 2), 0])
    r0, r1, r2 = rot(rotation_x(x_tilt)), rot(rotation_y(y_tilt)), rot(rotation_z(z_rot))
    return s2 @ s1 @ r2 @ r1 @ r0 @ s0


def test_elementary_rotations_are_right_handed():
    np.testing.assert_allclose(rotation_z(90) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(rotation_x(90) @ [0, 1, 0], [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(rotation_y(90) @ [0, 0, 1], [1, 0, 0], atol=1e-12)


def test_rotations_broadcast_over_angles():
    r = xyz_euler_to_matrix(0.0, np.arange(5.0), 10.0)
    assert r.shape == (5, 3, 3)
    identity = np.broadcast_to(np.eye(3), (5, 3, 3))
    np.testing.assert_allclose(r @ r.transpose(0, 2, 1), identity, atol=1e-12)


@pytest.mark.parametrize(
    "angles",
    [(10.0, -35.0, 80.0), (-3.0, 60.0, -95.0), (0.0, 0.0, 0.0), (179.0, 20.0, -179.0)],
)
def test_euler_round_trip(angles):
    matrix = xyz_euler_to_matrix(*angles)
    recovered = matrix_to_xyz_euler(matrix)
    np.testing.assert_allclose(xyz_euler_to_matrix(*recovered), matrix, atol=1e-12)


def test_euler_gimbal_lock_round_trip():
    matrix = xyz_euler_to_matrix(0.0, 90.0, 30.0)
    np.testing.assert_allclose(xyz_euler_to_matrix(*matrix_to_xyz_euler(matrix)), matrix, atol=1e-9)


def test_complete_rotation():
    r = xyz_euler_to_matrix(5.0, 40.0, 85.0)
    np.testing.assert_allclose(complete_rotation(r[:2]), r, atol=1e-12)


@pytest.mark.parametrize("tomo_size", [(4000, 4000, 1200), (4001, 3999, 1201)])
def test_relion_projection_matches_relion(rng, tomo_size):
    pixel_size, image_size = 1.35, (4096, 4095)
    n = 7
    x_tilt = rng.normal(0, 2, n)
    y_tilt = np.linspace(-60, 60, n)
    z_rot = rng.normal(85, 1, n)
    shifts = rng.normal(0, 40, (n, 2))
    p, t = from_relion(
        x_tilt,
        y_tilt,
        z_rot,
        shifts[:, 0],
        shifts[:, 1],
        pixel_size=pixel_size,
        image_size=image_size,
        tomo_size_px=tomo_size,
    )
    # RELION: centred Å -> decentred pixels with tomo_size / 2 (ParticleSet::getPosition).
    centred = rng.uniform(-1500, 1500, (10, 3))
    decentred_px = centred / pixel_size + np.asarray(tomo_size) / 2.0
    for i in range(n):
        affine = _relion_affine(
            x_tilt[i], y_tilt[i], z_rot[i], shifts[i], pixel_size, image_size, tomo_size
        )
        expected_px = (affine @ np.c_[decentred_px, np.ones(10)].T).T[:, :2]
        ours_px = (centred @ p[i].T + t[i]) / pixel_size
        np.testing.assert_allclose(ours_px, expected_px, atol=1e-9)


def test_relion_round_trip(rng):
    kwargs = {"pixel_size": 2.0, "image_size": (1000, 999), "tomo_size_px": (1001, 1000, 301)}
    angles = (rng.normal(0, 1, 5), np.linspace(-50, 50, 5), rng.normal(-80, 2, 5))
    shifts = rng.normal(0, 30, (5, 2))
    p, t = from_relion(*angles, shifts[:, 0], shifts[:, 1], **kwargs)
    back = to_relion(p, t, **kwargs)
    for key, expected in zip(("x_tilt", "y_tilt", "z_rot"), angles, strict=True):
        np.testing.assert_allclose(back[key], expected, atol=1e-9)
    np.testing.assert_allclose(back["x_shift_angst"], shifts[:, 0], atol=1e-9)
    np.testing.assert_allclose(back["y_shift_angst"], shifts[:, 1], atol=1e-9)


def test_to_relion_rejects_non_rigid():
    with pytest.raises(ValueError, match="orthonormal"):
        to_relion(np.ones((1, 2, 3)), np.zeros((1, 2)), pixel_size=1.0, image_size=(10, 10))


def test_tilt_series_validation_and_projection(tilt_series):
    assert tilt_series.n_tilts == 41
    np.testing.assert_allclose(tilt_series.tilt_angles, np.abs(np.arange(-60, 61, 3.0)), atol=1e-9)
    projected = tilt_series.project(np.zeros((2, 3)))
    assert projected.shape == (41, 2, 2)
    np.testing.assert_allclose(projected[:, 0], tilt_series.translations)
    assert not tilt_series.projections.flags.writeable
    with pytest.raises(ValueError, match="shape"):
        TiltSeries("x", ("a",), np.zeros((2, 2, 3)), np.zeros((1, 2)), 1.0)
    with pytest.raises(ValueError, match="duplicated"):
        TiltSeries("x", ("a", "a"), np.zeros((2, 2, 3)), np.zeros((2, 2)), 1.0)
    with pytest.raises(ValueError, match="pixel size"):
        TiltSeries("x", ("a",), np.zeros((1, 2, 3)), np.zeros((1, 2)), 0.0)


def test_tomogram_set_access(tomograms):
    assert tomograms.names == ("TS_01", "TS_02")
    assert "TS_02" in tomograms
    assert tomograms["TS_02"].name == "TS_02"
    assert tomograms.subset(["TS_02"]).names == ("TS_02",)
    with pytest.raises(KeyError):
        tomograms["nope"]
    with pytest.raises(KeyError):
        tomograms.subset(["nope"])
    with pytest.raises(ValueError, match="unique"):
        TomogramSet((tomograms["TS_01"], tomograms["TS_01"]))
