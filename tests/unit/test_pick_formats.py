"""SPA pick formats: EMAN box, crYOLO CBOX, cryoSPARC, Xmipp; registry sniffing; flip_y."""

from __future__ import annotations

import numpy as np
import pytest

from parapick.formats import default_registry
from parapick.formats.cryolo_cbox import CryoloCboxFormat
from parapick.formats.cryosparc import CryosparcFormat
from parapick.formats.eman_box import EmanBoxFormat
from parapick.formats.xmipp_pos import XmippPosFormat
from parapick.tomography import PickTable, annotate_picks, bind_picks

from helpers.pick_files import (
    write_cbox_legacy,
    write_cbox_star,
    write_cryosparc,
    write_eman_boxes,
    write_xmipp_pos,
)


@pytest.fixture
def picks() -> PickTable:
    return PickTable(
        tilt_ids=np.array(["mics/TS_01_001.mrc"] * 2 + ["mics/TS_01_002.mrc"], dtype=object),
        coordinates=np.array([[100.0, 200.0], [300.0, 50.0], [64.0, 128.0]]),
        scores=np.array([0.9, 0.2, 0.7]),
    )


def _by_tilt(table: PickTable) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {}
    for tilt, xy in zip(table.tilt_ids, table.coordinates, strict=True):
        out.setdefault(str(tilt), []).append(xy)
    return {k: np.array(v) for k, v in out.items()}


def test_eman_box_centres(tmp_path, picks):
    directory = write_eman_boxes(picks, tmp_path / "EMAN")
    fmt = EmanBoxFormat()
    assert fmt.sniff(directory)
    table = fmt.read_picks(directory)
    assert table.scores is None
    got = _by_tilt(table)
    np.testing.assert_allclose(got["TS_01_001"], picks.coordinates[:2])
    np.testing.assert_allclose(got["TS_01_002"], picks.coordinates[2:])
    single = fmt.read_picks(directory / "TS_01_002.box")
    np.testing.assert_allclose(single.coordinates, picks.coordinates[2:])
    (directory / "empty.box").write_text("")
    assert len(fmt.read_picks(directory)) == 3
    (tmp_path / "words.box").write_text("not a box file\n")
    assert not fmt.sniff(tmp_path / "words.box")


@pytest.mark.parametrize("writer", [write_cbox_star, write_cbox_legacy])
def test_cryolo_cbox_variants(tmp_path, picks, writer):
    directory = writer(picks, tmp_path / "CBOX")
    table = CryoloCboxFormat().read_picks(directory)
    order = np.argsort(table.coordinates[:, 0])
    expected = np.argsort(picks.coordinates[:, 0])
    np.testing.assert_allclose(table.coordinates[order], picks.coordinates[expected], atol=1e-4)
    np.testing.assert_allclose(table.scores[order], picks.scores[expected], atol=1e-6)


def test_xmipp_pos_read_and_write(tmp_path, picks):
    fmt = XmippPosFormat()
    directory = write_xmipp_pos(picks, tmp_path / "xmipp")  # as Xmipp writes it
    table = fmt.read_picks(directory)
    assert table.scores is None
    np.testing.assert_allclose(_by_tilt(table)["TS_01_001"], picks.coordinates[:2])
    written = fmt.write_picks(picks, tmp_path / "ours")
    assert [p.name for p in written] == ["TS_01_001.pos", "TS_01_002.pos"]
    assert written[0].read_text().startswith("# XMIPP_STAR_1")
    again = fmt.read_picks(tmp_path / "ours")
    np.testing.assert_allclose(np.sort(again.scores), np.sort(picks.scores), atol=1e-6)


def test_cryosparc(tmp_path, picks):
    path = write_cryosparc(picks, tmp_path / "picked.cs", (512, 400), pixel_size=1.35)
    fmt = CryosparcFormat()
    assert fmt.sniff(path)
    table = fmt.read_picks(path)
    assert list(table.tilt_ids) == ["TS_01_001.mrc", "TS_01_001.mrc", "TS_01_002.mrc"]
    np.testing.assert_allclose(table.coordinates, picks.coordinates, atol=1e-3)
    np.testing.assert_allclose(table.scores, picks.scores, atol=1e-6)
    assert table.pixel_size == pytest.approx(1.35)
    no_locations = tmp_path / "other.cs"
    with no_locations.open("wb") as handle:
        np.save(handle, np.zeros(2, dtype=[("uid", "<u8")]))
    with pytest.raises(ValueError, match="no particle locations"):
        fmt.read_picks(no_locations)
    (tmp_path / "text.cs").write_text("hello")
    assert not fmt.sniff(tmp_path / "text.cs")


def test_registry_detects_each_format(tmp_path, picks):
    registry = default_registry()
    cases = {
        "eman-box": write_eman_boxes(picks, tmp_path / "box"),
        "cryolo-cbox": write_cbox_star(picks, tmp_path / "cbox"),
        "xmipp-pos": write_xmipp_pos(picks, tmp_path / "pos"),
        "cryosparc": write_cryosparc(picks, tmp_path / "p.cs", (512, 512)),
    }
    for name, path in cases.items():
        assert registry.picks_reader(path=path).name == name
    assert registry.picks_reader(path=cases["eman-box"] / "TS_01_001.box").name == "eman-box"


def test_flip_y(tomograms):
    ts = tomograms["TS_01"]
    height = ts.image_size[1]
    picks = PickTable(
        tilt_ids=np.array([ts.tilt_ids[0]], dtype=object),
        coordinates=np.array([[10.0, 30.0]]),
        tilt_series=np.array(["TS_01"], dtype=object),
    )
    bound, _ = bind_picks(picks, tomograms, flip_y=True)
    np.testing.assert_allclose(
        bound["TS_01"].coordinates, [[10.0 * ts.pixel_size, (height - 30.0) * ts.pixel_size]]
    )
    binned = PickTable(**{**picks.__dict__, "pixel_size": 2 * ts.pixel_size})
    annotated, _ = annotate_picks(binned, tomograms, flip_y=True)
    np.testing.assert_allclose(annotated.coordinates, [[10.0, height / 2 - 30.0]])
