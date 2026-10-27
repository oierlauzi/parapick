"""Image references and MRC input/output."""

from __future__ import annotations

import mrcfile
import numpy as np
import pytest

from parapick.formats.images import (
    ImageReference,
    read_image_size,
    read_section,
    strip_stack_index,
    write_mrc,
)


@pytest.mark.parametrize(
    ("text", "path", "index", "suffix"),
    [
        ("a/b.mrc", "a/b.mrc", None, None),
        ("3@stack.mrcs", "stack.mrcs", 2, None),
        ("TS_01.st:mrc", "TS_01.st", None, "mrc"),
        ("12@TS_01.st:mrc", "TS_01.st", 11, "mrc"),
        ("C:/data/x.mrc", "C:/data/x.mrc", None, None),
        ("tilts/TS_01_-21.00.mrc", "tilts/TS_01_-21.00.mrc", None, None),
    ],
)
def test_image_reference_round_trip(text, path, index, suffix):
    reference = ImageReference.parse(text)
    assert (reference.path, reference.index, reference.type_suffix) == (path, index, suffix)
    assert str(reference) == text
    assert strip_stack_index(text) == path


def test_image_reference_rejects_zero_index():
    with pytest.raises(ValueError, match="1-based"):
        ImageReference.parse("0@stack.mrcs")


def test_mrc_detection():
    assert ImageReference.parse("x.st").is_mrc()
    assert ImageReference.parse("x.weird:mrc").is_mrc()
    assert not ImageReference.parse("x.tif").is_mrc()


def test_read_sections_from_stack_and_files(tmp_path):
    stack = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    with mrcfile.new(tmp_path / "stack.st") as mrc:
        mrc.set_data(stack)
    np.testing.assert_array_equal(read_section("2@stack.st", tmp_path / "anchor.star"), stack[1])
    assert read_image_size("stack.st", tmp_path / "anchor.star") == (5, 4)
    with pytest.raises(ValueError, match="3 sections"):
        read_section(str(tmp_path / "stack.st"))
    with pytest.raises(IndexError):
        read_section(f"4@{tmp_path / 'stack.st'}")

    single = write_mrc(tmp_path / "single.mrc", stack[2], pixel_size=1.5)
    np.testing.assert_array_equal(read_section(str(single)), stack[2])
    with mrcfile.open(single) as mrc:
        assert mrc.header.nz == 1
        assert float(mrc.voxel_size.x) == pytest.approx(1.5)

    with mrcfile.new(tmp_path / "one.mrcs") as mrc:
        mrc.set_data(stack[:1])
    np.testing.assert_array_equal(read_section(str(tmp_path / "one.mrcs")), stack[0])


def test_unsupported_and_invalid(tmp_path):
    with pytest.raises(ValueError, match="Unsupported image format"):
        read_section(str(tmp_path / "x.tif"))
    with pytest.raises(ValueError, match="2D"):
        write_mrc(tmp_path / "x.mrc", np.zeros((2, 2, 2)), 1.0)
