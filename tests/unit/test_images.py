"""Image references, the ImageIO interface and its vitrio implementation."""

from __future__ import annotations

from concurrent.futures import Future

import mrcfile
import numpy as np
import pytest

from parapick.formats.images import (
    ImageIO,
    ImageIOError,
    ImageReference,
    Pending,
    completed,
    default_image_io,
    failed,
    read_image_size,
    read_section,
    strip_stack_index,
)
from parapick.formats.vitrio_images import VitrioImageIO

STACK = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5)


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


def test_reference_resolution(tmp_path):
    (tmp_path / "x.mrc").touch()
    resolved = ImageReference.parse("2@x.mrc:mrc").resolved(tmp_path / "tomograms.star")
    assert resolved.path == str(tmp_path / "x.mrc")
    assert (resolved.index, resolved.type_suffix) == (1, "mrc")
    assert ImageReference.parse("y.mrc").resolved(None).path == "y.mrc"


def test_pending_helpers():
    done: Future[int] = completed(3)
    assert isinstance(done, Pending) and done.done() and done.result() == 3
    with pytest.raises(KeyError):
        failed(KeyError("x")).result()


def test_default_io_is_vitrio():
    io = default_image_io()
    assert isinstance(io, VitrioImageIO) and isinstance(io, ImageIO)
    assert default_image_io() is io


@pytest.fixture
def image_stack(tmp_path):
    """An MRC2014 image stack (space group 0), as IMOD writes tilt-series."""
    path = tmp_path / "stack.st"
    with mrcfile.new(path) as mrc:
        mrc.set_data(STACK)
        mrc.set_image_stack()
    return path


@pytest.fixture
def volume_stack(tmp_path):
    """The same sections flagged as a volume (space group 1), as some tools write them."""
    path = tmp_path / "volume.mrc"
    with mrcfile.new(path) as mrc:
        mrc.set_data(STACK)
    return path


@pytest.mark.parametrize("stack", ["image_stack", "volume_stack"])
def test_read_sections(request, tmp_path, stack):
    path = request.getfixturevalue(stack)
    anchor = tmp_path / "tomograms.star"
    np.testing.assert_array_equal(read_section(f"2@{path.name}", anchor), STACK[1])
    assert read_image_size(path.name, anchor) == (5, 4)
    batch = VitrioImageIO().read_batch([ImageReference(str(path), i) for i in (2, 0)]).result()
    assert batch.dtype == np.float32
    np.testing.assert_array_equal(batch, STACK[[2, 0]])
    with pytest.raises(ImageIOError, match="3 sections"):
        read_section(str(path))
    with pytest.raises(ImageIOError, match="out of range"):
        read_section(f"4@{path}")


def test_write_batch_is_readable_by_other_tools(tmp_path):
    io = VitrioImageIO()
    images = STACK.astype(np.float32)
    paths = [tmp_path / "out" / f"{i}.mrc" for i in range(3)]
    pending = io.write_batch(images, paths)
    assert pending.result() is None and pending.done()
    for path in paths:
        io.finish(path).result()
    for image, path in zip(images, paths, strict=True):
        with mrcfile.open(path) as mrc:  # what crYOLO uses
            assert mrc.header.nz == 1
            np.testing.assert_array_equal(mrc.data, image)
        np.testing.assert_array_equal(read_section(str(path)), image)


def test_io_errors(tmp_path, image_stack):
    io = VitrioImageIO()
    with pytest.raises(ImageIOError, match="Cannot read the header"):
        io.read_batch([ImageReference(str(tmp_path / "missing.mrc"))]).result()
    small = tmp_path / "small.mrc"
    io.write_batch(np.zeros((1, 2, 2), np.float32), [small]).result()
    io.finish(small).result()
    with pytest.raises(ImageIOError, match="batch"):
        io.read_batch([ImageReference(str(image_stack), 0), ImageReference(str(small))]).result()
    with pytest.raises(ValueError, match="B paths"):
        io.write_batch(np.zeros((2, 2, 2), np.float32), [small]).result()
    assert io.read_batch([]).result().shape[0] == 0
    with pytest.raises(ImageIOError, match="Cannot finish"):
        io.finish(tmp_path / "never_written.mrc").result()
    blocker = tmp_path / "file"
    blocker.touch()
    with pytest.raises(ImageIOError, match="Cannot write"):
        io.write_batch(np.zeros((1, 2, 2), np.float32), [blocker / "x.mrc"]).result()


def test_read_batch_mixes_files_and_stack_sections(tmp_path, image_stack):
    """Whole files and stack sections can share a batch (vitrio reads them in separate runs)."""
    io = VitrioImageIO(workers=2)
    single = tmp_path / "single.mrc"
    io.write_batch(STACK[1:2].astype(np.float32), [single]).result()
    io.finish(single).result()
    references = [
        ImageReference(str(single)),
        ImageReference(str(image_stack), 2),
        ImageReference(str(image_stack), 0),
        ImageReference(str(single)),
    ]
    batch = io.read_batch(references).result()
    np.testing.assert_array_equal(batch, STACK[[1, 2, 0, 1]])
