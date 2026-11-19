"""Binning by Fourier cropping (numpy only).

The first pixel stays where it was: binned pixel ``i`` is at native coordinate ``i * factor``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["binned_shape", "fourier_crop"]


def binned_shape(shape: tuple[int, ...], factor: int) -> tuple[int, ...]:
    """Shape after binning by ``factor`` (trailing pixels are dropped)."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    return tuple(max(1, n // factor) for n in shape)


def fourier_crop(data: ArrayLike, factor: int) -> NDArray[np.float32]:
    """Bin a 2D image or 3D volume by keeping its lowest frequencies.

    The mean is preserved. ``factor == 1`` returns a float32 copy.
    """
    array = np.asarray(data, dtype=np.float64)
    if factor == 1:
        return array.astype(np.float32)
    shape = binned_shape(array.shape, factor)
    spectrum = np.fft.fftshift(np.fft.fftn(array))
    window = tuple(
        slice(n // 2 - m // 2, n // 2 - m // 2 + m) for n, m in zip(array.shape, shape, strict=True)
    )
    cropped = np.fft.ifftn(np.fft.ifftshift(spectrum[window])).real
    scale = np.prod(shape) / np.prod(array.shape)
    result: NDArray[np.float32] = (cropped * scale).astype(np.float32)
    return result
