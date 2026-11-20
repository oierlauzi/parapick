"""Run Leopard-EM ``match_template`` (optional dependency: ``pip install parapick[leopard]``).

Leopard-EM (https://github.com/Lucaslab-Berkeley/Leopard-EM) does the template matching;
ParaPick only derives its parameters. Results obtained with it must cite :data:`CITATION`.

This is the only module that imports Leopard-EM (and PyTorch), and only when it is used.
"""

from __future__ import annotations

import gc
import importlib.util
import logging
from pathlib import Path
from typing import Any

__all__ = [
    "CITATION",
    "available",
    "gpu_ids_for",
    "read_results",
    "run_search",
    "template_box",
    "template_pixel_size",
    "write_binned",
    "write_config",
]

logger = logging.getLogger(__name__)

CITATION = (
    "Giammar MD, Dickerson JL, Hall LN, Lucas BA (2026). Leopard-EM: an extensible 2D "
    "template-matching package to accelerate in situ structural biology. Acta Cryst. D 82, "
    "30-42. https://doi.org/10.1107/S2059798325009982"
)
"""Leopard-EM (Lucas lab, UC Berkeley; BSD-3-Clause), which does the template matching."""


def available() -> bool:
    """Whether Leopard-EM is installed (without importing it)."""
    return importlib.util.find_spec("leopard_em") is not None


def gpu_ids_for(device: str) -> str:
    """Leopard-EM ``gpu_ids`` for a ParaPick ``--device`` choice: all the visible GPUs.

    Leopard-EM 1.3 cannot search on the CPU: its search fails in a worker process, and the
    run then hangs instead of failing. So a search without a GPU is refused here.
    """
    if device == "cpu":
        raise ValueError("Leopard-EM searches only on a GPU: use --device gpu or auto")
    import torch

    if not torch.cuda.is_available():
        raise ValueError("Leopard-EM needs a GPU, and none is visible (CUDA_VISIBLE_DEVICES?)")
    return "all"


def template_pixel_size(path: Path) -> float | None:
    """Voxel size (Å) in the header of a template, ``None`` if unset."""
    import mrcfile

    with mrcfile.open(path, header_only=True, permissive=True) as mrc:
        size = float(mrc.voxel_size.x)
    return size if size > 0 else None


def write_binned(source: Path, target: Path, factor: int) -> None:
    """Write ``source`` (2D image or 3D volume) binned by ``factor`` as an MRC file."""
    import mrcfile

    from parapick.matching.binning import fourier_crop

    with mrcfile.open(source, permissive=True) as mrc:
        data = fourier_crop(mrc.data, factor)
        size = float(mrc.voxel_size.x)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.mrc")
    with mrcfile.new(temporary, overwrite=True) as mrc:
        mrc.set_data(data)
        if size > 0:
            mrc.voxel_size = size * factor
    temporary.replace(target)


def write_config(config: dict[str, Any], path: Path) -> None:
    """Validate a ``match_template`` configuration with Leopard-EM and write it as YAML."""
    from leopard_em.pydantic_models.managers import MatchTemplateManager

    path.parent.mkdir(parents=True, exist_ok=True)
    MatchTemplateManager(**config).to_yaml(path)


def run_search(config_path: Path, *, orientation_batch_size: int) -> None:
    """Search one tilt image and write Leopard-EM's results (the HDF5 file of the config)."""
    import torch
    from leopard_em.pydantic_models.managers import MatchTemplateManager

    manager = MatchTemplateManager.from_yaml(config_path)
    try:
        manager.run_match_template(
            orientation_batch_size=orientation_batch_size, do_result_export=True
        )
    finally:
        del manager
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


RESULT_TENSORS = (
    "mip",
    "scaled_mip",
    "orientation_phi",
    "orientation_theta",
    "orientation_psi",
    "relative_defocus",
)


def read_results(path: Path) -> tuple[dict[str, Any], int]:
    """The result maps of a search (numpy arrays) and its number of projections."""
    import h5py

    with h5py.File(path, "r") as results:
        tensors = {name: results["tensors"][name][()] for name in RESULT_TENSORS}
        projections = int(results.attrs["total_projections"])
    return tensors, projections


def template_box(path: Path) -> int:
    """Box size (pixels) of a template volume."""
    import mrcfile

    with mrcfile.open(path, header_only=True, permissive=True) as mrc:
        return int(mrc.header.nx)
