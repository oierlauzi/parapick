"""Run Leopard-EM ``match_template`` (optional dependency: ``pip install parapick[leopard]``).

This is the only module that imports Leopard-EM (and PyTorch), and only when it is used.
"""

from __future__ import annotations

import gc
import importlib.util
import logging
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["available", "gpu_ids_for", "run_match_template", "template_pixel_size", "write_config"]

logger = logging.getLogger(__name__)


def available() -> bool:
    """Whether Leopard-EM is installed (without importing it)."""
    return importlib.util.find_spec("leopard_em") is not None


def gpu_ids_for(device: str) -> str:
    """Leopard-EM ``gpu_ids`` for a ParaPick ``--device`` choice."""
    if device == "cpu":
        return "cpu"
    if device == "gpu":
        return "all"
    import torch

    return "all" if torch.cuda.is_available() else "cpu"


def template_pixel_size(path: Path) -> float | None:
    """Voxel size (Å) in the header of a template, ``None`` if unset."""
    import mrcfile

    with mrcfile.open(path, header_only=True, permissive=True) as mrc:
        size = float(mrc.voxel_size.x)
    return size if size > 0 else None


def write_config(config: dict[str, Any], path: Path) -> None:
    """Validate a ``match_template`` configuration with Leopard-EM and write it as YAML."""
    from leopard_em.pydantic_models.managers import MatchTemplateManager

    path.parent.mkdir(parents=True, exist_ok=True)
    MatchTemplateManager(**config).to_yaml(path)


def run_match_template(
    config_path: Path,
    *,
    orientation_batch_size: int,
    false_positives: float,
    mask_radius: float,
) -> pd.DataFrame:
    """Search one tilt image and return its peaks (Leopard-EM's table, one row per peak).

    Peaks are thresholded by Leopard-EM for ``false_positives`` expected false positives on
    the image, and closer than ``mask_radius`` pixels to a higher one are dropped.
    ``pos_x_img``/``pos_y_img`` are the particle centres in pixels of the image.
    """
    import torch
    from leopard_em.pydantic_models.managers import MatchTemplateManager

    manager = MatchTemplateManager.from_yaml(config_path)
    try:
        manager.run_match_template(
            orientation_batch_size=orientation_batch_size, do_result_export=True
        )
        peaks: pd.DataFrame = manager.results_to_dataframe(
            locate_peaks_kwargs={"false_positives": false_positives, "mask_radius": mask_radius}
        )
    finally:
        del manager
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return peaks
