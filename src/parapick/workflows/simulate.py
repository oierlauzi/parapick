"""Synthetic tilt-series picking datasets for testing, benchmarking and demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parapick.formats import FormatRegistry, WriteContext, default_registry
from parapick.tomography import Particles, PickTable, TiltSeries, TomogramSet, xyz_euler_to_matrix

__all__ = ["SimulatedDataset", "SimulationConfig", "simulate", "write_dataset"]


@dataclass(frozen=True, kw_only=True)
class SimulationConfig:
    """Parameters of a synthetic dataset. Lengths in Å, angles in degrees.

    The specimen is a slab of ``thickness`` covering the field of view of the untilted image.
    Each particle is picked on each tilt with probability ``recall``, with isotropic Gaussian
    error ``pick_noise``. ``false_positives`` uniform random picks are added per tilt.
    """

    n_tilt_series: int = 2
    n_particles: int = 50
    tilt_min: float = -60.0
    tilt_max: float = 60.0
    tilt_step: float = 3.0
    tilt_axis: float = 85.0
    shift_jitter: float = 20.0
    pixel_size: float = 10.0
    image_size: tuple[int, int] = (512, 512)
    thickness: float = 1000.0
    min_separation: float = 100.0
    pick_noise: float = 10.0
    recall: float = 0.7
    false_positives: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.recall <= 1:
            raise ValueError("recall must be in (0, 1]")
        if self.tilt_step <= 0 or self.tilt_max < self.tilt_min:
            raise ValueError("Invalid tilt range")


@dataclass(frozen=True)
class SimulatedDataset:
    """Geometry, ground-truth particles and noisy picks."""

    tomograms: TomogramSet
    ground_truth: Particles
    picks: PickTable


def _place_particles(
    rng: np.random.Generator, n: int, box: np.ndarray, min_separation: float
) -> np.ndarray:
    """Uniform positions in the box, rejecting those closer than ``min_separation``."""
    points: list[np.ndarray] = []
    attempts = 0
    while len(points) < n and attempts < 1000 * max(n, 1):
        attempts += 1
        candidate = rng.uniform(-box / 2, box / 2)
        if all(np.linalg.norm(candidate - p) >= min_separation for p in points):
            points.append(candidate)
    if len(points) < n:
        raise ValueError("Could not place all particles; reduce n_particles or min_separation")
    return np.asarray(points).reshape(-1, 3)


def simulate(config: SimulationConfig | None = None) -> SimulatedDataset:
    """Generate a synthetic dataset (deterministic for a given seed)."""
    config = config or SimulationConfig()
    rng = np.random.default_rng(config.seed)
    angles = np.arange(config.tilt_min, config.tilt_max + 1e-9, config.tilt_step)
    width, height = config.image_size
    pixel = config.pixel_size
    image_extent = np.array([width, height], dtype=np.float64) * pixel
    # Keep particles inside the field of view of every tilt (along the tilt axis direction the
    # extent is preserved; across it the projection shrinks with cos(tilt)).
    margin = 0.9 * np.cos(np.deg2rad(np.max(np.abs(angles))))
    box = np.array([*(image_extent * margin), config.thickness])

    series, truths, pick_tables = [], [], []
    for s in range(config.n_tilt_series):
        name = f"TS_{s + 1:02d}"
        tilt_ids = tuple(f"tilts/{name}_{i + 1:03d}_{a:.1f}.mrc" for i, a in enumerate(angles))
        rotations = xyz_euler_to_matrix(0.0, angles, config.tilt_axis)
        projections = rotations[:, :2, :]
        shifts = rng.normal(0.0, config.shift_jitter, size=(len(angles), 2))
        translations = np.floor(np.array([width, height]) / 2.0) * pixel + shifts
        ts = TiltSeries(
            name=name,
            tilt_ids=tilt_ids,
            projections=projections,
            translations=translations,
            pixel_size=pixel,
            image_size=(width, height),
            volume_size=(float(image_extent[0]), float(image_extent[1]), config.thickness),
        )
        series.append(ts)

        positions = _place_particles(rng, config.n_particles, box, config.min_separation)
        truths.append(Particles(np.full(len(positions), name, dtype=object), positions))

        projected = ts.project(positions)  # (N, K, 2) Å
        ids, coords = [], []
        for i in range(ts.n_tilts):
            visible = rng.random(len(positions)) < config.recall
            true_picks = projected[i, visible] + rng.normal(
                0.0, config.pick_noise, size=(int(visible.sum()), 2)
            )
            false_picks = rng.uniform((0.0, 0.0), image_extent, size=(config.false_positives, 2))
            tilt_coords = np.concatenate([true_picks, false_picks])
            inside = np.all((tilt_coords >= 0) & (tilt_coords < image_extent), axis=1)
            coords.append(tilt_coords[inside] / pixel)
            ids += [tilt_ids[i]] * int(inside.sum())
        coordinates = np.concatenate(coords)
        pick_tables.append(
            PickTable(
                tilt_ids=np.array(ids, dtype=object),
                coordinates=coordinates,
                tilt_series=np.full(len(coordinates), name, dtype=object),
            )
        )

    return SimulatedDataset(
        tomograms=TomogramSet(tuple(series)),
        ground_truth=Particles.concatenate(truths),
        picks=PickTable.concatenate(pick_tables),
    )


def write_dataset(
    dataset: SimulatedDataset,
    directory: Path,
    *,
    output_format: str = "relion5",
    picks_format: str | None = None,
    registry: FormatRegistry | None = None,
) -> list[Path]:
    """Write ``tomograms``, ``ground_truth`` particles and ``picks`` into ``directory``."""
    registry = registry or default_registry()
    tomo_writer = registry.tomograms_writer(output_format)
    particles_writer = registry.particles_writer(output_format)
    picks_writer = registry.picks_writer(picks_format or output_format)

    def named(stem: str, fmt: object) -> Path:
        suffixes = getattr(fmt, "suffixes", ())
        return directory / f"{stem}{suffixes[0] if suffixes else ''}"

    tomograms_path = named("tomograms", tomo_writer)
    written = tomo_writer.write_tomograms(dataset.tomograms, tomograms_path)
    written += particles_writer.write_particles(
        dataset.ground_truth,
        named("ground_truth", particles_writer),
        WriteContext(tomograms=dataset.tomograms),
    )
    written += picks_writer.write_picks(dataset.picks, named("picks", picks_writer))
    return written
