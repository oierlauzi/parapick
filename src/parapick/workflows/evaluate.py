"""Compare localised particles with a reference (e.g. ground truth)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from parapick.tomography import Particles

__all__ = ["Evaluation", "Metrics", "evaluate"]


@dataclass(frozen=True)
class Metrics:
    """Detection metrics of one tilt-series (or of all of them)."""

    tilt_series: str
    n_predicted: int
    n_reference: int
    true_positives: int
    precision: float
    recall: float
    f1: float
    rmse: float
    """Root-mean-square distance (Å) between matched pairs."""

    @classmethod
    def from_counts(
        cls, name: str, n_predicted: int, n_reference: int, distances: np.ndarray[Any, Any]
    ) -> Metrics:
        """Build metrics from counts and the distances of the matched pairs."""
        tp = len(distances)
        precision = tp / n_predicted if n_predicted else float("nan")
        recall = tp / n_reference if n_reference else float("nan")
        denominator = n_predicted + n_reference
        return cls(
            tilt_series=name,
            n_predicted=n_predicted,
            n_reference=n_reference,
            true_positives=tp,
            precision=precision,
            recall=recall,
            f1=2 * tp / denominator if denominator else float("nan"),
            rmse=float(np.sqrt(np.mean(distances**2))) if tp else float("nan"),
        )


@dataclass(frozen=True)
class Evaluation:
    """Per tilt-series and pooled metrics."""

    per_tilt_series: tuple[Metrics, ...]
    overall: Metrics
    cutoff: float

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return {
            "cutoff": self.cutoff,
            "overall": asdict(self.overall),
            "tilt_series": [asdict(m) for m in self.per_tilt_series],
        }


def evaluate(predicted: Particles, reference: Particles, *, cutoff: float) -> Evaluation:
    """Match predictions to references one-to-one (Hungarian, distance ≤ ``cutoff`` Å).

    Precision, recall, F1 and the RMSE of matched pairs are reported per tilt-series and
    pooled over all of them.
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    names = list(dict.fromkeys([*reference.names, *predicted.names]))
    per_ts = []
    all_distances = []
    for name in names:
        pred = predicted.coordinates[predicted.tilt_series == name]
        ref = reference.coordinates[reference.tilt_series == name]
        distances = np.empty(0)
        if len(pred) and len(ref):
            cost = cdist(pred, ref)
            penalised = np.where(cost <= cutoff, cost, cutoff * 1e6)
            rows, cols = linear_sum_assignment(penalised)
            matched = cost[rows, cols]
            distances = matched[matched <= cutoff]
        all_distances.append(distances)
        per_ts.append(Metrics.from_counts(name, len(pred), len(ref), distances))
    overall = Metrics.from_counts(
        "all",
        len(predicted),
        len(reference),
        np.concatenate(all_distances) if all_distances else np.empty(0),
    )
    return Evaluation(tuple(per_ts), overall, cutoff)
