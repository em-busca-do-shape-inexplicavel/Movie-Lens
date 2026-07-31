"""Hyperparameter experiment selection without test-set leakage."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from training.configuration import TrainingPipelineConfig
from training.pipeline import MetricValue, TrainingPipelineResult


@dataclass(frozen=True, slots=True)
class CandidateExperimentResult:
    """One validation experiment and its MLflow run identifier."""

    name: str
    config: TrainingPipelineConfig
    result: TrainingPipelineResult
    run_id: str


def select_best_candidate(
    candidates: list[CandidateExperimentResult], metric: str, direction: str
) -> CandidateExperimentResult:
    """Select a candidate using validation metrics only."""
    if not candidates:
        raise ValueError("at least one candidate result is required")
    reverse = direction == "maximize"
    return sorted(
        candidates,
        key=lambda candidate: _numeric_metric(candidate.result.metrics, metric),
        reverse=reverse,
    )[0]


def save_experiment_summary(
    path: Path,
    candidates: list[CandidateExperimentResult],
    winner: CandidateExperimentResult,
    selection_metric: str,
    direction: str,
) -> Path:
    """Persist candidate metrics, MLflow run IDs, and winner rationale."""
    payload = {
        "selection_metric": selection_metric,
        "direction": direction,
        "winner": winner.name,
        "winner_run_id": winner.run_id,
        "candidates": [_candidate_payload(candidate) for candidate in candidates],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _numeric_metric(metrics: dict[str, MetricValue], name: str) -> float:
    value = metrics.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"selection metric {name!r} is missing or non-numeric")
    return float(value)


def _candidate_payload(candidate: CandidateExperimentResult) -> dict[str, object]:
    return {
        "name": candidate.name,
        "run_id": candidate.run_id,
        "config": asdict(candidate.config.model),
        "metrics": candidate.result.metrics,
    }
