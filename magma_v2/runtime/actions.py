"""RL action and reward logging for MAGMA v2."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def log_action(
    run_dir: str,
    agent: str,
    action: str,
    state_summary: str = "",
    input_refs: list[str] | None = None,
    output_refs: list[str] | None = None,
    success: bool = True,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path = Path(run_dir) / "actions.jsonl"
    event = {
        "timestamp": time.time(),
        "agent": agent,
        "action": action,
        "state_summary": state_summary,
        "input_refs": input_refs or [],
        "output_refs": output_refs or [],
        "success": success,
        "error": error,
        "metadata": metadata or {},
    }
    with path.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def compute_reward(auroc: float | None, tripod_score: float, reward_lambda: float = 0.5) -> float:
    predictive_score = 0.0 if auroc is None else max(0.0, min(1.0, float(auroc)))
    tripod = max(0.0, min(1.0, float(tripod_score)))
    lam = max(0.0, min(1.0, float(reward_lambda)))
    return lam * predictive_score + (1.0 - lam) * tripod

