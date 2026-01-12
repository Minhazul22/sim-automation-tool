from __future__ import annotations

import itertools
from dataclasses import asdict
from typing import Any

import numpy as np

from app.sim.model import PID, SimParams, run_pid_thermal
from app.sim.metrics import compute_metrics

def score_metrics(metrics: dict[str, Any]) -> float:
    """
    Lower is better.
    Heavily penalize not settling (NaN/None) and large overshoot.
    """
    # Integral Absolute Error: overall tracking quality
    overshoot = float(metrics.get("overshoot_abs", 0.0))

    # Accumulated absolute error over time
    iae = float(metrics.get("iae", 0.0))

    # Time to remain within tolerance band around setpoint
    settling = metrics.get("settling_time_s", None)

    # If system never settles, apply a large penalty
    if settling is None or (isinstance(settling, float) and np.isnan(settling)):
        settling_penalty = 1e6
        settling_value = 1e6
    else:
        settling_value = float(settling)
        settling_penalty = 0.0

    # Weighted sum of performance indicators
    # NOTE: weights are heuristic and can be tuned per application
    return iae + 50.0 * overshoot + 2.0 * settling_value + settling_penalty