from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimParams:
    dt: float = 0.1
    steps: int = 600
    T0: float = 20.0
    Tamb: float = 25.0
    tau: float = 30.0
    k_u: float = -0.8
    disturbance: float = 0.0


@dataclass(frozen=True)
class PID:
    kp: float
    ki: float
    kd: float
    u_min: float = 0.0
    u_max: float = 1.0


def run_pid_thermal(setpoint: float, pid: PID, p: SimParams) -> dict[str, np.ndarray]:
    # Basic guards (avoid weird inputs)
    if p.dt <= 0:
        raise ValueError("dt must be > 0")
    if p.steps < 2:
        raise ValueError("steps must be >= 2")

    t = np.arange(p.steps) * p.dt
    T = np.zeros(p.steps, dtype=float)
    u = np.zeros(p.steps, dtype=float)

    T[0] = p.T0
    integ = 0.0
    e_prev = 0.0

    for k in range(1, p.steps):
        # Error sign chosen for cooling: if T > setpoint, error is positive
        e = T[k - 1] - setpoint
        deriv = (e - e_prev) / p.dt

        # Compute control action then clamp to actuator limits
        raw_u = pid.kp * e + pid.ki * integ + pid.kd * deriv
        u_clamped = float(np.clip(raw_u, pid.u_min, pid.u_max))
        u[k] = u_clamped

        # Anti-windup: only integrate when we are not saturated
        # (use tolerance to avoid float equality edge cases)
        if abs(u_clamped - raw_u) < 1e-12:
            integ += e * p.dt

        # First-order thermal model: ambient pull + actuator cooling/heating + disturbance
        dTdt = -(T[k - 1] - p.Tamb) / p.tau + p.k_u * u[k] + p.disturbance
        T[k] = T[k - 1] + p.dt * dTdt

        e_prev = e

    return {"t": t, "T": T, "u": u}
