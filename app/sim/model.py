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
    t = np.arange(p.steps) * p.dt
    T = np.zeros(p.steps)
    u = np.zeros(p.steps)

    T[0] = p.T0
    integ = 0.0
    e_prev = 0.0

    for k in range(1, p.steps):
        e = T[k - 1] - setpoint
        integ += e * p.dt
        deriv = (e - e_prev) / p.dt

        raw_u = pid.kp * e + pid.ki * integ + pid.kd * deriv
        u[k] = float(np.clip(raw_u, pid.u_min, pid.u_max))

        dTdt = - (T[k - 1] - p.Tamb) / p.tau + p.k_u * u[k] + p.disturbance
        T[k] = T[k - 1] + p.dt * dTdt

        e_prev = e

    return {"t": t, "T": T, "u": u}
