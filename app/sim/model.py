from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimParams:
    # Simulation time step (seconds)
    dt: float = 0.1

    # Number of simulation steps
    steps: int = 600

    # Initial temperature
    T0: float = 20.0

    # Ambient temperature
    Tamb: float = 25.0

    # Thermal time constant
    tau: float = 30.0

    # Control gain (negative = cooling effect)
    k_u: float = -0.8

    # External disturbance (optional)
    disturbance: float = 0.0


@dataclass(frozen=True)
class PID:
    # PID gains
    kp: float
    ki: float
    kd: float

    # Actuator limits
    u_min: float = 0.0
    u_max: float = 1.0


def run_pid_thermal(setpoint: float, pid: PID, p: SimParams) -> dict[str, np.ndarray]:
    # Time vector
    t = np.arange(p.steps) * p.dt

    # Temperature and control arrays
    T = np.zeros(p.steps)
    u = np.zeros(p.steps)

    # Initial temperature
    T[0] = p.T0

    # PID internal state
    integ = 0.0
    e_prev = 0.0

    for k in range(1, p.steps):
        # Control error (measured - desired)
        e = T[k - 1] - setpoint

        # Error derivative
        deriv = (e - e_prev) / p.dt

        # Raw PID output
        raw_u = pid.kp * e + pid.ki * integ + pid.kd * deriv

        # Clamp control signal to actuator limits
        u_clamped = float(np.clip(raw_u, pid.u_min, pid.u_max))
        u[k] = u_clamped

        # Anti-windup:
        # Only integrate error if the actuator is not saturated
        if u_clamped == raw_u:
            integ += e * p.dt

        # Simple first-order thermal model
        dTdt = (
            -(T[k - 1] - p.Tamb) / p.tau
            + p.k_u * u[k]
            + p.disturbance
        )

        # Update temperature
        T[k] = T[k - 1] + p.dt * dTdt

        e_prev = e

    return {"t": t, "T": T, "u": u}