from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.sim.model import PID, SimParams, run_pid_thermal
from app.sim.metrics import compute_metrics

router = APIRouter()


class RunRequest(BaseModel):
    # Controller target and PID gains
    setpoint: float = Field(..., description="Target temperature (°C)")
    kp: float = Field(..., description="P gain")
    ki: float = Field(..., description="I gain")
    kd: float = Field(..., description="D gain")

    # Model settings (optional). Useful when you want to try different plant behaviour.
    dt: float = Field(0.1, description="Time step (s)")
    steps: int = Field(600, description="Number of steps")
    T0: float = Field(20.0, description="Start temperature (°C)")
    Tamb: float = Field(25.0, description="Ambient temperature (°C)")
    tau: float = Field(30.0, description="Time constant (s)")
    k_u: float = Field(-0.8, description="Actuator effect (negative = cooling)")
    disturbance: float = Field(0.0, description="Extra heat load (constant)")


class RunResponse(BaseModel):
    metrics: dict
    t: list[float]
    T: list[float]
    u: list[float]


@router.post("/simulations/run", response_model=RunResponse)
def run_simulation(
    req: RunRequest,
    # By default we return only the final values to keep responses small.
    include_series: bool = Query(False, description="Include full time series arrays in response"),
):
    # Build PID + model params from the request
    pid = PID(kp=req.kp, ki=req.ki, kd=req.kd)
    params = SimParams(
        dt=req.dt,
        steps=req.steps,
        T0=req.T0,
        Tamb=req.Tamb,
        tau=req.tau,
        k_u=req.k_u,
        disturbance=req.disturbance,
    )

    # Run simulation and compute performance numbers
    out = run_pid_thermal(setpoint=req.setpoint, pid=pid, p=params)
    metrics = compute_metrics(out["t"], out["T"], setpoint=req.setpoint)

    # Always return metrics. Time series is optional.
    response = {"metrics": metrics}

    if include_series:
        response.update(
            {
                "t": out["t"].tolist(),
                "T": out["T"].tolist(),
                "u": out["u"].tolist(),
            }
        )
    else:
        # Small response: only the last sample (still kept as a list for the schema)
        response.update(
            {
                "t": [float(out["t"][-1])],
                "T": [float(out["T"][-1])],
                "u": [float(out["u"][-1])],
            }
        )

    return response
