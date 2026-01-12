from fastapi import APIRouter
from pydantic import BaseModel
from app.sim.model import PID, SimParams, run_pid_thermal
from app.sim.metrics import compute_metrics

router = APIRouter()

class RunRequest(BaseModel):
    setpoint: float
    kp: float
    ki: float
    kd: float

class RunResponse(BaseModel):
    metrics: dict
    t: list[float]
    T: list[float]
    u: list[float]

@router.post("/simulations/run", response_model=RunResponse)
def run_simulation(req: RunRequest):
    pid = PID(kp=req.kp, ki=req.ki, kd=req.kd)
    params = SimParams()

    out = run_pid_thermal(req.setpoint, pid, params)
    metrics = compute_metrics(out["t"], out["T"], req.setpoint)

    return {
        "metrics": metrics,
        "t": out["t"].tolist(),
        "T": out["T"].tolist(),
        "u": out["u"].tolist(),
    }
