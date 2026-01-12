import numpy as np

def compute_metrics(t: np.ndarray, T: np.ndarray, setpoint: float, band: float = 0.5) -> dict:
    overshoot = float(np.max(np.abs(T - setpoint)))
    iae = float(np.trapezoid(np.abs(setpoint - T), t))


    settling_time = float("nan")
    within = np.abs(T - setpoint) <= band
    for i in range(len(T)):
        if within[i] and within[i:].all():
            settling_time = float(t[i])
            break

    return {
        "overshoot_abs": overshoot,
        "iae": iae,
        "settling_time_s": settling_time,
    }
