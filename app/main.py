from fastapi import FastAPI

from app.api.routes import router

# Main FastAPI app for the simulation tool.
# Exposes endpoints to run thermal PID simulations
# and to sweep controller parameters for tuning.
app = FastAPI(
    title="Simulation Automation Tool",
    version="0.1.0",
)

# Attach all API routes.
# Keeping routes in a separate module makes the app easier to extend.
app.include_router(router)
