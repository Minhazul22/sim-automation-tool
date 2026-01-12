from fastapi import FastAPI

from app.api.routes import router

# Create FastAPI application instance
# This application exposes REST endpoints for:
# - running single thermal PID simulations
# - performing automated PID parameter sweeps
app = FastAPI(
    title="Simulation Automation Tool",
    version="0.1.0",
)

# Register API routes defined in app/api/routes.py
# Using a router keeps the application modular and scalable
app.include_router(router)
