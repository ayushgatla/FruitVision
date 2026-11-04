from fastapi import APIRouter
from .routes_inspect import router as inspect_router
from .routes_simulation import router as simulation_router
from .routes_history import router as history_router
from .routes_analytics import router as analytics_router

api_router = APIRouter(prefix="/api")
api_router.include_router(inspect_router, prefix="/inspect", tags=["Fruit Inspection"])
api_router.include_router(simulation_router, prefix="/simulation", tags=["Conveyor Simulation"])
api_router.include_router(history_router, prefix="/history", tags=["Audit History"])
api_router.include_router(analytics_router, prefix="/analytics", tags=["Analytics & Telemetry"])

