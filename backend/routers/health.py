"""Health check endpoint."""

import logging

from fastapi import APIRouter

from backend.dependencies import get_component_status
from backend.schemas import ComponentStatus, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health():
    """Return the health status of all backend components.

    Components checked: LLM, ChromaDB, Redis, SQLite database.
    """
    components = get_component_status()
    all_ok = all(components.values())

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        components=ComponentStatus(**components),
    )
