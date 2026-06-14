"""FastAPI application entry point."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ensure_hf_offline_if_needed
ensure_hf_offline_if_needed()

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import chat, conversations, health
from backend.logging_config import set_request_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="ShopAgent API",
    description="Backend API for the intelligent shopping guide Agent",
    version="1.0.0",
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Inject request_id into context and response header."""
    rid = request.headers.get("X-Request-ID", "")
    set_request_id(rid)
    response = await call_next(request)
    from backend.logging_config import get_request_id
    response.headers["X-Request-ID"] = get_request_id()
    return response


# CORS: allow frontend dev server during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

# Register routers
app.include_router(chat.router, prefix="/api")
app.include_router(conversations.router, prefix="/api")
app.include_router(health.router, prefix="/api")


@app.on_event("startup")
async def startup():
    """Log startup — agent is lazily initialized on first request."""
    logging.info("ShopAgent API server started.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
