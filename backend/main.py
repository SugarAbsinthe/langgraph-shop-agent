"""FastAPI application entry point."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ensure_hf_offline_if_needed
ensure_hf_offline_if_needed()

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import chat, conversations, health

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(
    title="ShopAgent API",
    description="Backend API for the intelligent shopping guide Agent",
    version="1.0.0",
)

# CORS: allow frontend dev server during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
