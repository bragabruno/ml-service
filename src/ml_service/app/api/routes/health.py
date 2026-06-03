from __future__ import annotations

import time

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    uptime = time.monotonic() - getattr(request.app.state, "start_time", time.monotonic())
    return {
        "status": "healthy",
        "service": "ml-service",
        "version": "0.1.0",
        "uptime_seconds": round(uptime, 2),
        "ready": getattr(request.app.state, "ready", False),
    }


@router.get("/health/ready")
async def readiness(request: Request) -> dict[str, object]:
    ready = getattr(request.app.state, "ready", False)
    return {"ready": ready}


@router.get("/health/live")
async def liveness() -> dict[str, str]:
    return {"status": "alive"}
