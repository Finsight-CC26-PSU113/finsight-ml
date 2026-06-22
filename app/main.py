"""
OCR FinSight - FastAPI Application
===================================
Production-ready receipt scanning API.

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    uvicorn app.main:app --reload   # dev mode

Endpoints:
    POST /api/v1/predict  — upload receipt image, get structured JSON
    GET  /api/v1/health   — model status
    GET  /api/docs        — Swagger UI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.deps import shutdown_models, startup_models
from app.api.v1.router import api_router
from app.core.config import settings

app = FastAPI(
    title="OCR FinSight API",
    description="Receipt scanning — extracts store, date, items, and totals from receipt images.",
    version="3.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# ── Middleware ──────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lifespan events ─────────────────────────────────────────────────────────

app.add_event_handler("startup", startup_models)
app.add_event_handler("shutdown", shutdown_models)

# ── Routers ─────────────────────────────────────────────────────────────────

app.include_router(api_router, prefix=settings.API_V1_PREFIX)


# ── Root redirect ───────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "OCR FinSight API v3.0", "docs": "/api/docs"}


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level="info",
    )
