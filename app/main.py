"""
OCR FinSight - FastAPI Application
===================================
Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    uvicorn app.main:app --reload   # dev mode

Endpoints:
    POST /api/scan    — upload receipt image, get structured JSON
    GET  /api/health  — model status
    GET  /api/docs    — Swagger UI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.dependencies import startup_models
from app.routers import scan

app = FastAPI(
    title="OCR FinSight API",
    description="Receipt scanning — extracts store, date, items, and totals from receipt images.",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_event_handler("startup", startup_models)
app.include_router(scan.router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False, log_level="info")
