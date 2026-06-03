from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import detection, health
from app.core.config import settings

app = FastAPI(
    title="Backgammon Detection API",
    version="0.1.0",
    description="API for detecting backgammon board state from video files",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(detection.router, prefix="/api", tags=["detection"])
