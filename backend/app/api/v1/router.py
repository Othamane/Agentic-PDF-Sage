"""
API router for Agentic PDF Sage.
Main API endpoints for the application.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import chat, documents, health

# Create main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

api_router.include_router(
    documents.router,
    prefix="/documents",
    tags=["Documents"]
)

api_router.include_router(
    chat.router,
    prefix="/chat",
    tags=["Chat"]
)