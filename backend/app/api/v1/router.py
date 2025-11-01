"""
API router for Agentic PDF Sage.
Main API endpoints for the application.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import health
from app.api.v1.endpoints import enhanced_chat_endpoints, enhanced_document_endpoints

# Create main API router
api_router = APIRouter()

# Include endpoint routers
api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

api_router.include_router(
    enhanced_document_endpoints.router,
    prefix="/documents",
    tags=["Documents"]
)

api_router.include_router(
    enhanced_chat_endpoints.router,
    prefix="/chat",
    tags=["Chat"]
)