"""
Health check endpoints for monitoring and diagnostics.
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.database import async_engine
from app.services.free_llm_service import free_llm_service, free_embedding_service

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()


class HealthStatus(BaseModel):
    """Health status response model."""
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    version: str = "1.0.0"
    environment: str


class DetailedHealthStatus(HealthStatus):
    """Detailed health status with component checks."""
    components: Dict[str, Dict[str, Any]]
    system_info: Dict[str, Any]
    uptime_seconds: float


@router.get("/", response_model=HealthStatus)
async def basic_health_check():
    """
    Basic health check endpoint for load balancers.
    """
    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        environment=settings.ENVIRONMENT
    )


@router.get("/detailed", response_model=DetailedHealthStatus)
async def detailed_health_check():
    """
    Detailed health check with component status.
    """
    start_time = time.time()
    components = {}
    overall_status = "healthy"
    
    # Check database connectivity
    db_status = await _check_database()
    components["database"] = db_status
    if db_status["status"] != "healthy":
        overall_status = "degraded"
    
    # Check LLM service
    llm_status = await _check_llm_service()
    components["llm_service"] = llm_status
    if llm_status["status"] != "healthy":
        overall_status = "degraded"
    
    # Check embedding service
    embedding_status = await _check_embedding_service()
    components["embedding_service"] = embedding_status
    if embedding_status["status"] != "healthy":
        overall_status = "degraded"
    
    # Check file system
    fs_status = await _check_file_system()
    components["file_system"] = fs_status
    if fs_status["status"] != "healthy":
        overall_status = "degraded"
    
    # Get system information
    system_info = _get_system_info()
    
    # Calculate check duration
    check_duration = (time.time() - start_time) * 1000
    
    return DetailedHealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        environment=settings.ENVIRONMENT,
        components=components,
        system_info=system_info,
        uptime_seconds=check_duration
    )


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    """
    return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint.
    """
    try:
        # Quick checks for readiness
        db_healthy = await _quick_db_check()
        llm_available = await _quick_llm_check()
        
        if db_healthy and llm_available:
            return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready"
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


# Component health check functions

async def _check_database() -> Dict[str, Any]:
    """Check database connectivity and performance."""
    start_time = time.time()
    
    try:
        # Test database connection
        async with async_engine.begin() as conn:
            result = await conn.execute("SELECT 1")
            row = result.fetchone()
            
            if row and row[0] == 1:
                duration_ms = (time.time() - start_time) * 1000
                return {
                    "status": "healthy",
                    "response_time_ms": round(duration_ms, 2),
                    "message": "Database connection successful"
                }
            else:
                return {
                    "status": "unhealthy",
                    "message": "Database query returned unexpected result"
                }
                
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "unhealthy",
            "response_time_ms": round(duration_ms, 2),
            "message": f"Database connection failed: {str(e)}"
        }


async def _check_llm_service() -> Dict[str, Any]:
    """Check LLM service availability."""
    start_time = time.time()
    
    try:
        # Test LLM with a simple query
        test_response = await free_llm_service.generate_response(
            "Hello", 
            max_tokens=5
        )
        
        duration_ms = (time.time() - start_time) * 1000
        
        if test_response and isinstance(test_response, str):
            return {
                "status": "healthy",
                "response_time_ms": round(duration_ms, 2),
                "message": "LLM service responding",
                "provider": settings.LLM_PROVIDER,
                "model": settings.LLM_MODEL
            }
        else:
            return {
                "status": "degraded",
                "response_time_ms": round(duration_ms, 2),
                "message": "LLM service returned empty response"
            }
            
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.warning(f"LLM health check failed: {e}")
        return {
            "status": "unhealthy",
            "response_time_ms": round(duration_ms, 2),
             "message": "LLM service NOT responding",
            "Error" : e,
        }