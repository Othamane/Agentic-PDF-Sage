"""
Rate limiting middleware for Agentic PDF Sage.
Simple in-memory rate limiting implementation.
"""

import time
import asyncio
from typing import Callable, Dict, Tuple
from collections import defaultdict, deque
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from app.core.config import get_settings

settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple sliding window rate limiting middleware.
    Uses in-memory storage - for production, consider Redis.
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = None,
        window_size: int = None,
        cleanup_interval: int = 300  # 5 minutes
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute or settings.RATE_LIMIT_REQUESTS
        self.window_size = window_size or settings.RATE_LIMIT_WINDOW
        self.cleanup_interval = cleanup_interval
        
        # Storage for request timestamps per IP
        self.request_store: Dict[str, deque] = defaultdict(deque)
        self.last_cleanup = time.time()
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health checks and static files
        if self._should_skip_rate_limiting(request):
            return await call_next(request)
        
        client_ip = self._get_client_ip(request)
        current_time = time.time()
        
        async with self.lock:
            # Cleanup old entries periodically
            if current_time - self.last_cleanup > self.cleanup_interval:
                await self._cleanup_old_entries(current_time)
                self.last_cleanup = current_time
            
            # Check rate limit for this IP
            if await self._is_rate_limited(client_ip, current_time):
                return self._create_rate_limit_response(client_ip, current_time)
            
            # Record this request
            self.request_store[client_ip].append(current_time)
        
        response = await call_next(request)
        
        # Add rate limit headers
        remaining, reset_time = await self._get_rate_limit_info(client_ip, current_time)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(reset_time))
        
        return response
    
    def _should_skip_rate_limiting(self, request: Request) -> bool:
        """Check if request should skip rate limiting."""
        skip_paths = ["/health", "/metrics", "/static/"]
        path = request.url.path
        
        return any(path.startswith(skip_path) for skip_path in skip_paths)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (when behind reverse proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        # Fallback to direct connection IP
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    async def _is_rate_limited(self, client_ip: str, current_time: float) -> bool:
        """Check if client IP is rate limited."""
        requests = self.request_store[client_ip]
        
        # Remove old requests outside the window
        cutoff_time = current_time - self.window_size
        while requests and requests[0] <= cutoff_time:
            requests.popleft()
        
        # Check if limit exceeded
        return len(requests) >= self.requests_per_minute
    
    async def _get_rate_limit_info(self, client_ip: str, current_time: float) -> Tuple[int, float]:
        """Get rate limit info for headers."""
        requests = self.request_store[client_ip]
        
        # Remove old requests outside the window
        cutoff_time = current_time - self.window_size
        while requests and requests[0] <= cutoff_time:
            requests.popleft()
        
        remaining = max(0, self.requests_per_minute - len(requests))
        
        # Calculate reset time (when oldest request expires)
        if requests:
            reset_time = requests[0] + self.window_size
        else:
            reset_time = current_time + self.window_size
        
        return remaining, reset_time
    
    def _create_rate_limit_response(self, client_ip: str, current_time: float) -> JSONResponse:
        """Create rate limit exceeded response."""
        remaining, reset_time = asyncio.create_task(
            self._get_rate_limit_info(client_ip, current_time)
        ).result()
        
        retry_after = int(reset_time - current_time)
        
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": f"Too many requests. Try again in {retry_after} seconds.",
                "retry_after": retry_after,
                "limit": self.requests_per_minute,
                "window": self.window_size
            },
            headers={
                "X-RateLimit-Limit": str(self.requests_per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(reset_time)),
                "Retry-After": str(retry_after)
            }
        )
    
    async def _cleanup_old_entries(self, current_time: float) -> None:
        """Clean up old request entries to prevent memory leaks."""
        cutoff_time = current_time - (self.window_size * 2)  # Keep some buffer
        
        # Remove completely expired IPs
        expired_ips = []
        for ip, requests in self.request_store.items():
            if not requests or requests[-1] <= cutoff_time:
                expired_ips.append(ip)
            else:
                # Clean old requests for this IP
                while requests and requests[0] <= cutoff_time:
                    requests.popleft()
        
        # Remove expired IPs
        for ip in expired_ips:
            del self.request_store[ip]


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    Middleware to whitelist specific IP addresses.
    Useful for allowing certain IPs to bypass rate limiting.
    """
    
    def __init__(self, app, whitelist: list = None):
        super().__init__(app)
        self.whitelist = set(whitelist or [])
        
        # Add localhost and common internal IPs
        self.whitelist.update([
            "127.0.0.1",
            "::1",
            "localhost"
        ])
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = self._get_client_ip(request)
        
        # Mark whitelisted IPs in request state
        if client_ip in self.whitelist:
            request.state.whitelisted_ip = True
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"