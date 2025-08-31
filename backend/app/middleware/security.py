"""
Security middleware for Agentic PDF Sage.
Implements various security headers and protections.
"""

import time
import secrets
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    """
    
    def __init__(self, app, csp_nonce: bool = True):
        super().__init__(app)
        self.csp_nonce = csp_nonce
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate nonce for CSP if enabled
        nonce = secrets.token_urlsafe(16) if self.csp_nonce else None
        
        # Store nonce in request state for use in templates
        if nonce:
            request.state.csp_nonce = nonce
        
        response = await call_next(request)
        
        # Security Headers
        security_headers = {
            # Prevent clickjacking
            "X-Frame-Options": "DENY",
            
            # Prevent MIME type sniffing
            "X-Content-Type-Options": "nosniff",
            
            # Enable XSS protection
            "X-XSS-Protection": "1; mode=block",
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions policy (formerly Feature Policy)
            "Permissions-Policy": (
                "camera=(), "
                "microphone=(), "
                "geolocation=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "accelerometer=()"
            ),
            
            # Prevent browsers from inferring content types
            "X-Download-Options": "noopen",
            
            # Control DNS prefetching
            "X-DNS-Prefetch-Control": "off",
            
            # HSTS (only in production with HTTPS)
            # This will be handled by reverse proxy in production
            
            # Content Security Policy
            "Content-Security-Policy": self._build_csp(nonce),
        }
        
        # Add security headers to response
        for header, value in security_headers.items():
            response.headers[header] = value
        
        return response
    
    def _build_csp(self, nonce: str = None) -> str:
        """
        Build Content Security Policy header.
        """
        csp_parts = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline'" + (f" 'nonce-{nonce}'" if nonce else ""),
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
            "font-src 'self' https://fonts.gstatic.com",
            "img-src 'self' data: blob:",
            "connect-src 'self' https://api.openai.com",
            "frame-src 'none'",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
            "upgrade-insecure-requests",
        ]
        
        return "; ".join(csp_parts)


class RequestValidationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for basic request validation and sanitization.
    """
    
    def __init__(self, app, max_request_size: int = 50 * 1024 * 1024):  # 50MB
        super().__init__(app)
        self.max_request_size = max_request_size
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check request size
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                content_length = int(content_length)
                if content_length > self.max_request_size:
                    return Response(
                        content="Request too large",
                        status_code=413,
                        headers={"Content-Type": "text/plain"}
                    )
            except ValueError:
                return Response(
                    content="Invalid Content-Length header",
                    status_code=400,
                    headers={"Content-Type": "text/plain"}
                )
        
        # Validate User-Agent (basic bot detection)
        user_agent = request.headers.get("user-agent", "").lower()
        suspicious_agents = [
            "sqlmap", "nmap", "nikto", "masscan", "nessus",
            "acunetix", "burp", "zap", "w3af"
        ]
        
        if any(agent in user_agent for agent in suspicious_agents):
            return Response(
                content="Forbidden",
                status_code=403,
                headers={"Content-Type": "text/plain"}
            )
        
        # Add request start time for performance monitoring
        request.state.start_time = time.time()
        
        response = await call_next(request)
        
        # Add response time header (for monitoring)
        if hasattr(request.state, "start_time"):
            process_time = time.time() - request.state.start_time
            response.headers["X-Process-Time"] = str(process_time)
        
        return response