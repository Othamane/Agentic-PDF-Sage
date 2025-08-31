"""
Logging configuration for Agentic PDF Sage.
Structured logging with JSON output for production.
"""

import json
import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from app.core.config import get_settings

settings = get_settings()


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        
        # Basic log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                'filename', 'module', 'lineno', 'funcName', 'created',
                'msecs', 'relativeCreated', 'thread', 'threadName',
                'processName', 'process', 'getMessage', 'exc_info',
                'exc_text', 'stack_info'
            ]:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class ColoredFormatter(logging.Formatter):
    """
    Colored formatter for development console output.
    """
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        
        # Add color to level name
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Build log message
        log_parts = [
            f"{color}{record.levelname:8}{reset}",
            f"{timestamp}",
            f"{record.name:20}",
            f"{record.getMessage()}"
        ]
        
        # Add exception info if present
        if record.exc_info:
            log_parts.append(f"\n{self.formatException(record.exc_info)}")
        
        return " | ".join(log_parts)


def setup_logging():
    """
    Setup logging configuration based on environment.
    """
    
    # Ensure log directory exists
    log_dir = Path(settings.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Base logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": JSONFormatter,
            },
            "colored": {
                "()": ColoredFormatter,
            },
            "simple": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "colored" if settings.is_development else "json",
                "stream": sys.stdout
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.LOG_LEVEL,
                "formatter": "json",
                "filename": settings.LOG_FILE,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8"
            }
        },
        "loggers": {
            "app": {
                "level": settings.LOG_LEVEL,
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["file"],  # Only log access logs to file
                "propagate": False
            },
            "sqlalchemy.engine": {
                "level": "WARNING",  # Reduce SQLAlchemy noise
                "handlers": ["file"],
                "propagate": False
            },
            "transformers": {
                "level": "WARNING",  # Reduce transformers noise
                "handlers": ["file"],
                "propagate": False
            },
            "sentence_transformers": {
                "level": "WARNING",  # Reduce sentence-transformers noise
                "handlers": ["file"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console", "file"]
        }
    }
    
    # Apply logging configuration
    logging.config.dictConfig(config)
    
    # Log startup message
    logger = logging.getLogger("app.core.logging")
    logger.info(
        "Logging configured",
        extra={
            "environment": settings.ENVIRONMENT,
            "log_level": settings.LOG_LEVEL,
            "log_file": settings.LOG_FILE
        }
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"app.{name}")


class LoggerMixin:
    """
    Mixin class to add logging capability to any class.
    """
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__module__ + "." + self.__class__.__name__)


def log_function_call(func):
    """
    Decorator to log function calls with parameters and results.
    """
    import functools
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function entry
        logger.debug(
            f"Calling {func.__name__}",
            extra={
                "function": func.__name__,
                "args_count": len(args),
                "kwargs": list(kwargs.keys())
            }
        )
        
        try:
            # Call function
            result = await func(*args, **kwargs)
            
            # Log successful completion
            logger.debug(
                f"Completed {func.__name__}",
                extra={
                    "function": func.__name__,
                    "success": True
                }
            )
            
            return result
            
        except Exception as e:
            # Log exception
            logger.error(
                f"Error in {func.__name__}: {e}",
                extra={
                    "function": func.__name__,
                    "success": False,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # Log function entry
        logger.debug(
            f"Calling {func.__name__}",
            extra={
                "function": func.__name__,
                "args_count": len(args),
                "kwargs": list(kwargs.keys())
            }
        )
        
        try:
            # Call function
            result = func(*args, **kwargs)
            
            # Log successful completion
            logger.debug(
                f"Completed {func.__name__}",
                extra={
                    "function": func.__name__,
                    "success": True
                }
            )
            
            return result
            
        except Exception as e:
            # Log exception
            logger.error(
                f"Error in {func.__name__}: {e}",
                extra={
                    "function": func.__name__,
                    "success": False,
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            raise
    
    # Return appropriate wrapper based on function type
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# Security-focused logging helpers

def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize log data to remove sensitive information.
    """
    sensitive_keys = {
        'password', 'token', 'secret', 'key', 'api_key',
        'authorization', 'auth', 'credentials', 'private'
    }
    
    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()
        
        # Check if key contains sensitive information
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value)
        elif isinstance(value, list):
            sanitized[key] = [
                sanitize_log_data(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value
    
    return sanitized


def log_security_event(
    event_type: str,
    description: str,
    user_id: str = None,
    ip_address: str = None,
    additional_data: Dict[str, Any] = None
):
    """
    Log security-related events.
    """
    logger = get_logger("security")
    
    event_data = {
        "event_type": event_type,
        "description": description,
        "user_id": user_id,
        "ip_address": ip_address,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if additional_data:
        event_data.update(sanitize_log_data(additional_data))
    
    logger.warning(
        f"Security event: {event_type}",
        extra=event_data
    )