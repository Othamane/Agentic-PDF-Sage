"""
Database configuration and session management.
Fixed to properly handle PostgreSQL enum types.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Convert PostgreSQL URL to async format
async_database_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create async engine
async_engine = create_async_engine(
    async_database_url,
    echo=settings.DEBUG,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=3600,  # Recycle connections every hour
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Create sync engine for migrations
sync_engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# Create sync session factory for migrations
SessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)

# Base class for models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session for FastAPI dependency injection.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions in services.
    Use this in service classes instead of get_db().
    """
    session = AsyncSessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def create_db_session() -> AsyncSession:
    """
    Create a new database session for manual management.
    Remember to close the session when done.
    """
    return AsyncSessionLocal()


async def create_enum_types():
    """Create PostgreSQL enum types if they don't exist."""
    try:
        async with async_engine.begin() as conn:
            # Drop existing enum types if they exist to recreate them
            await conn.execute(text("DROP TYPE IF EXISTS documentstatus CASCADE;"))
            await conn.execute(text("DROP TYPE IF EXISTS steptype CASCADE;"))
            
            # Create document_status enum with correct values
            await conn.execute(text("""
                CREATE TYPE documentstatus AS ENUM ('uploaded', 'processing', 'processed', 'failed', 'deleted');
            """))
            
            # Create step_type enum with correct values
            await conn.execute(text("""
                CREATE TYPE steptype AS ENUM ('planning', 'retrieval', 'synthesis', 'validation', 'refinement', 'error');
            """))
            
            logger.info("Enum types created successfully")
            
    except Exception as e:
        logger.error(f"Failed to create enum types: {e}")
        raise


async def init_db() -> None:
    """
    Initialize database tables and enum types.
    """
    try:
        # First create enum types
        await create_enum_types()
        
        # Import all models here to ensure they're registered
        from app.models import conversation, document, agent_step, retrieval_log  # noqa
        
        async with async_engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def close_db() -> None:
    """
    Close database connections.
    """
    try:
        await async_engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


# Database event listeners for connection management
@event.listens_for(sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set database pragmas on connect (for SQLite, if used)."""
    if "sqlite" in settings.DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


@event.listens_for(sync_engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    """Log SQL queries in debug mode."""
    if settings.DEBUG:
        logger.debug(f"SQL Query: {statement}")
        logger.debug(f"Parameters: {parameters}")