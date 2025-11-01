"""
Document model for storing uploaded PDF documents.
Final fix to resolve PostgreSQL enum case issues.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, Integer, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy import Enum as SQLEnum

from app.core.database import Base


class DocumentStatus(Enum): 
    """Enumeration of document processing statuses."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    DELETED = "deleted"


class Document(Base):
    """
    Model for storing uploaded PDF documents and their metadata.
    """
    
    __tablename__ = "documents"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    # Document metadata
    filename = Column(
        String(255),
        nullable=False,
        comment="Original filename of the uploaded document"
    )
    
    title = Column(
        String(500),
        nullable=True,
        comment="Extracted or user-provided title"
    )
    
    description = Column(
        Text,
        nullable=True,
        comment="User-provided description of the document"
    )
    
    # File information
    file_size = Column(
        Integer,
        nullable=False,
        comment="File size in bytes"
    )
    
    mime_type = Column(
        String(100),
        nullable=False,
        default="application/pdf",
        comment="MIME type of the file"
    )
    
    file_hash = Column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
        comment="SHA-256 hash of the file content for deduplication"
    )
    
    # Storage information
    storage_path = Column(
        String(500),
        nullable=False,
        comment="Path where the file is stored"
    )
    
    # Processing status - FIXED: Use lowercase default
    status = Column( 
        SQLEnum(DocumentStatus, name='document_status' , values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=DocumentStatus.UPLOADED,  # Use string value directly
        index=True,
        comment="Current processing status"
    )
    
    # Content analysis
    page_count = Column(
        Integer,
        nullable=True,
        comment="Number of pages in the document"
    )
    
    word_count = Column(
        Integer,
        nullable=True,
        comment="Estimated word count"
    )
    
    chunk_count = Column(
        Integer,
        nullable=True,
        comment="Number of text chunks created for vector search"
    )
    
    # Extracted content
    extracted_text = Column(
        Text,
        nullable=True,
        comment="Full extracted text content (for search/backup)"
    )
    
    summary = Column(
        Text,
        nullable=True,
        comment="AI-generated summary of the document"
    )
    
    keywords = Column(
        Text,
        nullable=True,
        comment="Comma-separated list of extracted keywords"
    )
    
    # Processing metadata
    processing_started_at = Column(
        DateTime,
        nullable=True,
        comment="When processing started"
    )
    
    processing_completed_at = Column(
        DateTime,
        nullable=True,
        comment="When processing completed"
    )
    
    processing_error = Column(
        Text,
        nullable=True,
        comment="Error message if processing failed"
    )
    
    # Vector store information
    vector_store_path = Column(
        String(500),
        nullable=True,
        comment="Path to the FAISS vector store file"
    )
    
    embedding_model = Column(
        String(100),
        nullable=True,
        comment="Name of the embedding model used"
    )
    
    # User information (for future multi-user support)
    user_id = Column(
        String(255),
        nullable=True,
        index=True,
        comment="Optional user identifier"
    )
    
    # Access control
    is_public = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="Whether this document is publicly accessible"
    )
    
    # Timestamps
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
        comment="When the document was uploaded"
    )
    
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        comment="When the document was last updated"
    )
    
    # Relationships
    retrieval_logs = relationship(
        "RetrievalLog",
        back_populates="document",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename={self.filename}, status={self.status})>"
    
    def _get_status_value(self) -> str:
        """Get status as string value, handling both enum and string cases."""
        if isinstance(self.status, DocumentStatus):
            return self.status.value
        elif isinstance(self.status, str):
            return self.status
        else:
            return DocumentStatus.UPLOADED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": str(self.id),
            "filename": self.filename,
            "title": self.title,
            "description": self.description,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "file_hash": self.file_hash,
            "status": self._get_status_value(),
            "page_count": self.page_count,
            "word_count": self.word_count,
            "chunk_count": self.chunk_count,
            "summary": self.summary,
            "keywords": self.keywords.split(",") if self.keywords else [],
            "processing_started_at": self.processing_started_at.isoformat() if self.processing_started_at else None,
            "processing_completed_at": self.processing_completed_at.isoformat() if self.processing_completed_at else None,
            "processing_error": self.processing_error,
            "embedding_model": self.embedding_model,
            "user_id": self.user_id,
            "is_public": self.is_public,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert document to summary dictionary (without full text)."""
        return {
            "id": str(self.id),
            "filename": self.filename,
            "title": self.title, 
            "description": self.description,
            "file_size": self.file_size,
            "status": self._get_status_value(),
            "page_count": self.page_count,
            "word_count": self.word_count,
            "chunk_count": self.chunk_count,
            "summary": self.summary,
            "keywords": self.keywords.split(",") if self.keywords else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @property
    def is_processed(self) -> bool:
        """Check if document has been successfully processed."""
        status_value = self._get_status_value()
        return status_value == DocumentStatus.PROCESSED
    
    @property
    def is_processing(self) -> bool:
        """Check if document is currently being processed."""
        status_value = self._get_status_value()
        return status_value == DocumentStatus.PROCESSING
    
    @property
    def has_error(self) -> bool:
        """Check if document processing failed."""
        status_value = self._get_status_value()
        return status_value == DocumentStatus.FAILED
    
    def mark_processing_started(self):
        """Mark document as processing started."""
        self.status = DocumentStatus.PROCESSING  # Use .value explicitly
        self.processing_started_at = datetime.utcnow()
        self.processing_error = None
    
    def mark_processing_completed(self):
        """Mark document as processing completed."""
        self.status = DocumentStatus.PROCESSED  # Use .value explicitly
        self.processing_completed_at = datetime.utcnow()
        self.processing_error = None
    
    def mark_processing_failed(self, error_message: str):
        """Mark document as processing failed."""
        self.status = DocumentStatus.FAILED  # Use .value explicitly
        self.processing_completed_at = datetime.utcnow()
        self.processing_error = error_message