"""
Retrieval log model for tracking document chunk retrievals.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import Column, String, Text, DateTime, Float, ForeignKey , Integer , Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class RetrievalLog(Base): 
    """
    Model for logging document chunk retrievals during conversations.
    Helps track which parts of documents are being used to answer questions.
    """
    
    __tablename__ = "retrieval_logs"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    # Foreign keys
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the conversation"
    )
    
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the source document"
    )
    
    # Retrieved content
    chunk_content = Column(
        Text,
        nullable=False,
        comment="The actual text chunk that was retrieved"
    )
    
    chunk_index = Column(
        "chunk_index",
        Integer,
        nullable=True,
        comment="Index of the chunk within the document"
    )
    
    # Relevance scoring
    relevance_score = Column(
        Float,
        nullable=True,
        comment="Relevance score from vector similarity search"
    )
    
    embedding_distance = Column(
        Float,
        nullable=True,
        comment="Distance in embedding space"
    )
    
    # Search metadata
    search_query = Column(
        String(1000),
        nullable=True,
        comment="The query used to retrieve this chunk"
    )
    
    retrieval_method = Column(
        String(50),
        nullable=True,
        default="vector_similarity",
        comment="Method used for retrieval (vector_similarity, keyword, etc.)"
    )
    
    # Position information
    page_number = Column(
        "page_number",
        Integer,
        nullable=True,
        comment="Page number where this chunk appears in the document"
    )
    
    character_start = Column(
        "character_start",
        Integer,
        nullable=True,
        comment="Starting character position in the document"
    )
    
    character_end = Column(
        "character_end",
        Integer,
        nullable=True,
        comment="Ending character position in the document"
    )
    
    # Usage tracking
    was_used_in_response = Column( 
        "was_used_in_response",
        Boolean, 
        nullable=True,
        default=True,
        comment="Whether this chunk was actually used in the final response"
    )
    
    # Timestamps
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
        comment="When this chunk was retrieved"
    )
    
    # Relationships
    conversation = relationship(
        "Conversation",
        back_populates="retrieval_logs"
    )
    
    document = relationship(
        "Document",
        back_populates="retrieval_logs"
    )
    
    def __repr__(self) -> str:
        return f"<RetrievalLog(id={self.id}, doc={self.document_id}, score={self.relevance_score})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert retrieval log to dictionary."""
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "document_id": str(self.document_id),
            "chunk_content": self.chunk_content,
            "chunk_index": self.chunk_index,
            "relevance_score": self.relevance_score,
            "embedding_distance": self.embedding_distance,
            "search_query": self.search_query,
            "retrieval_method": self.retrieval_method,
            "page_number": self.page_number,
            "character_start": self.character_start,
            "character_end": self.character_end,
            "was_used_in_response": self.was_used_in_response,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def to_source_dict(self) -> Dict[str, Any]:
        """Convert to source citation format."""
        return {
            "document_id": str(self.document_id),
            "content": self.chunk_content[:200] + "..." if len(self.chunk_content) > 200 else self.chunk_content,
            "relevance_score": self.relevance_score,
            "page_number": self.page_number,
            "retrieval_method": self.retrieval_method
        }
    
    @classmethod
    def create_from_chunk(
        cls,
        conversation_id: uuid.UUID,
        document_id: uuid.UUID,
        chunk_content: str,
        relevance_score: Optional[float] = None,
        search_query: Optional[str] = None,
        **kwargs
    ) -> "RetrievalLog":
        """Create retrieval log from chunk data."""
        return cls(
            conversation_id=conversation_id,
            document_id=document_id,
            chunk_content=chunk_content,
            relevance_score=relevance_score,
            search_query=search_query,
            **kwargs
        )