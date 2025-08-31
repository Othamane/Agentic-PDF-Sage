"""
Conversation model for storing chat interactions.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from sqlalchemy import Column, String, Text, DateTime, JSON, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class Conversation(Base):
    """
    Model for storing conversation interactions between user and AI.
    """
    
    __tablename__ = "conversations"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    # Conversation content
    user_message = Column(Text, nullable=False, comment="User's input message")
    ai_response = Column(Text, nullable=False, comment="AI's response")
    
    # Reasoning trace (JSON field for storing agent steps)
    reasoning_trace = Column(
        JSON,
        nullable=True,
        comment="JSON array of reasoning steps taken by the agent"
    )
    
    # Metadata
    session_id = Column(
        String(255),
        nullable=True,
        index=True,
        comment="Optional session identifier for grouping conversations"
    )
    
    user_id = Column(
        String(255),
        nullable=True,
        index=True,
        comment="Optional user identifier (for future authentication)"
    )
    
    # Performance metrics
    response_time_ms = Column(
    "response_time_ms",  # name positional
    Integer,
    nullable=True,
    comment="Response time in milliseconds"
)
    token_count = Column(
    "token_count",
    Integer,
    nullable=True,
    comment="Total tokens used for this conversation" 
)
    
    # Timestamps
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
        comment="When the conversation was created"
    )
    
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
        comment="When the conversation was last updated"
    )
    
    # Relationships
    agent_steps = relationship(
        "AgentStep",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="AgentStep.step_number"
    )
    
    retrieval_logs = relationship(
        "RetrievalLog",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="RetrievalLog.created_at"
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, created_at={self.created_at})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation to dictionary."""
        return {
            "id": str(self.id),
            "user_message": self.user_message,
            "ai_response": self.ai_response,
            "reasoning_trace": self.reasoning_trace,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "response_time_ms": self.response_time_ms,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create conversation from dictionary."""
        return cls(
            user_message=data["user_message"],
            ai_response=data["ai_response"],
            reasoning_trace=data.get("reasoning_trace"),
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            response_time_ms=data.get("response_time_ms"),
            token_count=data.get("token_count")
        )