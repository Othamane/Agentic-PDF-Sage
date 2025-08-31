"""
Agent step model for tracking reasoning steps.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from sqlalchemy import Column, String, Text, DateTime, JSON, Integer, ForeignKey, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy import Enum as SQLEnum

from app.core.database import Base


class StepType(Enum):
    """Enumeration of agent step types."""
    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    REFINEMENT = "refinement"
    ERROR = "error"


class AgentStep(Base):
    """
    Model for storing individual reasoning steps taken by the agent.
    """
    
    __tablename__ = "agent_steps"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True
    )
    
    # Foreign key to conversation
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Reference to the parent conversation"
    )
    
    # Step information
    step_number = Column(
        Integer,
        nullable=False,
        comment="Order of this step within the conversation"
    )
    
    step_type = Column(
        SQLEnum(StepType),
        nullable=False,
        index=True,
        comment="Type of reasoning step"
    )
    
    # Step content
    input_data = Column(
        JSON,
        nullable=True,
        comment="Input data for this step (JSON)"
    )
    
    output_data = Column(
        JSON,
        nullable=True,
        comment="Output data from this step (JSON)"
    )
    
    reasoning = Column(
        Text,
        nullable=True,
        comment="Human-readable description of the reasoning"
    )
    
    # Performance metrics
    duration_ms = Column(
        Float,
        nullable=True,
        comment="Duration of this step in milliseconds"
    )
    
    token_count = Column(
        Integer,
        nullable=True,
        comment="Number of tokens used in this step"
    )
    
    # Error tracking
    error_message = Column(
        Text,
        nullable=True,
        comment="Error message if step failed"
    )
    
    # Timestamps
    created_at = Column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True,
        comment="When this step was executed"
    )
    
    # Relationships
    conversation = relationship(
        "Conversation",
        back_populates="agent_steps"
    )
    
    def __repr__(self) -> str:
        return f"<AgentStep(id={self.id}, type={self.step_type}, step={self.step_number})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent step to dictionary."""
        return {
            "id": str(self.id),
            "conversation_id": str(self.conversation_id),
            "step_number": self.step_number,
            "step_type": self.step_type.value if self.step_type else None,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "reasoning": self.reasoning,
            "duration_ms": self.duration_ms,
            "token_count": self.token_count,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def create_from_dict(cls, data: Dict[str, Any]) -> "AgentStep":
        """Create agent step from dictionary."""
        return cls(
            conversation_id=data["conversation_id"],
            step_number=data["step_number"],
            step_type=StepType(data["step_type"]) if data.get("step_type") else None,
            input_data=data.get("input_data"),
            output_data=data.get("output_data"),
            reasoning=data.get("reasoning"),
            duration_ms=data.get("duration_ms"),
            token_count=data.get("token_count"),
            error_message=data.get("error_message")
        )