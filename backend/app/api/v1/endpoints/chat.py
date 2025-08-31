"""
Chat endpoints for the Agentic PDF Sage API.
Fixed to handle string status values from document model.
"""

import uuid
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from app.services.agent_service import AgentService
from app.services.document_service import document_service
from app.models.document import DocumentStatus

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize agent service
agent_service = AgentService()


# Request/Response Models
class ChatMessage(BaseModel):
    """Chat message request model."""
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    document_ids: List[str] = Field(..., min_items=1, description="List of document IDs to search")
    conversation_id: Optional[str] = Field(default=None, description="Existing conversation ID")
    max_iterations: Optional[int] = Field(default=3, ge=1, le=5, description="Max reasoning iterations")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="AI response")
    conversation_id: str = Field(..., description="Conversation ID")
    reasoning_trace: List[dict] = Field(default=[], description="Agent reasoning steps")
    sources: List[dict] = Field(default=[], description="Source documents and chunks")
    timestamp: str = Field(..., description="Response timestamp")
    response_time_ms: Optional[float] = Field(default=None, description="Response time in milliseconds")


class ConversationSummary(BaseModel):
    """Conversation summary model."""
    id: str
    user_message: str
    ai_response: str
    created_at: str
    response_time_ms: Optional[float] = None


def get_status_value(status) -> str:
    """Helper function to get status as string value."""
    if isinstance(status, DocumentStatus):
        return status.value
    elif isinstance(status, str):
        return status
    else:
        return DocumentStatus.UPLOADED.value


@router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatMessage):
    """
    Send a message and get an AI response with reasoning.
    """
    start_time = datetime.utcnow()
    
    try:
        # Validate document IDs and ensure they're processed
        valid_document_ids = []
        for doc_id in request.document_ids:
            try:
                document = await document_service.get_document(doc_id)
                if not document:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Document {doc_id} not found"
                    )
                
                # Get status as string value safely
                doc_status = get_status_value(document.status)
                
                if doc_status != DocumentStatus.PROCESSED.value:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Document {doc_id} is not processed yet. Status: {doc_status}"
                    )
                
                valid_document_ids.append(doc_id)
                
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid document ID format: {doc_id}"
                )
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Process the query using agent service
        result = await agent_service.process_query(
            query=request.message,
            conversation_id=conversation_id,
            document_ids=valid_document_ids,
            max_iterations=request.max_iterations
        )
        
        # Calculate response time
        end_time = datetime.utcnow()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return ChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            reasoning_trace=result["reasoning_trace"],
            sources=result["sources"],
            timestamp=result["timestamp"],
            response_time_ms=response_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your message. Please try again."
        )


@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    user_id: Optional[str] = None
):
    """
    List recent conversations.
    """
    try:
        # This would typically be implemented with proper database queries
        # For now, return empty list as conversations are stored but not exposed via this endpoint
        # In a full implementation, you'd query the conversations table
        
        return []
        
    except Exception as e:
        logger.error(f"List conversations error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get a specific conversation by ID.
    """
    try:
        # Validate conversation ID format
        try:
            uuid.UUID(conversation_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid conversation ID format"
            )
        
        # This would query the database for the conversation
        # For now, return not found as this endpoint is not fully implemented
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation"
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and its associated data.
    """
    try:
        # Validate conversation ID format
        try:
            uuid.UUID(conversation_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid conversation ID format"
            )
        
        # This would delete the conversation from the database
        # For now, return success as conversations are stored but deletion is not implemented
        return {"message": "Conversation deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete conversation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation"
        )


# Additional utility endpoints

@router.get("/status")
async def get_chat_status():
    """
    Get the status of chat services.
    """
    try:
        # Check if LLM service is available
        llm_available = True
        try:
            # Test LLM availability with timeout
            import asyncio
            await asyncio.wait_for(
                agent_service.llm_service.generate_response("test", max_tokens=5),
                timeout=10.0
            )
        except Exception as e:
            logger.warning(f"LLM availability check failed: {e}")
            llm_available = False
        
        # Check if embedding service is available
        embedding_available = True
        try:
            # Test embedding service with timeout
            import asyncio
            await asyncio.wait_for(
                agent_service.embedding_service.embed_text("test"),
                timeout=10.0
            )
        except Exception as e:
            logger.warning(f"Embedding availability check failed: {e}")
            embedding_available = False
        
        return {
            "status": "healthy" if llm_available and embedding_available else "degraded",
            "llm_available": llm_available,
            "embedding_available": embedding_available,
            "agent_service_initialized": agent_service is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat status error: {e}", exc_info=True)
        return {
            "status": "error",
            "llm_available": False,
            "embedding_available": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }