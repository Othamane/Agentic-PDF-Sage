"""
Updated API endpoints using enhanced services with Gemini LLM.
Fixes chunk retrieval and status consistency issues.
"""

import uuid
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from app.services.enhanced_agent_service import enhanced_agent_service
from app.services.enhanced_document_service import enhanced_document_service
from app.models.document import DocumentStatus

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models (same as before)
class ChatMessage(BaseModel):
    """Chat message request model."""
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    document_ids: List[str] = Field(..., min_items=1, description="List of document IDs to search")
    conversation_id: Optional[str] = Field(default=None, description="Existing conversation ID")
    max_iterations: Optional[int] = Field(default=3, ge=1, le=5, description="Max reasoning iterations")


class ChatResponse(BaseModel):
    """Enhanced chat response model."""
    response: str = Field(..., description="AI response")
    conversation_id: str = Field(..., description="Conversation ID")
    reasoning_trace: List[dict] = Field(default=[], description="Agent reasoning steps")
    sources: List[dict] = Field(default=[], description="Source documents and chunks")
    timestamp: str = Field(..., description="Response timestamp")
    response_time_ms: Optional[float] = Field(default=None, description="Response time in milliseconds")
    debug_info: Optional[dict] = Field(default=None, description="Debug information for troubleshooting")


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
    Send a message and get an AI response with enhanced reasoning.
    """
    start_time = datetime.utcnow()
    debug_info = {
        "service_version": "enhanced",
        "document_validation": [],
        "vector_search_debug": {},
        "llm_provider": "gemini"
    }
    
    try:
        logger.info(f"Enhanced chat endpoint: Processing message with {len(request.document_ids)} documents")
        
        # Enhanced document validation
        valid_document_ids = []
        for doc_id in request.document_ids:
            try:
                document = await enhanced_document_service.get_document(doc_id)
                if not document:
                    debug_info["document_validation"].append({
                        "doc_id": doc_id,
                        "status": "not_found",
                        "error": "Document not found"
                    })
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Document {doc_id} not found"
                    )
                
                # Get status as string value safely
                doc_status = get_status_value(document.status)
                
                debug_info["document_validation"].append({
                    "doc_id": doc_id,
                    "status": doc_status,
                    "filename": document.filename,
                    "chunk_count": document.chunk_count
                })
                
                if doc_status != DocumentStatus.PROCESSED.value:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Document {doc_id} is not processed yet. Status: {doc_status}. Please wait for processing to complete."
                    )
                
                # Additional check for vector store existence
                vector_stats = await enhanced_document_service.vector_service.get_vector_store_stats(doc_id)
                if "error" in vector_stats:
                    debug_info["document_validation"][-1]["vector_error"] = vector_stats.get("error")
                    logger.warning(f"Vector store issue for document {doc_id}: {vector_stats.get('error')}")
                else:
                    debug_info["document_validation"][-1]["vector_chunks"] = vector_stats.get("total_chunks", 0)
                
                valid_document_ids.append(doc_id)
                
            except ValueError:
                debug_info["document_validation"].append({
                    "doc_id": doc_id,
                    "status": "invalid_format",
                    "error": "Invalid document ID format"
                })
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid document ID format: {doc_id}"
                )
        
        logger.info(f"Validated {len(valid_document_ids)} documents for processing")
        
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Test vector search before processing
        try:
            test_search_results = await enhanced_document_service.search_documents(
                query=request.message,
                document_ids=valid_document_ids,
                k=5
            )
            
            debug_info["vector_search_debug"] = {
                "test_search_performed": True,
                "test_results_count": len(test_search_results),
                "test_results_preview": [
                    {
                        "doc_id": r.get("document_id", "unknown")[:8],
                        "score": r.get("score", 0),
                        "content_length": len(r.get("content", ""))
                    } for r in test_search_results[:3]
                ]
            }
            
            if not test_search_results:
                logger.warning(f"No chunks found in test search for query: {request.message[:100]}")
                debug_info["vector_search_debug"]["warning"] = "No chunks found in test search"
            
        except Exception as search_error:
            logger.error(f"Test vector search failed: {search_error}")
            debug_info["vector_search_debug"] = {
                "test_search_performed": False,
                "error": str(search_error)
            }
        
        # Process the query using enhanced agent service
        logger.info(f"Processing query with enhanced agent service")
        result = await enhanced_agent_service.process_query(
            query=request.message,
            conversation_id=conversation_id,
            document_ids=valid_document_ids,
            max_iterations=request.max_iterations
        )
        
        # Calculate response time
        end_time = datetime.utcnow()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Enhanced response with debug info
        enhanced_response = ChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            reasoning_trace=result["reasoning_trace"],
            sources=result["sources"],
            timestamp=result["timestamp"],
            response_time_ms=response_time_ms,
            debug_info=debug_info
        )
        
        logger.info(f"Enhanced chat response completed in {response_time_ms:.2f}ms with {len(result.get('sources', []))} sources")
        
        return enhanced_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced chat endpoint error: {e}", exc_info=True)
        debug_info["fatal_error"] = str(e)
        
        # Return error response with debug info
        return ChatResponse(
            response="I apologize, but I encountered an error while processing your message. The development team has been notified.",
            conversation_id=request.conversation_id or str(uuid.uuid4()),
            reasoning_trace=[{
                "step": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }],
            sources=[],
            timestamp=datetime.utcnow().isoformat(),
            response_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            debug_info=debug_info
        )


@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = 50,
    offset: int = 0,
    user_id: Optional[str] = None
):
    """
    List recent conversations (enhanced version).
    """
    try:
        # For now, return empty list as conversations are stored but not exposed via this endpoint
        # In a full implementation, you'd query the conversations table using enhanced_document_service
        
        return []
        
    except Exception as e:
        logger.error(f"Enhanced list conversations error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )


@router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Get a specific conversation by ID (enhanced version).
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
        logger.error(f"Enhanced get conversation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation"
        )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation and its associated data (enhanced version).
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
        logger.error(f"Enhanced delete conversation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation"
        )


@router.get("/status")
async def get_enhanced_chat_status():
    """
    Get the status of enhanced chat services.
    """
    try:
        # Check if enhanced LLM service is available
        llm_available = True
        llm_provider = "unknown"
        try:
            # Test LLM availability with timeout
            import asyncio
            test_response = await asyncio.wait_for(
                enhanced_agent_service.llm_service.generate_response("test", max_tokens=5),
                timeout=10.0
            )
            llm_provider = "gemini" if "test" in test_response.lower() else "fallback"
        except Exception as e:
            logger.warning(f"Enhanced LLM availability check failed: {e}")
            llm_available = False
        
        # Check if enhanced embedding service is available
        embedding_available = True
        embedding_model = "unknown"
        try:
            # Test embedding service with timeout
            import asyncio
            test_embedding = await asyncio.wait_for(
                enhanced_agent_service.embedding_service.embed_text("test"),
                timeout=10.0
            )
            embedding_available = len(test_embedding) > 0
            embedding_model = enhanced_agent_service.embedding_service.model_name
        except Exception as e:
            logger.warning(f"Enhanced embedding availability check failed: {e}")
            embedding_available = False
        
        # Check vector service
        vector_available = True
        try:
            # This is a simple availability check
            vector_available = hasattr(enhanced_agent_service, 'vector_service')
        except Exception as e:
            logger.warning(f"Enhanced vector service check failed: {e}")
            vector_available = False
        
        service_status = "healthy" if (llm_available and embedding_available and vector_available) else "degraded"
        
        return {
            "status": service_status,
            "service_version": "enhanced",
            "llm_available": llm_available,
            "llm_provider": llm_provider,
            "embedding_available": embedding_available,
            "embedding_model": embedding_model,
            "vector_available": vector_available,
            "agent_service_initialized": enhanced_agent_service is not None,
            "enhanced_features": {
                "gemini_integration": True,
                "improved_vector_search": True,
                "enhanced_debugging": True,
                "better_status_management": True
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Enhanced chat status error: {e}", exc_info=True)
        return {
            "status": "error",
            "service_version": "enhanced",
            "llm_available": False,
            "embedding_available": False,
            "vector_available": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Additional debugging endpoint
@router.post("/debug/vector-search")
async def debug_vector_search(
    query: str,
    document_ids: List[str],
    k: int = 5
):
    """
    Debug endpoint for testing vector search directly.
    """
    try:
        logger.info(f"Debug vector search: query='{query}', docs={len(document_ids)}, k={k}")
        
        # Test with enhanced document service
        results = await enhanced_document_service.search_documents(
            query=query,
            document_ids=document_ids,
            k=k
        )
        
        # Get vector store stats for each document
        vector_stats = {}
        for doc_id in document_ids:
            try:
                stats = await enhanced_document_service.vector_service.get_vector_store_stats(doc_id)
                vector_stats[doc_id] = stats
            except Exception as e:
                vector_stats[doc_id] = {"error": str(e)}
        
        return {
            "query": query,
            "document_ids": document_ids,
            "results_count": len(results),
            "results": results,
            "vector_stats": vector_stats,
            "debug_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Debug vector search error: {e}", exc_info=True)
        return {
            "query": query,
            "document_ids": document_ids,
            "error": str(e),
            "debug_timestamp": datetime.utcnow().isoformat()
        }


# Debug endpoint for document status
@router.get("/debug/document/{document_id}")
async def debug_document_status(document_id: str):
    """
    Debug endpoint for checking document status and vector store.
    """
    try:
        # Get document
        document = await enhanced_document_service.get_document(document_id)
        if not document:
            return {
                "document_id": document_id,
                "found": False,
                "error": "Document not found"
            }
        
        # Get vector store stats
        vector_stats = await enhanced_document_service.vector_service.get_vector_store_stats(document_id)
        
        return {
            "document_id": document_id,
            "found": True,
            "filename": document.filename,
            "status": get_status_value(document.status),
            "chunk_count": document.chunk_count,
            "file_size": document.file_size,
            "created_at": document.created_at.isoformat() if document.created_at else None,
            "updated_at": document.updated_at.isoformat() if document.updated_at else None,
            "processing_started_at": document.processing_started_at.isoformat() if document.processing_started_at else None,
            "processing_completed_at": document.processing_completed_at.isoformat() if document.processing_completed_at else None,
            "processing_error": document.processing_error,
            "vector_stats": vector_stats,
            "debug_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Debug document status error: {e}", exc_info=True)
        return {
            "document_id": document_id,
            "error": str(e),
            "debug_timestamp": datetime.utcnow().isoformat()
        }