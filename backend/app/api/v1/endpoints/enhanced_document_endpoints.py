"""
Updated document management endpoints using enhanced document service.
Fixes status consistency and improves error handling.
"""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, status, Query
from pydantic import BaseModel, Field

from app.services.enhanced_document_service import enhanced_document_service
from app.models.document import DocumentStatus

logger = logging.getLogger(__name__)

router = APIRouter()


# Response Models (same as before but enhanced)
class DocumentSummary(BaseModel):
    """Document summary response model."""
    id: str
    filename: str
    title: Optional[str] = None
    description: Optional[str] = None
    file_size: int
    status: str
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    chunk_count: Optional[int] = None
    summary: Optional[str] = None
    keywords: List[str] = []
    created_at: str
    updated_at: str


class DocumentDetail(DocumentSummary):
    """Detailed document response model."""
    mime_type: str
    file_hash: str
    embedding_model: Optional[str] = None
    processing_started_at: Optional[str] = None
    processing_completed_at: Optional[str] = None
    processing_error: Optional[str] = None
    is_public: bool = False


class DocumentUploadResponse(BaseModel):
    """Enhanced document upload response model."""
    id: str
    filename: str
    status: str
    message: str
    upload_time: str
    processing_info: Optional[dict] = None


class SearchResult(BaseModel):
    """Document search result model."""
    document_id: str
    content: str
    score: float
    chunk_index: Optional[int] = None
    metadata: dict = {}


class SearchResponse(BaseModel):
    """Enhanced search response model."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    debug_info: Optional[dict] = None


def get_status_value(doc_status) -> str:
    """Helper function to safely get status as string value."""
    if isinstance(doc_status, DocumentStatus):
        return doc_status.value
    elif isinstance(doc_status, str):
        return doc_status
    else:
        return DocumentStatus.UPLOADED.value


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None)
):
    """
    Upload a PDF document for processing with enhanced handling.
    """
    try:
        logger.info(f"Enhanced document upload: {file.filename}")
        
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Upload and process document using enhanced service
        document = await enhanced_document_service.upload_document(
            file=file.file,
            filename=file.filename,
            title=title,
            description=description,
            user_id=user_id
        )
        
        # Safely get status value
        doc_status = get_status_value(document.status)
        
        # Enhanced response with processing info
        processing_info = {
            "service_version": "enhanced",
            "expected_processing_time": "1-5 minutes",
            "status_check_endpoint": f"/api/documents/{document.id}/status",
            "vector_store_creation": "in_progress"
        }
        
        response = DocumentUploadResponse(
            id=str(document.id),
            filename=document.filename,
            status=doc_status,
            message="Document uploaded successfully. Enhanced processing started with Gemini LLM.",
            upload_time=document.created_at.isoformat(),
            processing_info=processing_info
        )
        
        logger.info(f"Enhanced document upload completed: {document.id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced document upload error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document. Please try again."
        )


@router.get("/", response_model=List[DocumentSummary])
async def list_documents(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Number of documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip")
):
    """
    List documents with enhanced filtering and status handling.
    """
    try:
        logger.info(f"Enhanced list documents: status_filter={status_filter}, limit={limit}")
        
        # Parse status filter
        status_enum = None
        if status_filter:
            try:
                status_enum = DocumentStatus(status_filter)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status_filter}. Valid statuses: {[s.value for s in DocumentStatus]}"
                )
        
        # Get documents from enhanced service
        documents = await enhanced_document_service.list_documents(
            user_id=user_id,
            status=status_enum,
            limit=limit,
            offset=offset
        )
        
        # Convert to response models with enhanced data
        response = []
        for doc in documents:
            doc_dict = doc.to_summary_dict()
            # Ensure status is properly converted
            doc_dict["status"] = get_status_value(doc.status)
            response.append(DocumentSummary(**doc_dict))
        
        logger.info(f"Enhanced list documents returned {len(response)} documents")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced list documents error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str):
    """
    Get detailed information about a specific document with enhanced data.
    """
    try:
        logger.info(f"Enhanced get document: {document_id}")
        
        document = await enhanced_document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Convert to response model with enhanced status handling
        doc_dict = document.to_dict()
        doc_dict["status"] = get_status_value(document.status)
        
        logger.info(f"Enhanced get document completed: {document_id}, status: {doc_dict['status']}")
        return DocumentDetail(**doc_dict)
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )
    except Exception as e:
        logger.error(f"Enhanced get document error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its associated data using enhanced service.
    """
    try:
        logger.info(f"Enhanced delete document: {document_id}")
        
        success = await enhanced_document_service.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or already deleted"
            )
        
        logger.info(f"Enhanced delete document completed: {document_id}")
        return {"message": "Document deleted successfully", "service_version": "enhanced"}
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )
    except Exception as e:
        logger.error(f"Enhanced delete document error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


@router.get("/{document_id}/content")
async def get_document_content(document_id: str):
    """
    Get the extracted text content of a document using enhanced service.
    """
    try:
        logger.info(f"Enhanced get document content: {document_id}")
        
        content = await enhanced_document_service.get_document_content(document_id)
        
        if content is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or content not available"
            )
        
        return {
            "document_id": document_id,
            "content": content,
            "length": len(content),
            "service_version": "enhanced",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )
    except Exception as e:
        logger.error(f"Enhanced get document content error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document content"
        )


@router.post("/search", response_model=SearchResponse)
async def search_documents(
    query: str = Query(..., min_length=1, max_length=500),
    document_ids: Optional[List[str]] = Query(default=None, description="Specific documents to search"),
    k: int = Query(default=10, ge=1, le=50, description="Number of results to return")
):
    """
    Search across documents using enhanced vector similarity.
    """
    start_time = datetime.utcnow()
    debug_info = {
        "service_version": "enhanced",
        "vector_service": "enhanced",
        "embedding_model": "sentence-transformers"
    }
    
    try:
        logger.info(f"Enhanced document search: query='{query[:100]}...', k={k}")
        
        # Perform enhanced search
        results = await enhanced_document_service.search_documents(
            query=query,
            document_ids=document_ids,
            k=k
        )
        
        # Calculate search time
        end_time = datetime.utcnow()
        search_time_ms = (end_time - start_time).total_seconds() * 1000
        
        # Convert to response format
        search_results = []
        for result in results:
            search_results.append(SearchResult(**result))
        
        debug_info.update({
            "documents_searched": len(document_ids) if document_ids else "all_processed",
            "results_found": len(search_results),
            "search_time_ms": search_time_ms
        })
        
        response = SearchResponse(
            query=query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time_ms,
            debug_info=debug_info
        )
        
        logger.info(f"Enhanced document search completed: {len(search_results)} results in {search_time_ms:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"Enhanced document search error: {e}", exc_info=True)
        debug_info["error"] = str(e)
        
        return SearchResponse(
            query=query,
            results=[],
            total_results=0,
            search_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            debug_info=debug_info
        )


@router.get("/{document_id}/status")
async def get_document_status(document_id: str):
    """
    Get the processing status of a document with enhanced information.
    """
    try:
        logger.info(f"Enhanced get document status: {document_id}")
        
        document = await enhanced_document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Calculate processing time if applicable
        processing_time_ms = None
        if document.processing_started_at and document.processing_completed_at:
            delta = document.processing_completed_at - document.processing_started_at
            processing_time_ms = delta.total_seconds() * 1000
        
        # Safely get status value
        doc_status = get_status_value(document.status)
        
        # Get vector store stats for enhanced info
        vector_stats = {}
        try:
            vector_stats = await enhanced_document_service.vector_service.get_vector_store_stats(document_id)
        except Exception as e:
            vector_stats = {"error": str(e)}
        
        response = {
            "document_id": document_id,
            "status": doc_status,
            "processing_started_at": document.processing_started_at.isoformat() if document.processing_started_at else None,
            "processing_completed_at": document.processing_completed_at.isoformat() if document.processing_completed_at else None,
            "processing_time_ms": processing_time_ms,
            "processing_error": document.processing_error,
            "chunk_count": document.chunk_count,
            "word_count": document.word_count,
            "page_count": document.page_count,
            "file_size": document.file_size,
            "vector_stats": vector_stats,
            "service_version": "enhanced",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Enhanced get document status completed: {document_id}, status: {doc_status}")
        return response
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )
    except Exception as e:
        logger.error(f"Enhanced get document status error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document status"
        )


# Enhanced statistics endpoint
@router.get("/stats/overview")
async def get_documents_overview():
    """
    Get overview statistics about documents using enhanced service.
    """
    try:
        logger.info("Enhanced documents overview requested")
        
        # Get documents by status using enhanced service
        all_docs = await enhanced_document_service.list_documents(limit=1000)  # Reasonable limit
        
        # Calculate enhanced statistics
        total_documents = len(all_docs)
        status_counts = {}
        total_size = 0
        total_pages = 0
        total_chunks = 0
        processing_times = []
        
        for doc in all_docs:
            # Use enhanced status handling
            doc_status = get_status_value(doc.status)
            status_counts[doc_status] = status_counts.get(doc_status, 0) + 1
            total_size += doc.file_size
            
            if doc.page_count:
                total_pages += doc.page_count
            
            if doc.chunk_count:
                total_chunks += doc.chunk_count
            
            # Calculate processing time if available
            if doc.processing_started_at and doc.processing_completed_at:
                delta = doc.processing_completed_at - doc.processing_started_at
                processing_times.append(delta.total_seconds())
        
        # Enhanced statistics
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "total_documents": total_documents,
            "status_breakdown": status_counts,
            "total_size_bytes": total_size,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "average_processing_time_seconds": avg_processing_time,
            "processed_documents": status_counts.get(DocumentStatus.PROCESSED.value, 0),
            "processing_success_rate": (
                status_counts.get(DocumentStatus.PROCESSED.value, 0) / 
                max(1, total_documents - status_counts.get(DocumentStatus.UPLOADED.value, 0) - status_counts.get(DocumentStatus.PROCESSING.value, 0))
            ) if total_documents > 0 else 0,
            "service_version": "enhanced",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Enhanced documents overview error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents overview"
        )


# Enhanced debug endpoint
@router.get("/debug/{document_id}/vector-store")
async def debug_document_vector_store(document_id: str):
    """
    Debug endpoint for checking document vector store status.
    """
    try:
        logger.info(f"Enhanced vector store debug: {document_id}")
        
        # Get document info
        document = await enhanced_document_service.get_document(document_id)
        if not document:
            return {
                "document_id": document_id,
                "document_found": False,
                "error": "Document not found"
            }
        
        # Get detailed vector store stats
        vector_stats = await enhanced_document_service.vector_service.get_vector_store_stats(document_id)
        
        # Test vector search
        test_search_results = []
        try:
            test_search_results = await enhanced_document_service.search_documents(
                query="test search",
                document_ids=[document_id],
                k=3
            )
        except Exception as search_error:
            vector_stats["search_test_error"] = str(search_error)
        
        return {
            "document_id": document_id,
            "document_found": True,
            "document_status": get_status_value(document.status),
            "document_chunk_count": document.chunk_count,
            "vector_stats": vector_stats,
            "test_search_results_count": len(test_search_results),
            "test_search_preview": [
                {
                    "score": r.get("score", 0),
                    "content_length": len(r.get("content", ""))
                } for r in test_search_results
            ],
            "service_version": "enhanced",
            "debug_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Enhanced vector store debug error: {e}", exc_info=True)
        return {
            "document_id": document_id,
            "error": str(e),
            "service_version": "enhanced",
            "debug_timestamp": datetime.utcnow().isoformat()
        }