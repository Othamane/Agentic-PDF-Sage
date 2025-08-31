"""
Document management endpoints for the Agentic PDF Sage API.
Fixed to handle string status values and resolve variable naming conflicts.
"""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, status, Query
from pydantic import BaseModel, Field

from app.services.document_service import document_service
from app.models.document import DocumentStatus

logger = logging.getLogger(__name__)

router = APIRouter()


# Response Models
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
    """Document upload response model."""
    id: str
    filename: str
    status: str
    message: str
    upload_time: str


class SearchResult(BaseModel):
    """Document search result model."""
    document_id: str
    content: str
    score: float
    chunk_index: Optional[int] = None
    metadata: dict = {}


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float


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
    Upload a PDF document for processing.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are supported"
            )
        
        # Upload and process document
        document = await document_service.upload_document(
            file=file.file,
            filename=file.filename,
            title=title,
            description=description,
            user_id=user_id
        )
        
        # Safely get status value
        doc_status = get_status_value(document.status)
        
        return DocumentUploadResponse(
            id=str(document.id),
            filename=document.filename,
            status=doc_status,
            message="Document uploaded successfully. Processing started.",
            upload_time=document.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error: {e}", exc_info=True)
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
    List documents with optional filtering.
    """
    try:
        # Parse status filter
        status_enum = None
        if status_filter:
            try:
                status_enum = DocumentStatus(status_filter)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status_filter}"
                )
        
        # Get documents from service
        documents = await document_service.list_documents(
            user_id=user_id,
            status=status_enum,
            limit=limit,
            offset=offset
        )
        
        # Convert to response models
        response = []
        for doc in documents:
            doc_dict = doc.to_summary_dict()
            response.append(DocumentSummary(**doc_dict))
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List documents error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents"
        )


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str):
    """
    Get detailed information about a specific document.
    """
    try:
        document = await document_service.get_document(document_id)
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Convert to response model
        doc_dict = document.to_dict()
        return DocumentDetail(**doc_dict)
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )
    except Exception as e:
        logger.error(f"Get document error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document"
        )


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its associated data.
    """
    try:
        success = await document_service.delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or already deleted"
            )
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid document ID format"
        )
    except Exception as e:
        logger.error(f"Delete document error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )


@router.get("/{document_id}/content")
async def get_document_content(document_id: str):
    """
    Get the extracted text content of a document.
    """
    try:
        content = await document_service.get_document_content(document_id)
        
        if content is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found or content not available"
            )
        
        return {
            "document_id": document_id,
            "content": content,
            "length": len(content),
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
        logger.error(f"Get document content error: {e}", exc_info=True)
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
    Search across documents using vector similarity.
    """
    start_time = datetime.utcnow()
    
    try:
        # Perform search
        results = await document_service.search_documents(
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
        
        return SearchResponse(
            query=query,
            results=search_results,
            total_results=len(search_results),
            search_time_ms=search_time_ms
        )
        
    except Exception as e:
        logger.error(f"Document search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed. Please try again."
        )


@router.get("/{document_id}/status")
async def get_document_status(document_id: str):
    """
    Get the processing status of a document.
    """
    try:
        document = await document_service.get_document(document_id)
        
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
        
        return {
            "document_id": document_id,
            "status": doc_status,
            "processing_started_at": document.processing_started_at.isoformat() if document.processing_started_at else None,
            "processing_completed_at": document.processing_completed_at.isoformat() if document.processing_completed_at else None,
            "processing_time_ms": processing_time_ms,
            "processing_error": document.processing_error,
            "chunk_count": document.chunk_count,
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
        logger.error(f"Get document status error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document status"
        )


# Statistics endpoint
@router.get("/stats/overview")
async def get_documents_overview():
    """
    Get overview statistics about documents.
    """
    try:
        # Get documents by status
        all_docs = await document_service.list_documents(limit=1000)  # Reasonable limit
        
        # Calculate statistics
        total_documents = len(all_docs)
        status_counts = {}
        total_size = 0
        total_pages = 0
        
        for doc in all_docs:
            # Fixed: Use different variable name to avoid conflict with fastapi.status
            doc_status = get_status_value(doc.status) 
            status_counts[doc_status] = status_counts.get(doc_status, 0) + 1
            total_size += doc.file_size
            if doc.page_count:
                total_pages += doc.page_count
        
        return {
            "total_documents": total_documents,
            "status_breakdown": status_counts,
            "total_size_bytes": total_size,
            "total_pages": total_pages,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Documents overview error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents overview"
        ) 