"""
Enhanced document processing service with proper status handling.
Fixes status inconsistency and session management issues.
"""

import asyncio
import hashlib
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, BinaryIO

import PyPDF2
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, update
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.database import get_db_session, create_db_session
from app.models.document import Document, DocumentStatus
from app.services.enhanced_vector_service import enhanced_vector_service
from app.services.gemini_llm_service import improved_llm_service

logger = logging.getLogger(__name__)
settings = get_settings()


class EnhancedDocumentService:
    """
    Enhanced service for managing document upload, processing, and storage.
    Fixes status consistency and improves session management.
    """
    
    def __init__(self):
        """Initialize the document service."""
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Add vector service reference
        self.vector_service = enhanced_vector_service
        
        # Allowed file extensions
        self.allowed_extensions = {'.pdf'}
        self.max_file_size = settings.MAX_FILE_SIZE
        
        # Processing status lock to prevent race conditions
        self._processing_locks = {}
    
    async def upload_document(
        self,
        file: BinaryIO,
        filename: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Document:
        """
        Upload and process a document with proper status management.
        """
        document = None
        try:
            # Validate file
            await self._validate_file(file, filename)
            
            # Calculate file hash for deduplication
            file_content = await self._read_file_content(file)
            file_hash = self._calculate_hash(file_content)
            
            # Check for existing document with same hash
            async with get_db_session() as db:
                stmt = select(Document).where(Document.file_hash == file_hash)
                result = await db.execute(stmt)
                existing_doc = result.scalar_one_or_none()
                
                if existing_doc:
                    raise ValueError("Document with identical content already exists")
            
            # Save file to storage
            storage_path = await self._save_file(file_content, filename, file_hash)
            
            # Create document record with explicit status
            document = Document(
                filename=filename,
                title=title or self._extract_title_from_filename(filename),
                description=description,
                file_size=len(file_content),
                file_hash=file_hash,
                storage_path=str(storage_path),
                status=DocumentStatus.UPLOADED,  # Explicit enum value
                user_id=user_id
            )
            
            # Save to database with proper session handling
            async with get_db_session() as db:
                db.add(document)
                await db.flush()  # Get the ID without committing
                
                # Refresh to get all fields
                await db.refresh(document)
                
                # Log the initial status
                logger.info(f"Document {document.id} created with status: {document.status}")
                
                # Commit the transaction
                await db.commit()
            
            # Start background processing (non-blocking)
            asyncio.create_task(self._process_document_async(str(document.id)))
            
            logger.info(f"Document uploaded successfully: {document.id}")
            return document
            
        except Exception as e:
            logger.error(f"Document upload failed: {e}", exc_info=True)
            
            # Clean up file if document creation failed
            if document is None and 'storage_path' in locals():
                try:
                    if os.path.exists(storage_path):
                        os.remove(storage_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up file: {cleanup_error}")
            
            raise
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID with proper session handling."""
        try:
            async with get_db_session() as db:
                # Use select with explicit ID conversion
                stmt = select(Document).where(Document.id == document_id)
                result = await db.execute(stmt)
                document = result.scalar_one_or_none()
                
                if document:
                    # Ensure status is properly loaded
                    logger.debug(f"Retrieved document {document_id} with status: {document.status}")
                
                return document
                
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    async def list_documents(
        self,
        user_id: Optional[str] = None,
        status: Optional[DocumentStatus] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Document]:
        """List documents with proper status handling."""
        try:
            async with get_db_session() as db:
                # Build query with SQLAlchemy select
                stmt = select(Document)
                
                # Add filters
                if user_id:
                    stmt = stmt.where(Document.user_id == user_id)
                
                if status:
                    stmt = stmt.where(Document.status == status)
                
                # Add ordering and pagination
                stmt = stmt.order_by(Document.created_at.desc()).limit(limit).offset(offset)
                
                result = await db.execute(stmt)
                documents = result.scalars().all()
                
                # Log status for debugging
                for doc in documents:
                    logger.debug(f"Document {doc.id} status: {doc.status}")
                
                return list(documents)
                
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    async def update_document_status(
        self,
        document_id: str,
        status: DocumentStatus,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update document status with proper transaction handling.
        """
        try:
            async with get_db_session() as db:
                # Use update statement for atomic operation
                stmt = update(Document).where(Document.id == document_id)
                
                update_data = {
                    "status": status,
                    "updated_at": datetime.utcnow()
                }
                
                if status == DocumentStatus.PROCESSING:
                    update_data["processing_started_at"] = datetime.utcnow()
                    update_data["processing_error"] = None
                elif status == DocumentStatus.PROCESSED:
                    update_data["processing_completed_at"] = datetime.utcnow()
                    update_data["processing_error"] = None
                elif status == DocumentStatus.FAILED:
                    update_data["processing_completed_at"] = datetime.utcnow()
                    update_data["processing_error"] = error_message
                
                stmt = stmt.values(**update_data)
                
                result = await db.execute(stmt)
                await db.commit()
                
                if result.rowcount > 0:
                    logger.info(f"Updated document {document_id} status to {status.value}")
                    return True
                else:
                    logger.warning(f"No document found with ID {document_id} for status update")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to update document status {document_id}: {e}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its associated data."""
        try:
            async with get_db_session() as db:
                # Get document
                stmt = select(Document).where(Document.id == document_id)
                result = await db.execute(stmt)
                document = result.scalar_one_or_none()
                
                if not document:
                    return False
                
                # Delete vector store
                await enhanced_vector_service.delete_vector_store(document_id)
                
                # Delete physical file
                if os.path.exists(document.storage_path):
                    os.remove(document.storage_path)
                
                # Update document status to deleted
                await self.update_document_status(document_id, DocumentStatus.DELETED)
            
            logger.info(f"Document deleted successfully: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def get_document_content(self, document_id: str) -> Optional[str]:
        """Get the extracted text content of a document."""
        try:
            document = await self.get_document(document_id)
            if not document or not document.extracted_text:
                return None
            
            return document.extracted_text
            
        except Exception as e:
            logger.error(f"Failed to get document content {document_id}: {e}")
            return None
    
    async def search_documents(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search across documents using enhanced vector similarity."""
        try:
            if not document_ids:
                # Get all processed documents
                documents = await self.list_documents(status=DocumentStatus.PROCESSED)
                document_ids = [str(doc.id) for doc in documents]
            
            if not document_ids:
                logger.warning("No processed documents found for search")
                return []
            
            logger.info(f"Searching {len(document_ids)} documents for: '{query[:100]}...'")
            
            # Search using enhanced vector service
            similar_chunks = await enhanced_vector_service.search_similar(
                document_ids=document_ids,
                query=query,
                k=k
            )
            
            # Format results
            results = []
            for chunk in similar_chunks:
                results.append({
                    "document_id": chunk.metadata.get("document_id"),
                    "content": chunk.page_content,
                    "score": chunk.metadata.get("score", 0.0),
                    "chunk_index": chunk.metadata.get("chunk_index"),
                    "metadata": chunk.metadata
                })
            
            logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}", exc_info=True)
            return []
    
    # Private helper methods - enhanced versions
    
    async def _validate_file(self, file: BinaryIO, filename: str):
        """Validate uploaded file."""
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        if file_ext not in self.allowed_extensions:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
        
        if file_size == 0:
            raise ValueError("Empty file")
    
    async def _read_file_content(self, file: BinaryIO) -> bytes:
        """Read file content into memory."""
        file.seek(0)
        content = file.read()
        return content
    
    def _calculate_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of file content."""
        return hashlib.sha256(content).hexdigest()
    
    async def _save_file(self, content: bytes, filename: str, file_hash: str) -> Path:
        """Save file to storage."""
        # Create subdirectory based on hash prefix
        subdir = self.upload_dir / file_hash[:2]
        subdir.mkdir(exist_ok=True)
        
        # Use hash as filename to avoid conflicts
        file_ext = Path(filename).suffix
        storage_path = subdir / f"{file_hash}{file_ext}"
        
        # Write file
        with open(storage_path, 'wb') as f:
            f.write(content)
        
        return storage_path
    
    def _extract_title_from_filename(self, filename: str) -> str:
        """Extract title from filename."""
        return Path(filename).stem.replace('_', ' ').replace('-', ' ').title()
    
    async def _process_document_async(self, document_id: str):
        """
        Process document in background with proper status management and session handling.
        """
        # Prevent multiple processing of the same document
        if document_id in self._processing_locks:
            logger.warning(f"Document {document_id} is already being processed")
            return
        
        self._processing_locks[document_id] = True
        
        try:
            logger.info(f"Starting document processing: {document_id}")
            
            # Update status to processing
            success = await self.update_document_status(
                document_id, 
                DocumentStatus.PROCESSING
            )
            
            if not success:
                raise Exception("Failed to update document status to processing")
            
            # Get document with fresh session
            document = await self.get_document(document_id)
            if not document:
                raise Exception("Document not found")
            
            # Extract text from PDF
            extracted_text = await self._extract_pdf_text(document.storage_path)
            
            if not extracted_text.strip():
                raise Exception("No text could be extracted from PDF")
            
            # Count pages and words
            page_count = await self._count_pdf_pages(document.storage_path)
            word_count = len(extracted_text.split())
            
            # Generate summary using Gemini
            try:
                summary = await asyncio.wait_for(
                    self._generate_summary(extracted_text), 
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"Summary generation timed out for document {document_id}")
                summary = "Summary generation timed out"
            except Exception as e:
                logger.warning(f"Summary generation failed for document {document_id}: {e}")
                summary = "Summary generation failed"
            
            # Extract keywords
            keywords = await self._extract_keywords(extracted_text)
            
            # Create vector store with enhanced service
            try:
                vector_store_path = await asyncio.wait_for(
                    enhanced_vector_service.create_vector_store(
                        document_id=document_id,
                        text_content=extracted_text,
                        metadata={
                            "filename": document.filename,
                            "title": document.title,
                            "page_count": page_count
                        }
                    ),
                    timeout=300.0  # 5 minutes timeout
                )
            except asyncio.TimeoutError:
                raise Exception("Vector store creation timed out")
            
            # Get vector store stats
            vector_stats = await enhanced_vector_service.get_vector_store_stats(document_id)
            chunk_count = vector_stats.get("total_chunks", 0)
            
            # Update document with all extracted information
            async with get_db_session() as db:
                stmt = update(Document).where(Document.id == document_id).values(
                    status=DocumentStatus.PROCESSED,
                    processing_completed_at=datetime.utcnow(),
                    extracted_text=extracted_text,
                    page_count=page_count,
                    word_count=word_count,
                    chunk_count=chunk_count,
                    summary=summary,
                    keywords=",".join(keywords),
                    vector_store_path=vector_store_path,
                    embedding_model=settings.EMBEDDING_MODEL,
                    updated_at=datetime.utcnow()
                )
                
                await db.execute(stmt)
                await db.commit()
            
            logger.info(f"Document processing completed successfully: {document_id}")
            
        except Exception as e:
            logger.error(f"Document processing failed for {document_id}: {e}", exc_info=True)
            
            # Update status to failed
            await self.update_document_status(
                document_id, 
                DocumentStatus.FAILED, 
                str(e)[:1000]
            )
        
        finally:
            # Remove processing lock
            if document_id in self._processing_locks:
                del self._processing_locks[document_id]
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file."""
        loop = asyncio.get_event_loop()
        
        def _extract():
            try:
                text = ""
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                
                return text.strip()
                
            except Exception as e:
                logger.error(f"PDF text extraction failed: {e}")
                return ""
        
        return await loop.run_in_executor(None, _extract)
    
    async def _count_pdf_pages(self, file_path: str) -> int:
        """Count pages in PDF file."""
        loop = asyncio.get_event_loop()
        
        def _count():
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    return len(pdf_reader.pages)
            except Exception:
                return 0
        
        return await loop.run_in_executor(None, _count)
    
    async def _generate_summary(self, text: str, max_length: int = 500) -> str:
        """Generate document summary using Gemini."""
        try:
            # Limit input text for summarization
            max_input = 8000  # Gemini can handle more
            if len(text) > max_input:
                text = text[:max_input] + "..."
            
            prompt = f"""Please provide a concise summary of the following document in 2-3 sentences:

{text}

Summary:"""
            
            summary = await improved_llm_service.generate_response(
                prompt, 
                max_tokens=200
            )
            
            # Clean up summary
            summary = summary.strip()
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
            
            return summary
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return "Summary generation not available"
    
    async def _extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from document text."""
        try:
            # Simple keyword extraction using word frequency
            import re
            from collections import Counter
            
            # Clean and tokenize text
            text = text.lower()
            words = re.findall(r'\b\w{4,}\b', text)  # Words with 4+ characters
            
            # Remove common stop words
            stop_words = {
                'that', 'with', 'have', 'this', 'will', 'they', 'from', 'been',
                'said', 'each', 'which', 'their', 'time', 'would', 'there',
                'could', 'other', 'more', 'very', 'what', 'know', 'just',
                'first', 'into', 'over', 'think', 'also', 'your', 'work',
                'life', 'only', 'can', 'still', 'should', 'after', 'being',
                'now', 'made', 'before', 'here', 'through', 'when', 'where',
                'much', 'used', 'during', 'good', 'well', 'such', 'many',
                'may', 'use', 'make', 'way', 'even', 'new', 'want', 'come',
                'take', 'get', 'see', 'give', 'back', 'call', 'how', 'its',
                'who', 'did', 'yes', 'his', 'her', 'him', 'had', 'let',
                'put', 'too', 'old', 'any', 'day', 'same', 'right', 'look',
                'down', 'way', 'find', 'long', 'great', 'little', 'own',
                'say', 'man', 'year', 'part', 'show', 'every', 'never',
                'place', 'large', 'turn', 'ask', 'become', 'follow', 'around'
            }
            
            # Filter out stop words and count frequency
            filtered_words = [word for word in words if word not in stop_words]
            word_counts = Counter(filtered_words)
            
            # Get top keywords
            keywords = [word for word, count in word_counts.most_common(max_keywords)]
            
            return keywords
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []


# Global enhanced document service instance
enhanced_document_service = EnhancedDocumentService()