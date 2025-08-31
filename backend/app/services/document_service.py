"""
Document processing service for PDF files.
Handles upload, text extraction, and vector store creation.
Fixed with proper database session management and error handling.
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
from sqlalchemy import text

from app.core.config import get_settings
from app.core.database import get_db_session, create_db_session
from app.models.document import Document, DocumentStatus
from app.services.vector_service import vector_service
from app.services.free_llm_service import free_llm_service


logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentService:
    """
    Service for managing document upload, processing, and storage.
    """
    
    def __init__(self):
        """Initialize the document service."""
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Allowed file extensions
        self.allowed_extensions = {'.pdf'}
        self.max_file_size = settings.MAX_FILE_SIZE
    
    async def upload_document(
        self,
        file: BinaryIO,
        filename: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Document:
        """
        Upload and process a document.
        
        Args:
            file: File object to upload
            filename: Original filename
            title: Optional document title
            description: Optional document description
            user_id: Optional user identifier
        
        Returns:
            Document model instance
        """
        try:
            # Validate file
            await self._validate_file(file, filename)
            
            # Calculate file hash for deduplication
            file_content = await self._read_file_content(file)
            file_hash = self._calculate_hash(file_content)
            
            # Check for existing document with same hash
            async with get_db_session() as db:
                result = await db.execute(
                    text("SELECT * FROM documents WHERE file_hash = :hash"),
                    {"hash": file_hash}
                )
                if result.first():
                    raise ValueError("Document with identical content already exists")
            
            # Save file to storage
            storage_path = await self._save_file(file_content, filename, file_hash)
            
            # Create document record
            document = Document(
                filename=filename,
                title=title or self._extract_title_from_filename(filename),
                description=description,
                file_size=len(file_content),
                file_hash=file_hash,
                storage_path=str(storage_path),
                status=DocumentStatus.UPLOADED,
                user_id=user_id
            )
            
            # Save to database
            async with get_db_session() as db:
                db.add(document)
                await db.flush()  # Get the ID
                await db.refresh(document)  # Refresh to get all fields
            
            # Start background processing
            asyncio.create_task(self._process_document_async(str(document.id)))
            
            logger.info(f"Document uploaded successfully: {document.id}")
            return document
            
            raise
        except Exception as e:
            logger.error(f"Document upload failed: {e}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        try:
            async with get_db_session() as db:
                result = await db.execute(
                    text("SELECT * FROM documents WHERE id = :id"),
                    {"id": document_id}
                )
                row = result.first()
                if row:
                    # Convert row to dict and create Document instance
                    doc_data = dict(row._mapping)
                    return Document(**doc_data)
                return None
                
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
        """List documents with optional filtering."""
        try:
            query = "SELECT * FROM documents WHERE 1=1"
            params = {}
            
            if user_id:
                query += " AND user_id = :user_id"
                params["user_id"] = user_id
            
            if status:
                query += " AND status = :status"
                params["status"] = status.value
            
            query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
            params.update({"limit": limit, "offset": offset})
            
            async with get_db_session() as db:
                result = await db.execute(text(query), params)
                rows = result.fetchall()
                return [Document(**dict(row._mapping)) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and its associated data."""
        try:
            async with get_db_session() as db:
                # Get document
                result = await db.execute(
                    text("SELECT * FROM documents WHERE id = :id"),
                    {"id": document_id}
                )
                row = result.first()
                if not row:
                    return False
                
                doc_data = dict(row._mapping)
                document = Document(**doc_data)
                
                # Delete vector store
                await vector_service.delete_vector_store(document_id)
                
                # Delete physical file
                if os.path.exists(document.storage_path):
                    os.remove(document.storage_path)
                
                # Update document status
                await db.execute(
                    text("UPDATE documents SET status = :status WHERE id = :id"),
                    {"status": DocumentStatus.DELETED.value, "id": document_id}
                )
            
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
        """Search across documents using vector similarity."""
        try:
            if not document_ids:
                # Get all processed documents
                documents = await self.list_documents(status=DocumentStatus.PROCESSED)
                document_ids = [str(doc.id) for doc in documents]
            
            if not document_ids:
                return []
            
            # Search using vector service
            similar_chunks = await vector_service.search_similar(
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
            
            return results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []
    
    # Private helper methods
    
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
        # Create subdirectory based on hash prefix for better distribution
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
        """Process document in background with proper session management."""
        db_session = None
        try:
            logger.info(f"Starting document processing: {document_id}")
            
            # Create a dedicated session for this background task
            db_session = await create_db_session()
            
            # Update status to processing
            await db_session.execute(
                text("UPDATE documents SET status = :status, processing_started_at = :started_at WHERE id = :id"),
                {
                    "status": DocumentStatus.PROCESSING.value,
                    "started_at": datetime.utcnow(),
                    "id": document_id
                }
            )
            await db_session.commit()
            
            # Get document
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
            
            # Generate summary using free LLM
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
            
            # Create vector store
            try:
                vector_store_path = await asyncio.wait_for(
                    vector_service.create_vector_store(
                        document_id=document_id,
                        text_content=extracted_text,
                        metadata={
                            "filename": document.filename,
                            "title": document.title,
                            "page_count": page_count
                        }
                    ),
                    timeout=300.0  # 5 minutes timeout for vector creation
                )
            except asyncio.TimeoutError:
                raise Exception("Vector store creation timed out")
            
            # Get vector store stats
            vector_stats = await vector_service.get_vector_store_stats(document_id)
            chunk_count = vector_stats.get("total_chunks", 0)
            
            # Update document with extracted information
            await db_session.execute(text("""
                UPDATE documents SET 
                    status = :status,
                    processing_completed_at = :completed_at,
                    extracted_text = :extracted_text,
                    page_count = :page_count,
                    word_count = :word_count,
                    chunk_count = :chunk_count,
                    summary = :summary,
                    keywords = :keywords,
                    vector_store_path = :vector_store_path,
                    embedding_model = :embedding_model
                WHERE id = :id
            """), {
                "status": DocumentStatus.PROCESSED.value,
                "completed_at": datetime.utcnow(),
                "extracted_text": extracted_text,
                "page_count": page_count,
                "word_count": word_count,
                "chunk_count": chunk_count,
                "summary": summary,
                "keywords": ",".join(keywords),
                "vector_store_path": vector_store_path,
                "embedding_model": settings.EMBEDDING_MODEL,
                "id": document_id
            })
            await db_session.commit()
            
            logger.info(f"Document processing completed: {document_id}")
            
        except Exception as e:
            logger.error(f"Document processing failed for {document_id}: {e}")
            
            # Update status to failed
            if db_session:
                try:
                    await db_session.execute(text("""
                        UPDATE documents SET 
                            status = :status,
                            processing_completed_at = :completed_at,
                            processing_error = :error
                        WHERE id = :id
                    """), {
                        "status": DocumentStatus.FAILED.value,
                        "completed_at": datetime.utcnow(),
                        "error": str(e)[:1000],  # Limit error message length
                        "id": document_id
                    })
                    await db_session.commit()
                except Exception as commit_error:
                    logger.error(f"Failed to update error status: {commit_error}")
        
        finally:
            if db_session:
                await db_session.close()
    
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
        """Generate document summary using free LLM."""
        try:
            # Limit input text for summarization
            max_input = 4000  # Conservative limit for free models
            if len(text) > max_input:
                text = text[:max_input] + "..."
            
            prompt = f"""Please provide a concise summary of the following document in about 2-3 sentences:

{text}

Summary:"""
            
            summary = await free_llm_service.generate_response(
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
            # In production, could use more sophisticated NLP
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


# Global document service instance
document_service = DocumentService()