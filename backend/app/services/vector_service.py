"""
Vector service for document embeddings and similarity search.
Uses free sentence-transformers for embeddings and FAISS for search.
Optimized for production with memory management and connection pooling.
"""

import asyncio
import logging
import os
import pickle
import time
import weakref
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
from langchain.vectorstores import FAISS
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.services.free_llm_service import free_embedding_service

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorService:
    """
    Service for managing document embeddings and vector search.
    Uses completely free tools - no API costs.
    Optimized for production with memory management.
    """
    
    def __init__(self):
        """Initialize the vector service."""
        self.embedding_service = free_embedding_service
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Vector store storage directory
        self.vector_store_dir = Path(settings.UPLOAD_DIR) / "vector_stores"
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Use WeakValueDictionary for automatic memory cleanup
        self._vector_store_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._cache_stats: Dict[str, Dict[str, Any]] = {}
        
        # Performance settings
        self.max_cache_size = 50  # Maximum number of cached vector stores
        self.cache_ttl_seconds = 3600  # 1 hour TTL for unused stores
        
        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())
    
    async def create_vector_store(
        self,
        document_id: str,
        text_content: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Create a vector store from document text.
        
        Args:
            document_id: Unique document identifier
            text_content: Full text content of the document
            metadata: Additional metadata for the document
        
        Returns:
            Path to the created vector store
        """
        try:
            logger.info(f"Creating vector store for document {document_id}")
            
            # Split text into chunks
            chunks = await self._split_text(text_content)
            
            if not chunks:
                raise ValueError("No text chunks generated from document")
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    **(metadata or {})
                }
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            # Generate embeddings in batches for better memory management
            logger.info(f"Generating embeddings for {len(documents)} chunks")
            embeddings = await self._generate_embeddings_batched([doc.page_content for doc in documents])
            
            # Create FAISS vector store
            vector_store = await self._create_faiss_store(documents, embeddings)
            
            # Save vector store
            store_path = await self._save_vector_store(document_id, vector_store)
            
            # Cache the vector store with stats
            self._cache_vector_store(document_id, vector_store)
            
            logger.info(f"Vector store created successfully: {store_path}")
            return store_path
            
        except Exception as e:
            logger.error(f"Failed to create vector store for {document_id}: {e}")
            raise
    
    async def search_similar(
        self,
        document_ids: List[str],
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Document]:
        """
        Search for similar documents across multiple vector stores.
        
        Args:
            document_ids: List of document IDs to search
            query: Search query
            k: Number of results to return per document
            score_threshold: Minimum similarity score
        
        Returns:
            List of similar documents with scores
        """
        try:
            # Process in batches to prevent memory issues
            batch_size = 5
            all_results = []
            
            for i in range(0, len(document_ids), batch_size):
                batch_ids = document_ids[i:i + batch_size]
                batch_results = await self._search_batch(batch_ids, query, k, score_threshold)
                all_results.extend(batch_results)
                
                # Allow other tasks to run
                await asyncio.sleep(0.01)
            
            # Sort by relevance score and return top results
            all_results.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)
            return all_results[:k * min(len(document_ids), 10)]  # Limit total results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    async def _search_batch(
        self,
        document_ids: List[str],
        query: str,
        k: int,
        score_threshold: float
    ) -> List[Document]:
        """Search a batch of documents concurrently."""
        tasks = []
        
        for document_id in document_ids:
            task = self._search_single_document(document_id, query, k, score_threshold)
            tasks.append(task)
        
        # Execute searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine valid results
        all_results = []
        for result in results:
            if isinstance(result, list):
                all_results.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Search error in batch: {result}")
        
        return all_results
    
    async def _search_single_document(
        self,
        document_id: str,
        query: str,
        k: int,
        score_threshold: float
    ) -> List[Document]:
        """Search within a single document."""
        try:
            vector_store = await self._load_vector_store_optimized(document_id)
            if vector_store is None:
                logger.warning(f"Vector store not found for document {document_id}")
                return []
            
            # Search in this vector store
            results = await self._search_vector_store(
                vector_store, query, k, score_threshold
            )
            
            return results
            
        except Exception as e:
            logger.warning(f"Search failed for document {document_id}: {e}")
            return []
    
    async def get_document_chunks(
        self,
        document_id: str,
        chunk_indices: List[int] = None
    ) -> List[Document]:
        """
        Get specific chunks from a document's vector store.
        
        Args:
            document_id: Document identifier
            chunk_indices: Specific chunk indices to retrieve (None for all)
        
        Returns:
            List of document chunks
        """
        try:
            vector_store = await self._load_vector_store_optimized(document_id)
            if vector_store is None:
                return []
            
            # Get all documents from vector store
            if hasattr(vector_store.docstore, '_dict'):
                all_docs = list(vector_store.docstore._dict.values())
            else:
                # Fallback for different LangChain versions
                all_docs = []
                try:
                    for doc_id in vector_store.index_to_docstore_id.values():
                        if doc_id in vector_store.docstore:
                            all_docs.append(vector_store.docstore[doc_id])
                except Exception as e:
                    logger.warning(f"Could not retrieve documents: {e}")
                    return []
            
            if chunk_indices is None:
                return all_docs
            
            # Filter by chunk indices
            filtered_docs = []
            for doc in all_docs:
                chunk_idx = doc.metadata.get('chunk_index')
                if chunk_idx is not None and chunk_idx in chunk_indices:
                    filtered_docs.append(doc)
            
            return filtered_docs
            
        except Exception as e:
            logger.error(f"Failed to get document chunks for {document_id}: {e}")
            return []
    
    async def delete_vector_store(self, document_id: str) -> bool:
        """
        Delete a vector store for a document.
        
        Args:
            document_id: Document identifier
        
        Returns:
            True if deleted successfully
        """
        try:
            # Remove from cache and stats
            if document_id in self._cache_stats:
                del self._cache_stats[document_id]
            
            # Delete files
            store_path = self.vector_store_dir / f"{document_id}.faiss"
            metadata_path = self.vector_store_dir / f"{document_id}.pkl"
            
            if store_path.exists():
                store_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info(f"Vector store deleted for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vector store for {document_id}: {e}")
            return False
    
    async def get_vector_store_stats(self, document_id: str) -> Dict[str, Any]:
        """
        Get statistics about a vector store.
        
        Args:
            document_id: Document identifier
        
        Returns:
            Dictionary with vector store statistics
        """
        try:
            vector_store = await self._load_vector_store_optimized(document_id)
            if vector_store is None:
                return {}
            
            # Get basic stats
            if hasattr(vector_store.docstore, '_dict'):
                all_docs = list(vector_store.docstore._dict.values())
            else:
                all_docs = []
            
            stats = {
                "document_id": document_id,
                "total_chunks": len(all_docs),
                "embedding_dimension": settings.VECTOR_DIMENSION,
                "embedding_model": settings.EMBEDDING_MODEL,
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP
            }
            
            if all_docs:
                # Calculate text statistics
                total_chars = sum(len(doc.page_content) for doc in all_docs)
                avg_chunk_size = total_chars / len(all_docs)
                
                stats.update({
                    "total_characters": total_chars,
                    "average_chunk_size": int(avg_chunk_size),
                    "min_chunk_size": min(len(doc.page_content) for doc in all_docs),
                    "max_chunk_size": max(len(doc.page_content) for doc in all_docs)
                })
            
            # Add cache stats
            if document_id in self._cache_stats:
                stats["cache_stats"] = self._cache_stats[document_id]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get vector store stats for {document_id}: {e}")
            return {}
    
    # Private helper methods - optimized versions
    
    async def _split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        loop = asyncio.get_event_loop()
        chunks = await loop.run_in_executor(
            None,
            lambda: self.text_splitter.split_text(text)
        )
        return chunks
    
    async def _generate_embeddings_batched(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings in batches for better memory management."""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = await self.embedding_service.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Allow other tasks to run
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Batch embedding failed for batch {i//batch_size}: {e}")
                # Add empty embeddings for failed batch
                empty_embedding = [0.0] * settings.VECTOR_DIMENSION
                all_embeddings.extend([empty_embedding] * len(batch))
        
        return all_embeddings
    
    async def _create_faiss_store(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ) -> FAISS:
        """Create FAISS vector store from documents and embeddings."""
        loop = asyncio.get_event_loop()
        
        def _create_store():
            try:
                # Create embeddings array
                embedding_array = np.array(embeddings, dtype=np.float32)
                
                # Create FAISS store
                import faiss
                
                # Use flat index for simplicity (can be optimized for large datasets)
                dimension = len(embeddings[0])
                index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embedding_array)
                
                # Add embeddings to index
                index.add(embedding_array)
                
                # Create FAISS vector store
                from langchain.vectorstores.faiss import FAISS
                
                # Create docstore
                docstore = {str(i): doc for i, doc in enumerate(documents)}
                index_to_docstore_id = {i: str(i) for i in range(len(documents))}
                
                vector_store = FAISS(
                    embedding_function=None,  # We handle embeddings manually
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id
                )
                
                return vector_store
                
            except Exception as e:
                logger.error(f"FAISS store creation failed: {e}")
                raise
        
        return await loop.run_in_executor(None, _create_store)
    
    async def _save_vector_store(self, document_id: str, vector_store: FAISS) -> str:
        """Save vector store to disk."""
        loop = asyncio.get_event_loop()
        
        def _save():
            store_path = self.vector_store_dir / f"{document_id}.faiss"
            metadata_path = self.vector_store_dir / f"{document_id}.pkl"
            
            try:
                # Save FAISS index
                import faiss
                faiss.write_index(vector_store.index, str(store_path))
                
                # Save metadata
                metadata = {
                    'docstore': dict(vector_store.docstore._dict) if hasattr(vector_store.docstore, '_dict') else {},
                    'index_to_docstore_id': vector_store.index_to_docstore_id
                }
                
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
                
                return str(store_path)
                
            except Exception as e:
                logger.error(f"Failed to save vector store: {e}")
                raise
        
        return await loop.run_in_executor(None, _save)
    
    async def _load_vector_store_optimized(self, document_id: str) -> Optional[FAISS]:
        """Load vector store from disk with caching and stats."""
        # Check cache first and update access time
        if document_id in self._vector_store_cache:
            self._update_cache_stats(document_id)
            return self._vector_store_cache[document_id]
        
        # Clean cache if too large
        await self._enforce_cache_limits()
        
        loop = asyncio.get_event_loop()
        
        def _load():
            store_path = self.vector_store_dir / f"{document_id}.faiss"
            metadata_path = self.vector_store_dir / f"{document_id}.pkl"
            
            if not store_path.exists() or not metadata_path.exists():
                return None
            
            try:
                # Load FAISS index
                import faiss
                index = faiss.read_index(str(store_path))
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Recreate FAISS vector store
                from langchain.vectorstores.faiss import FAISS
                
                vector_store = FAISS(
                    embedding_function=None,
                    index=index,
                    docstore=metadata['docstore'],
                    index_to_docstore_id=metadata['index_to_docstore_id']
                )
                
                return vector_store
                
            except Exception as e:
                logger.error(f"Failed to load vector store {document_id}: {e}")
                return None
        
        vector_store = await loop.run_in_executor(None, _load)
        
        # Cache if loaded successfully
        if vector_store:
            self._cache_vector_store(document_id, vector_store)
        
        return vector_store
    
    def _cache_vector_store(self, document_id: str, vector_store: FAISS):
        """Cache vector store with stats tracking."""
        self._vector_store_cache[document_id] = vector_store
        self._cache_stats[document_id] = {
            'last_access': time.time(),
            'access_count': 1,
            'loaded_at': time.time()
        }
    
    def _update_cache_stats(self, document_id: str):
        """Update cache access statistics."""
        if document_id in self._cache_stats:
            stats = self._cache_stats[document_id]
            stats['last_access'] = time.time()
            stats['access_count'] = stats.get('access_count', 0) + 1
    
    async def _enforce_cache_limits(self):
        """Enforce cache size limits by removing LRU items."""
        if len(self._cache_stats) <= self.max_cache_size:
            return
        
        # Sort by last access time
        sorted_items = sorted(
            self._cache_stats.items(),
            key=lambda x: x[1]['last_access']
        )
        
        # Remove oldest items
        items_to_remove = len(sorted_items) - self.max_cache_size + 10  # Remove extra for buffer
        for i in range(min(items_to_remove, len(sorted_items))):
            doc_id, _ = sorted_items[i]
            if doc_id in self._cache_stats:
                del self._cache_stats[doc_id]
    
    async def _search_vector_store(
        self,
        vector_store: FAISS,
        query: str,
        k: int,
        score_threshold: float
    ) -> List[Document]:
        """Search within a single vector store with timeout."""
        try:
            # Generate query embedding with timeout
            query_embedding = await asyncio.wait_for(
                self.embedding_service.embed_text(query),
                timeout=30.0
            )
            
            loop = asyncio.get_event_loop()
            
            def _search():
                try:
                    # Normalize query embedding
                    import faiss
                    query_array = np.array([query_embedding], dtype=np.float32)
                    faiss.normalize_L2(query_array)
                    
                    # Search
                    scores, indices = vector_store.index.search(query_array, k)
                    
                    results = []
                    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                        if score >= score_threshold and idx >= 0:  # Check for valid index
                            doc_id = vector_store.index_to_docstore_id.get(idx)
                            if doc_id:
                                if hasattr(vector_store.docstore, '_dict'):
                                    doc = vector_store.docstore._dict.get(doc_id)
                                else:
                                    doc = vector_store.docstore.get(doc_id)
                                
                                if doc:
                                    # Create a copy to avoid modifying cached version
                                    doc_copy = Document(
                                        page_content=doc.page_content,
                                        metadata=doc.metadata.copy()
                                    )
                                    doc_copy.metadata['score'] = float(score)
                                    doc_copy.metadata['rank'] = i
                                    results.append(doc_copy)
                    
                    return results
                    
                except Exception as e:
                    logger.error(f"FAISS search error: {e}")
                    return []
            
            return await asyncio.wait_for(
                loop.run_in_executor(None, _search),
                timeout=10.0
            )
            
        except asyncio.TimeoutError:
            logger.warning("Vector store search timed out")
            return []
        except Exception as e:
            logger.error(f"Vector store search failed: {e}")
            return []
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old cache entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                current_time = time.time()
                expired_docs = []
                
                for doc_id, stats in self._cache_stats.items():
                    if current_time - stats['last_access'] > self.cache_ttl_seconds:
                        expired_docs.append(doc_id)
                
                for doc_id in expired_docs:
                    if doc_id in self._cache_stats:
                        del self._cache_stats[doc_id]
                
                if expired_docs:
                    logger.info(f"Cleaned up {len(expired_docs)} expired cache entries")
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")


# Global vector service instance
vector_service = VectorService()