"""
Enhanced vector service for document embeddings and similarity search.
Fixed chunk retrieval issues and improved debugging capabilities.
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
import faiss
from langchain.vectorstores import FAISS
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import get_settings
from app.services.gemini_llm_service import improved_embedding_service

logger = logging.getLogger(__name__)
settings = get_settings()


class ImprovedVectorService:
    """
    Enhanced vector service with better chunk retrieval and debugging.
    """
    
    def __init__(self):
        """Initialize the vector service."""
        self.embedding_service = improved_embedding_service
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Vector store storage directory
        self.vector_store_dir = Path(settings.UPLOAD_DIR) / "vector_stores"
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache management
        self._vector_store_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._cache_stats: Dict[str, Dict[str, Any]] = {}
        
        # Performance settings
        self.max_cache_size = 50
        self.cache_ttl_seconds = 3600
        
        # Debug mode
        self.debug_mode = settings.DEBUG
        
        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())
    
    async def create_vector_store(
        self,
        document_id: str,
        text_content: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Create a vector store from document text with enhanced error handling.
        """
        try:
            logger.info(f"Creating vector store for document {document_id}")
            
            if not text_content or not text_content.strip():
                raise ValueError("Document text content is empty")
            
            # Split text into chunks with better handling
            chunks = await self._split_text_enhanced(text_content)
            
            if not chunks:
                raise ValueError("No text chunks generated from document")
            
            logger.info(f"Generated {len(chunks)} chunks for document {document_id}")
            
            # Create documents with enhanced metadata
            documents = []
            for i, chunk in enumerate(chunks):
                if not chunk or not chunk.strip():
                    logger.warning(f"Skipping empty chunk {i} for document {document_id}")
                    continue
                    
                doc_metadata = {
                    "document_id": document_id,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "chunk_length": len(chunk),
                    "chunk_words": len(chunk.split()),
                    **(metadata or {})
                }
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            if not documents:
                raise ValueError("No valid document chunks created")
            
            logger.info(f"Created {len(documents)} valid documents for embedding")
            
            # Generate embeddings with better error handling
            embeddings = await self._generate_embeddings_enhanced(
                [doc.page_content for doc in documents]
            )
            
            if len(embeddings) != len(documents):
                raise ValueError(f"Embedding count mismatch: {len(embeddings)} vs {len(documents)}")
            
            # Create FAISS vector store with validation
            vector_store = await self._create_faiss_store_enhanced(documents, embeddings)
            
            # Validate the vector store
            await self._validate_vector_store(vector_store, document_id)
            
            # Save vector store
            store_path = await self._save_vector_store_enhanced(document_id, vector_store)
            
            # Cache the vector store
            self._cache_vector_store(document_id, vector_store)
            
            logger.info(f"Vector store created successfully: {store_path}")
            return store_path
            
        except Exception as e:
            logger.error(f"Failed to create vector store for {document_id}: {e}", exc_info=True)
            raise
    
    async def search_similar(
        self,
        document_ids: List[str],
        query: str,
        k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Document]:
        """
        Enhanced similarity search with better debugging and validation.
        """
        try:
            logger.info(f"Searching {len(document_ids)} documents for query: '{query[:100]}...'")
            
            if not query or not query.strip():
                logger.warning("Empty query provided")
                return []
            
            if not document_ids:
                logger.warning("No document IDs provided")
                return []
            
            # Process documents individually to isolate issues
            all_results = []
            failed_documents = []
            
            for doc_id in document_ids:
                try:
                    results = await self._search_single_document_enhanced(
                        doc_id, query, k, score_threshold
                    )
                    
                    if results:
                        all_results.extend(results)
                        logger.info(f"Found {len(results)} chunks in document {doc_id}")
                    else:
                        logger.warning(f"No chunks found in document {doc_id}")
                        failed_documents.append(doc_id)
                        
                except Exception as e:
                    logger.error(f"Search failed for document {doc_id}: {e}")
                    failed_documents.append(doc_id)
            
            if failed_documents:
                logger.warning(f"Search failed for documents: {failed_documents}")
            
            # Sort by relevance score and return top results
            if all_results:
                all_results.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)
                top_results = all_results[:k * min(len(document_ids), 10)]
                logger.info(f"Returning {len(top_results)} total results")
                return top_results
            else:
                logger.warning("No results found in any document")
                return []
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}", exc_info=True)
            return []
    
    async def _search_single_document_enhanced(
        self,
        document_id: str,
        query: str,
        k: int,
        score_threshold: float
    ) -> List[Document]:
        """Enhanced single document search with detailed logging."""
        try:
            # Load vector store with validation
            vector_store = await self._load_vector_store_enhanced(document_id)
            if vector_store is None:
                logger.warning(f"Vector store not found for document {document_id}")
                return []
            
            # Validate vector store has content
            if not hasattr(vector_store, 'index') or vector_store.index is None:
                logger.error(f"Vector store for {document_id} has no index")
                return []
            
            # Check if vector store has documents
            doc_count = self._get_vector_store_doc_count(vector_store)
            if doc_count == 0:
                logger.warning(f"Vector store for {document_id} is empty")
                return []
            
            logger.info(f"Searching vector store with {doc_count} documents for {document_id}")
            
            # Perform search with enhanced error handling
            results = await self._search_vector_store_enhanced(
                vector_store, query, k, score_threshold, document_id
            )
            
            logger.info(f"Vector search returned {len(results)} results for {document_id}")
            
            return results
            
        except Exception as e:
            logger.error(f"Enhanced search failed for document {document_id}: {e}", exc_info=True)
            return []
    
    def _get_vector_store_doc_count(self, vector_store: FAISS) -> int:
        """Get the number of documents in a vector store."""
        try:
            if hasattr(vector_store, 'index') and vector_store.index is not None:
                return vector_store.index.ntotal
            return 0
        except Exception as e:
            logger.warning(f"Could not get vector store document count: {e}")
            return 0
    
    async def _search_vector_store_enhanced(
        self,
        vector_store: FAISS,
        query: str,
        k: int,
        score_threshold: float,
        document_id: str = None
    ) -> List[Document]:
        """Enhanced vector store search with detailed debugging."""
        try:
            logger.info(f"Generating query embedding for: '{query[:50]}...'")
            
            # Generate query embedding with timeout and validation
            query_embedding = await asyncio.wait_for(
                self.embedding_service.embed_text(query),
                timeout=30.0
            )
            
            if not query_embedding or len(query_embedding) == 0:
                logger.error("Query embedding is empty")
                return []
            
            logger.info(f"Generated query embedding with dimension: {len(query_embedding)}")
            
            loop = asyncio.get_event_loop()
            
            def _search():
                try:
                    # Prepare query vector
                    query_array = np.array([query_embedding], dtype=np.float32)
                    
                    # Normalize for cosine similarity
                    faiss.normalize_L2(query_array)
                    
                    # Perform search
                    logger.info(f"Performing FAISS search with k={k}")
                    scores, indices = vector_store.index.search(query_array, k)
                    
                    logger.info(f"FAISS returned scores: {scores[0][:5]} and indices: {indices[0][:5]}")
                    
                    results = []
                    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                        # Check for valid results
                        if idx < 0:  # Invalid index
                            logger.warning(f"Invalid index {idx} at position {i}")
                            continue
                            
                        if score < score_threshold:
                            logger.info(f"Score {score} below threshold {score_threshold}")
                            continue
                        
                        # Get document from store
                        doc_id = vector_store.index_to_docstore_id.get(idx)
                        if not doc_id:
                            logger.warning(f"No document ID for index {idx}")
                            continue
                        
                        # Retrieve document
                        doc = None
                        if hasattr(vector_store.docstore, '_dict'):
                            doc = vector_store.docstore._dict.get(doc_id)
                        else:
                            try:
                                doc = vector_store.docstore.search(doc_id)
                            except:
                                logger.warning(f"Could not retrieve document {doc_id}")
                                continue
                        
                        if not doc:
                            logger.warning(f"Document {doc_id} not found in docstore")
                            continue
                        
                        # Create result document
                        doc_copy = Document(
                            page_content=doc.page_content,
                            metadata=doc.metadata.copy()
                        )
                        doc_copy.metadata['score'] = float(score)
                        doc_copy.metadata['rank'] = i
                        doc_copy.metadata['search_document_id'] = document_id
                        
                        results.append(doc_copy)
                        
                        logger.debug(f"Added result {i}: score={score:.4f}, content_length={len(doc.page_content)}")
                    
                    logger.info(f"Returning {len(results)} valid search results")
                    return results
                    
                except Exception as e:
                    logger.error(f"FAISS search error: {e}", exc_info=True)
                    return []
            
            return await asyncio.wait_for(
                loop.run_in_executor(None, _search),
                timeout=30.0
            )
            
        except asyncio.TimeoutError:
            logger.warning("Vector store search timed out")
            return []
        except Exception as e:
            logger.error(f"Enhanced vector store search failed: {e}", exc_info=True)
            return []
    
    async def _split_text_enhanced(self, text: str) -> List[str]:
        """Enhanced text splitting with validation."""
        try:
            if not text or not text.strip():
                return []
            
            loop = asyncio.get_event_loop()
            chunks = await loop.run_in_executor(
                None,
                lambda: self.text_splitter.split_text(text)
            )
            
            # Filter out empty chunks
            valid_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            
            logger.info(f"Split text into {len(valid_chunks)} valid chunks")
            
            if self.debug_mode and valid_chunks:
                logger.debug(f"First chunk preview: {valid_chunks[0][:200]}...")
                logger.debug(f"Chunk sizes: {[len(chunk) for chunk in valid_chunks[:5]]}")
            
            return valid_chunks
            
        except Exception as e:
            logger.error(f"Text splitting failed: {e}")
            return []
    
    async def _generate_embeddings_enhanced(self, texts: List[str]) -> List[List[float]]:
        """Enhanced embedding generation with validation."""
        try:
            if not texts:
                return []
            
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Process in smaller batches to avoid memory issues
            batch_size = 50
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                try:
                    batch_embeddings = await self.embedding_service.embed_texts(batch)
                    
                    # Validate embeddings
                    if not batch_embeddings or len(batch_embeddings) != len(batch):
                        logger.error(f"Embedding batch {i//batch_size} failed: expected {len(batch)}, got {len(batch_embeddings) if batch_embeddings else 0}")
                        # Create zero embeddings as fallback
                        empty_embedding = [0.0] * 384  # Default dimension
                        batch_embeddings = [empty_embedding] * len(batch)
                    
                    # Validate embedding dimensions
                    for j, embedding in enumerate(batch_embeddings):
                        if not embedding or len(embedding) == 0:
                            logger.warning(f"Empty embedding at batch {i//batch_size}, item {j}")
                            batch_embeddings[j] = [0.0] * 384
                    
                    all_embeddings.extend(batch_embeddings)
                    
                    # Allow other tasks to run
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Batch embedding failed for batch {i//batch_size}: {e}")
                    # Add zero embeddings for failed batch
                    empty_embedding = [0.0] * 384
                    all_embeddings.extend([empty_embedding] * len(batch))
            
            logger.info(f"Generated {len(all_embeddings)} embeddings")
            
            if self.debug_mode and all_embeddings:
                logger.debug(f"First embedding dimension: {len(all_embeddings[0])}")
                logger.debug(f"First embedding preview: {all_embeddings[0][:5]}")
            
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Enhanced embedding generation failed: {e}", exc_info=True)
            return []
    
    async def _create_faiss_store_enhanced(
        self,
        documents: List[Document],
        embeddings: List[List[float]]
    ) -> FAISS:
        """Enhanced FAISS store creation with validation."""
        loop = asyncio.get_event_loop()
        
        def _create_store():
            try:
                logger.info(f"Creating FAISS store with {len(documents)} documents and {len(embeddings)} embeddings")
                
                # Validate inputs
                if len(documents) != len(embeddings):
                    raise ValueError(f"Document count {len(documents)} != embedding count {len(embeddings)}")
                
                if not embeddings or not embeddings[0]:
                    raise ValueError("Embeddings are empty")
                
                # Create embeddings array
                embedding_array = np.array(embeddings, dtype=np.float32)
                logger.info(f"Created embedding array with shape: {embedding_array.shape}")
                
                # Create FAISS index
                dimension = embedding_array.shape[1]
                logger.info(f"Creating FAISS index with dimension: {dimension}")
                
                # Use IndexFlatIP for inner product (cosine similarity when normalized)
                index = faiss.IndexFlatIP(dimension)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embedding_array)
                
                # Add embeddings to index
                index.add(embedding_array)
                logger.info(f"Added {embedding_array.shape[0]} vectors to FAISS index")
                
                # Create document store
                docstore_dict = {}
                index_to_docstore_id = {}
                
                for i, doc in enumerate(documents):
                    doc_id = str(i)
                    docstore_dict[doc_id] = doc
                    index_to_docstore_id[i] = doc_id
                
                logger.info(f"Created docstore with {len(docstore_dict)} documents")
                
                # Create FAISS vector store
                from langchain.docstore.in_memory import InMemoryDocstore
                
                docstore = InMemoryDocstore(docstore_dict)
                
                vector_store = FAISS(
                    embedding_function=None,  # We handle embeddings manually
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id=index_to_docstore_id
                )
                
                logger.info("FAISS vector store created successfully")
                return vector_store
                
            except Exception as e:
                logger.error(f"Enhanced FAISS store creation failed: {e}", exc_info=True)
                raise
        
        return await loop.run_in_executor(None, _create_store)
    
    async def _validate_vector_store(self, vector_store: FAISS, document_id: str):
        """Validate that the vector store was created correctly."""
        try:
            # Check index
            if not hasattr(vector_store, 'index') or vector_store.index is None:
                raise ValueError("Vector store has no index")
            
            # Check document count
            doc_count = vector_store.index.ntotal
            if doc_count == 0:
                raise ValueError("Vector store is empty")
            
            # Check docstore
            if not hasattr(vector_store, 'docstore') or vector_store.docstore is None:
                raise ValueError("Vector store has no docstore")
            
            # Test a simple search
            test_query = "test"
            test_embedding = await self.embedding_service.embed_text(test_query)
            
            if not test_embedding:
                raise ValueError("Could not generate test embedding")
            
            query_array = np.array([test_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            scores, indices = vector_store.index.search(query_array, min(3, doc_count))
            
            if len(scores[0]) == 0:
                raise ValueError("Vector store search returned no results")
            
            logger.info(f"Vector store validation passed for {document_id}: {doc_count} documents")
            
        except Exception as e:
            logger.error(f"Vector store validation failed for {document_id}: {e}")
            raise
    
    async def _save_vector_store_enhanced(self, document_id: str, vector_store: FAISS) -> str:
        """Enhanced vector store saving with validation."""
        loop = asyncio.get_event_loop()
        
        def _save():
            store_path = self.vector_store_dir / f"{document_id}.faiss"
            metadata_path = self.vector_store_dir / f"{document_id}.pkl"
            
            try:
                # Save FAISS index
                faiss.write_index(vector_store.index, str(store_path))
                logger.info(f"Saved FAISS index to {store_path}")
                
                # Prepare metadata
                docstore_dict = {}
                if hasattr(vector_store.docstore, '_dict'):
                    docstore_dict = dict(vector_store.docstore._dict)
                else:
                    # Extract documents from InMemoryDocstore
                    for doc_id in vector_store.index_to_docstore_id.values():
                        try:
                            doc = vector_store.docstore.search(doc_id)
                            docstore_dict[doc_id] = doc
                        except Exception as e:
                            logger.warning(f"Could not extract document {doc_id}: {e}")
                
                metadata = {
                    'docstore': docstore_dict,
                    'index_to_docstore_id': vector_store.index_to_docstore_id,
                    'document_count': len(docstore_dict),
                    'index_dimension': vector_store.index.d if hasattr(vector_store.index, 'd') else None
                }
                
                # Save metadata
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
                
                logger.info(f"Saved metadata to {metadata_path}")
                
                # Validate saved files
                if not store_path.exists() or store_path.stat().st_size == 0:
                    raise ValueError("FAISS index file not saved properly")
                
                if not metadata_path.exists() or metadata_path.stat().st_size == 0:
                    raise ValueError("Metadata file not saved properly")
                
                return str(store_path)
                
            except Exception as e:
                logger.error(f"Enhanced vector store save failed: {e}")
                # Clean up partial files
                if store_path.exists():
                    store_path.unlink()
                if metadata_path.exists():
                    metadata_path.unlink()
                raise
        
        return await loop.run_in_executor(None, _save)
    
    async def _load_vector_store_enhanced(self, document_id: str) -> Optional[FAISS]:
        """Enhanced vector store loading with validation."""
        # Check cache first
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
                logger.warning(f"Vector store files not found for {document_id}")
                return None
            
            try:
                logger.info(f"Loading vector store for {document_id}")
                
                # Load FAISS index
                index = faiss.read_index(str(store_path))
                logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Validate metadata
                if 'docstore' not in metadata or 'index_to_docstore_id' not in metadata:
                    raise ValueError("Invalid metadata format")
                
                logger.info(f"Loaded metadata with {len(metadata['docstore'])} documents")
                
                # Recreate FAISS vector store
                from langchain.docstore.in_memory import InMemoryDocstore
                
                docstore = InMemoryDocstore(metadata['docstore'])
                
                vector_store = FAISS(
                    embedding_function=None,
                    index=index,
                    docstore=docstore,
                    index_to_docstore_id=metadata['index_to_docstore_id']
                )
                
                logger.info(f"Successfully loaded vector store for {document_id}")
                return vector_store
                
            except Exception as e:
                logger.error(f"Enhanced vector store load failed for {document_id}: {e}")
                return None
        
        vector_store = await loop.run_in_executor(None, _load)
        
        # Cache if loaded successfully
        if vector_store:
            self._cache_vector_store(document_id, vector_store)
        
        return vector_store
    
    # ... (rest of the methods remain the same but with enhanced logging)
    
    def _cache_vector_store(self, document_id: str, vector_store: FAISS):
        """Cache vector store with stats tracking."""
        self._vector_store_cache[document_id] = vector_store
        self._cache_stats[document_id] = {
            'last_access': time.time(),
            'access_count': 1,
            'loaded_at': time.time()
        }
        logger.info(f"Cached vector store for {document_id}")
    
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
        
        sorted_items = sorted(
            self._cache_stats.items(),
            key=lambda x: x[1]['last_access']
        )
        
        items_to_remove = len(sorted_items) - self.max_cache_size + 10
        for i in range(min(items_to_remove, len(sorted_items))):
            doc_id, _ = sorted_items[i]
            if doc_id in self._cache_stats:
                del self._cache_stats[doc_id]
                logger.info(f"Removed {doc_id} from cache due to size limits")
    
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
    
    async def get_vector_store_stats(self, document_id: str) -> Dict[str, Any]:
        """Get detailed statistics about a vector store."""
        try:
            vector_store = await self._load_vector_store_enhanced(document_id)
            if vector_store is None:
                return {"error": "Vector store not found"}
            
            doc_count = self._get_vector_store_doc_count(vector_store)
            
            stats = {
                "document_id": document_id,
                "total_chunks": doc_count,
                "embedding_dimension": vector_store.index.d if hasattr(vector_store.index, 'd') else None,
                "embedding_model": settings.EMBEDDING_MODEL,
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "index_type": type(vector_store.index).__name__ if vector_store.index else None
            }
            
            # Add cache stats
            if document_id in self._cache_stats:
                stats["cache_stats"] = self._cache_stats[document_id]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get vector store stats for {document_id}: {e}")
            return {"error": str(e)}
    
    async def delete_vector_store(self, document_id: str) -> bool:
        """Delete a vector store for a document."""
        try:
            # Remove from cache
            if document_id in self._cache_stats:
                del self._cache_stats[document_id]
            
            # Delete files
            store_path = self.vector_store_dir / f"{document_id}.faiss"
            metadata_path = self.vector_store_dir / f"{document_id}.pkl"
            
            deleted_files = []
            if store_path.exists():
                store_path.unlink()
                deleted_files.append("index")
            
            if metadata_path.exists():
                metadata_path.unlink()
                deleted_files.append("metadata")
            
            logger.info(f"Deleted vector store files for {document_id}: {deleted_files}")
            return len(deleted_files) > 0
            
        except Exception as e:
            logger.error(f"Failed to delete vector store for {document_id}: {e}")
            return False


# Global enhanced vector service instance
enhanced_vector_service = ImprovedVectorService()