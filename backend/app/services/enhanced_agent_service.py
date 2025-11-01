"""
Enhanced agentic reasoning service using Gemini LLM for PDF-based Q&A.
Implements iterative retrieval and reasoning with improved chunk retrieval.
"""

import asyncio
import logging
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema.document import Document
from sqlalchemy import text

from app.core.config import get_settings
from app.models.conversation import Conversation
from app.models.agent_step import AgentStep, StepType
from app.models.retrieval_log import RetrievalLog
from app.core.database import get_db_session, create_db_session
from app.services.gemini_llm_service import improved_llm_service, improved_embedding_service
from app.services.enhanced_vector_service import enhanced_vector_service

logger = logging.getLogger(__name__)
settings = get_settings()


class EnhancedAgentService:
    """
    Enhanced agentic PDF reasoning service with Gemini LLM and improved vector search.
    """
    
    def __init__(self):
        """Initialize the enhanced agent service."""
        self.llm_service = improved_llm_service
        self.embedding_service = improved_embedding_service
        self.vector_service = enhanced_vector_service
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Enhanced performance settings
        self.max_context_chunks = 15  # Increased for better context
        self.max_context_length = 8000  # Gemini can handle more
        self.timeout_per_step = 60.0  # Reduced timeout with faster Gemini
        self.total_timeout = 180.0  # 3 minutes total
        
        # System prompts for different reasoning stages
        self.system_prompts = {
            "planner": self._get_planner_prompt(),
            "retriever": self._get_retriever_prompt(),
            "synthesizer": self._get_synthesizer_prompt(),
            "validator": self._get_validator_prompt()
        }
    
    async def process_query(
        self,
        query: str,
        conversation_id: str,
        document_ids: List[str],
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Process a user query with enhanced agentic reasoning using Gemini.
        """
        start_time = time.time()
        logger.info(f"Processing query with enhanced agent: {query[:100]}...")
        
        # Initialize conversation tracking
        conversation_uuid = uuid.UUID(conversation_id)
        reasoning_trace = []
        all_retrieved_chunks = []
        
        try:
            # Use asyncio.wait_for for total timeout control
            result = await asyncio.wait_for(
                self._process_with_enhanced_limits(
                    query, conversation_uuid, document_ids, 
                    max_iterations, reasoning_trace, all_retrieved_chunks
                ),
                timeout=self.total_timeout
            )
            
            # Add processing time to result
            result["processing_time_ms"] = (time.time() - start_time) * 1000
            return result
            
        except asyncio.TimeoutError:
            processing_time = (time.time() - start_time) * 1000
            logger.warning(f"Enhanced query processing timed out after {processing_time:.2f}ms")
            
            # Log timeout to database
            await self._log_error(conversation_uuid, query, "Processing timeout")
            
            return {
                "response": "I apologize, but processing your request took too long. Please try a simpler question or check if the documents are properly processed.",
                "reasoning_trace": reasoning_trace,
                "sources": self._format_sources(all_retrieved_chunks),
                "conversation_id": conversation_id,
                "error": "timeout",
                "processing_time_ms": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Enhanced query processing error: {e}", exc_info=True)
            
            # Log error to database
            await self._log_error(conversation_uuid, query, str(e))
            
            return {
                "response": "I apologize, but I encountered an error processing your question. Please try again.",
                "reasoning_trace": reasoning_trace,
                "sources": self._format_sources(all_retrieved_chunks),
                "conversation_id": conversation_id,
                "error": str(e),
                "processing_time_ms": processing_time,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _process_with_enhanced_limits(
        self,
        query: str,
        conversation_uuid: uuid.UUID,
        document_ids: List[str],
        max_iterations: int,
        reasoning_trace: List[Dict],
        all_retrieved_chunks: List[Document]
    ) -> Dict[str, Any]:
        """Process query with enhanced context and iteration limits."""
        
        # Step 1: Enhanced Planning Phase
        plan = await asyncio.wait_for(
            self._enhanced_planning_phase(query, reasoning_trace),
            timeout=self.timeout_per_step
        )
        
        # Step 2: Enhanced Iterative Retrieval and Reasoning
        context_chunks = []
        for iteration in range(max_iterations):
            logger.info(f"Enhanced reasoning iteration {iteration + 1}/{max_iterations}")
            
            try:
                # Enhanced retrieval with better chunk selection
                retrieved_chunks = await asyncio.wait_for(
                    self._enhanced_retrieval_phase(
                        query, plan, document_ids, context_chunks, reasoning_trace
                    ),
                    timeout=self.timeout_per_step
                )
                
                if retrieved_chunks:
                    # Smart context management
                    remaining_slots = self.max_context_chunks - len(context_chunks)
                    if remaining_slots > 0:
                        # Prioritize high-scoring chunks
                        retrieved_chunks.sort(
                            key=lambda x: x.metadata.get('score', 0), 
                            reverse=True
                        )
                        selected_chunks = retrieved_chunks[:remaining_slots]
                        context_chunks.extend(selected_chunks)
                        all_retrieved_chunks.extend(selected_chunks)
                        
                        logger.info(f"Added {len(selected_chunks)} chunks to context (total: {len(context_chunks)})")
                
                # Enhanced sufficiency check
                if (len(context_chunks) >= self.max_context_chunks or 
                    await self._enhanced_has_sufficient_information(query, context_chunks, reasoning_trace)):
                    logger.info(f"Sufficient information gathered after {iteration + 1} iterations")
                    break
                
                # Enhanced plan refinement
                if iteration < max_iterations - 1:
                    plan = await asyncio.wait_for(
                        self._enhanced_refine_plan(query, context_chunks, plan, reasoning_trace),
                        timeout=self.timeout_per_step
                    )
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout in enhanced iteration {iteration + 1}")
                reasoning_trace.append({
                    "step": "retrieval",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": f"Timeout in iteration {iteration + 1}",
                    "iteration": iteration + 1
                })
                break
        
        # Step 3: Enhanced Synthesis Phase
        final_response = await asyncio.wait_for(
            self._enhanced_synthesis_phase(query, context_chunks, reasoning_trace),
            timeout=self.timeout_per_step * 2  # More time for synthesis
        )
        
        # Step 4: Enhanced Validation Phase
        try:
            validated_response = await asyncio.wait_for(
                self._enhanced_validation_phase(query, final_response, context_chunks, reasoning_trace),
                timeout=self.timeout_per_step
            )
        except asyncio.TimeoutError:
            logger.warning("Enhanced validation phase timed out, using original response")
            validated_response = final_response
        
        # Log to database (background task)
        asyncio.create_task(self._log_conversation_async(
            conversation_uuid, query, validated_response, 
            reasoning_trace, all_retrieved_chunks
        ))
        
        return {
            "response": validated_response,
            "reasoning_trace": reasoning_trace,
            "sources": self._format_sources(all_retrieved_chunks),
            "conversation_id": str(conversation_uuid),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Enhanced phase methods with complete implementations
    
    async def _enhanced_planning_phase(self, query: str, reasoning_trace: List[Dict]) -> Dict[str, Any]:
        """Enhanced planning phase with better query analysis."""
        step_start = datetime.utcnow()
        
        planning_prompt = self.system_prompts["planner"].format(query=query)
        
        try:
            plan_response = await self.llm_service.generate_response(
                planning_prompt, 
                max_tokens=500
            )
            
            plan = {
                "original_query": query,
                "search_terms": self._enhanced_extract_search_terms(plan_response, query),
                "query_type": self._enhanced_classify_query_type(query),
                "expected_answer_type": self._enhanced_get_expected_answer_type(plan_response),
                "complexity_score": self._calculate_query_complexity(query)
            }
            
            reasoning_trace.append({
                "step": "planning",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": query,
                "output": plan,
                "reasoning": plan_response[:500]
            })
            
            logger.info(f"Enhanced planning completed: {plan['query_type']} query with {len(plan['search_terms'])} search terms")
            return plan
            
        except Exception as e:
            logger.error(f"Enhanced planning phase error: {e}")
            reasoning_trace.append({
                "step": "planning",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": query,
                "output": None,
                "error": str(e)
            })
            
            # Return enhanced fallback plan
            return {
                "original_query": query,
                "search_terms": self._enhanced_extract_search_terms("", query),
                "query_type": self._enhanced_classify_query_type(query),
                "expected_answer_type": "descriptive",
                "complexity_score": 1.0
            }
    
    async def _enhanced_retrieval_phase(
        self,
        query: str,
        plan: Dict[str, Any],
        document_ids: List[str],
        existing_context: List[Document],
        reasoning_trace: List[Dict]
    ) -> List[Document]:
        """Enhanced retrieval phase with better chunk selection."""
        step_start = datetime.utcnow()
        
        try:
            logger.info(f"Enhanced retrieval for {len(document_ids)} documents")
            
            # Generate multiple search queries
            search_queries = self._generate_search_queries(query, plan)
            
            # Calculate k based on complexity and document count
            complexity_score = plan.get("complexity_score", 1.0)
            base_k = max(3, int(5 * complexity_score))
            k_per_doc = max(1, base_k // len(document_ids)) if document_ids else 5
            
            logger.info(f"Using k={k_per_doc} per document for {len(search_queries)} search queries")
            
            # Perform enhanced vector search
            all_chunks = []
            for search_query in search_queries[:3]:  # Limit to 3 queries max
                try:
                    chunks = await self.vector_service.search_similar(
                        document_ids=document_ids,
                        query=search_query,
                        k=k_per_doc,
                        score_threshold=0.1
                    )
                    all_chunks.extend(chunks)
                    logger.info(f"Search query '{search_query[:50]}...' found {len(chunks)} chunks")
                except Exception as e:
                    logger.warning(f"Search query '{search_query[:50]}...' failed: {e}")
            
            # Enhanced deduplication and filtering
            unique_chunks = self._enhanced_deduplicate_chunks(all_chunks, existing_context)
            
            # Enhanced relevance filtering
            relevant_chunks = await self._enhanced_filter_relevant_chunks(
                query, unique_chunks, existing_context, plan
            )
            
            reasoning_trace.append({
                "step": "retrieval",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": {
                    "query": query,
                    "search_queries": search_queries,
                    "document_ids": document_ids,
                    "k_per_doc": k_per_doc
                },
                "output": {
                    "total_chunks_found": len(all_chunks),
                    "unique_chunks": len(unique_chunks),
                    "relevant_chunks": len(relevant_chunks),
                    "average_score": sum(c.metadata.get('score', 0) for c in relevant_chunks) / len(relevant_chunks) if relevant_chunks else 0
                },
                "reasoning": f"Enhanced retrieval found {len(relevant_chunks)} relevant chunks from {len(document_ids)} documents"
            })
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Enhanced retrieval phase error: {e}", exc_info=True)
            reasoning_trace.append({
                "step": "retrieval",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": {"query": query, "document_ids": document_ids},
                "output": None,
                "error": str(e)
            })
            return []
    
    async def _enhanced_synthesis_phase(
        self,
        query: str,
        context_chunks: List[Document],
        reasoning_trace: List[Dict]
    ) -> str:
        """Enhanced synthesis phase with better context management."""
        step_start = datetime.utcnow()
        
        try:
            if not context_chunks:
                return "I couldn't find relevant information in the documents to answer your question. Please ensure the documents are properly processed and contain content related to your query."
            
            # Enhanced context preparation
            context_parts = []
            total_length = 0
            
            # Sort chunks by relevance score
            sorted_chunks = sorted(
                context_chunks, 
                key=lambda x: x.metadata.get('score', 0), 
                reverse=True
            )
            
            for i, chunk in enumerate(sorted_chunks):
                # Enhanced chunk formatting
                score = chunk.metadata.get('score', 0)
                doc_id = chunk.metadata.get('document_id', 'unknown')
                chunk_text = f"[Source {i+1}, Score: {score:.3f}, Doc: {doc_id[:8]}...]\n{chunk.page_content}\n"
                
                if total_length + len(chunk_text) > self.max_context_length:
                    break
                    
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
            
            context_text = "\n".join(context_parts)
            
            # Enhanced synthesis prompt
            synthesis_prompt = self.system_prompts["synthesizer"].format(
                query=query,
                context=context_text
            )
            
            response = await self.llm_service.generate_response(
                synthesis_prompt,
                max_tokens=settings.MAX_TOKENS
            )
            
            reasoning_trace.append({
                "step": "synthesis",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": {
                    "query": query,
                    "context_chunks": len(context_chunks),
                    "context_length": len(context_text),
                    "top_scores": [c.metadata.get('score', 0) for c in sorted_chunks[:5]]
                },
                "output": response[:500],
                "reasoning": f"Enhanced synthesis from {len(context_chunks)} chunks with avg score {sum(c.metadata.get('score', 0) for c in context_chunks) / len(context_chunks):.3f}"
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Enhanced synthesis phase error: {e}")
            reasoning_trace.append({
                "step": "synthesis",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": {"query": query},
                "output": None,
                "error": str(e)
            })
            return "I apologize, but I couldn't generate a response based on the available information. Please try rephrasing your question."
    
    async def _enhanced_validation_phase(
        self,
        query: str,
        response: str,
        context_chunks: List[Document],
        reasoning_trace: List[Dict]
    ) -> str:
        """Enhanced validation phase with better quality checks."""
        step_start = datetime.utcnow()
        
        try:
            # Enhanced context for validation
            context_text = "\n\n".join([
                chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content
                for chunk in context_chunks[:5]  # Top 5 chunks
            ])
            
            validation_prompt = self.system_prompts["validator"].format(
                query=query,
                response=response[:1000],
                context=context_text
            )
            
            validation_result = await self.llm_service.generate_response(
                validation_prompt,
                max_tokens=300
            )
            
            # Enhanced validation logic
            is_valid = self._assess_response_quality(validation_result, response, context_chunks)
            
            reasoning_trace.append({
                "step": "validation",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": {
                    "query": query,
                    "response_length": len(response),
                    "context_chunks": len(context_chunks)
                },
                "output": {
                    "is_valid": is_valid,
                    "validation_notes": validation_result[:300],
                    "quality_score": self._calculate_response_quality_score(response, context_chunks)
                },
                "reasoning": f"Enhanced validation: {'passed' if is_valid else 'needs improvement'}"
            })
            
            return response  # Return original response for now
            
        except Exception as e:
            logger.error(f"Enhanced validation phase error: {e}")
            reasoning_trace.append({
                "step": "validation",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": {"query": query},
                "output": None,
                "error": str(e)
            })
            return response
    
    # Enhanced helper methods with complete implementations
    
    def _enhanced_extract_search_terms(self, plan_response: str, original_query: str) -> List[str]:
        """Enhanced search term extraction with better NLP."""
        terms = []
        
        # Extract from plan response
        lines = plan_response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['search', 'term', 'key', 'concept']):
                words = line.split()
                terms.extend([word.strip('",.:()[]{}') for word in words if len(word) > 3])
        
        # Extract from original query
        import re
        query_words = re.findall(r'\b\w{4,}\b', original_query.lower())
        
        # Remove common stop words
        stop_words = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which', 'that', 'this',
            'does', 'can', 'will', 'would', 'could', 'should', 'might', 'may',
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during'
        }
        
        filtered_terms = [term for term in terms + query_words if term.lower() not in stop_words]
        
        # Return unique terms, limited to 8
        return list(dict.fromkeys(filtered_terms))[:8]
    
    def _enhanced_classify_query_type(self, query: str) -> str:
        """Enhanced query type classification."""
        query_lower = query.lower()
        
        # Multi-keyword classification
        if any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning']):
            return 'definition'
        elif any(word in query_lower for word in ['how to', 'how do', 'steps', 'process', 'procedure']):
            return 'procedural'
        elif any(word in query_lower for word in ['why', 'reason', 'cause', 'because']):
            return 'causal'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs', 'contrast']):
            return 'comparative'
        elif any(word in query_lower for word in ['list', 'enumerate', 'examples', 'types']):
            return 'enumerative'
        elif any(word in query_lower for word in ['analysis', 'analyze', 'examine', 'evaluate']):
            return 'analytical'
        elif any(word in query_lower for word in ['summary', 'summarize', 'overview', 'brief']):
            return 'summary'
        else:
            return 'general'
    
    def _enhanced_get_expected_answer_type(self, plan_response: str) -> str:
        """Enhanced answer type prediction."""
        plan_lower = plan_response.lower()
        
        if any(word in plan_lower for word in ['number', 'quantity', 'amount', 'count', 'percentage']):
            return 'numerical'
        elif any(word in plan_lower for word in ['yes', 'no', 'true', 'false', 'confirm']):
            return 'boolean'
        elif any(word in plan_lower for word in ['list', 'items', 'points', 'bullet']):
            return 'list'
        elif any(word in plan_lower for word in ['step', 'procedure', 'process', 'method']):
            return 'procedural'
        else:
            return 'descriptive'
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score (0.5 to 2.0)."""
        complexity_score = 1.0
        
        # Length factor
        if len(query) > 100:
            complexity_score += 0.3
        elif len(query) < 20:
            complexity_score -= 0.2
        
        # Multiple question marks or complex words
        if query.count('?') > 1:
            complexity_score += 0.2
        
        complex_words = ['analyze', 'compare', 'evaluate', 'synthesize', 'relationship', 'implications']
        if any(word in query.lower() for word in complex_words):
            complexity_score += 0.4
        
        # Multiple topics
        if any(word in query.lower() for word in ['and', 'also', 'additionally', 'furthermore']):
            complexity_score += 0.3
        
        return max(0.5, min(2.0, complexity_score))
    
    def _generate_search_queries(self, query: str, plan: Dict[str, Any]) -> List[str]:
        """Generate multiple search queries for better retrieval."""
        queries = [query]  # Original query first
        
        search_terms = plan.get("search_terms", [])
        query_type = plan.get("query_type", "general")
        
        # Add search term combinations
        if len(search_terms) >= 2:
            queries.append(" ".join(search_terms[:3]))
        
        # Add type-specific variations
        if query_type == "definition":
            for term in search_terms[:2]:
                queries.append(f"definition of {term}")
        elif query_type == "procedural":
            for term in search_terms[:2]:
                queries.append(f"how to {term}")
        elif query_type == "comparative":
            queries.append(" ".join(search_terms[:2]) + " comparison")
        
        return queries[:4]  # Limit to 4 queries
    
    def _enhanced_deduplicate_chunks(
        self, 
        chunks: List[Document], 
        existing_context: List[Document] = None
    ) -> List[Document]:
        """Enhanced deduplication with similarity checking."""
        if not chunks:
            return []
        
        unique_chunks = []
        seen_content = set()
        
        # Add existing context to seen content
        if existing_context:
            for chunk in existing_context:
                content_hash = hash(chunk.page_content[:200])
                seen_content.add(content_hash)
        
        for chunk in chunks:
            # Enhanced deduplication using multiple factors
            content_hash = hash(chunk.page_content[:200])
            
            # Check for exact duplicates
            if content_hash in seen_content:
                continue
            
            # Check for high similarity with existing chunks
            is_similar = False
            for existing_chunk in unique_chunks:
                similarity = self._calculate_text_similarity(
                    chunk.page_content, 
                    existing_chunk.page_content
                )
                if similarity > 0.8:  # 80% similarity threshold
                    is_similar = True
                    break
            
            if not is_similar:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        try:
            # Simple word-based similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            if not union:
                return 0.0
            
            return len(intersection) / len(union)
        except:
            return 0.0
    
    async def _enhanced_filter_relevant_chunks(
        self,
        query: str,
        chunks: List[Document],
        existing_context: List[Document],
        plan: Dict[str, Any]
    ) -> List[Document]:
        """Enhanced relevance filtering with plan context."""
        if not chunks:
            return []
        
        search_terms = plan.get("search_terms", [])
        query_type = plan.get("query_type", "general")
        
        relevant_chunks = []
        
        for chunk in chunks:
            relevance_score = chunk.metadata.get('score', 0)
            
            # Enhanced relevance calculation
            enhanced_score = relevance_score
            
            # Boost for search term matches
            content_lower = chunk.page_content.lower()
            term_matches = sum(1 for term in search_terms if term.lower() in content_lower)
            enhanced_score += term_matches * 0.1
            
            # Boost for query type relevance
            if query_type == "definition" and any(word in content_lower for word in ["definition", "means", "refers to"]):
                enhanced_score += 0.15
            elif query_type == "procedural" and any(word in content_lower for word in ["step", "process", "method"]):
                enhanced_score += 0.15
            elif query_type == "comparative" and any(word in content_lower for word in ["compare", "versus", "difference"]):
                enhanced_score += 0.15
            
            # Update metadata with enhanced score
            chunk.metadata['enhanced_score'] = enhanced_score
            
            # Filter by enhanced threshold
            if enhanced_score > 0.1:
                relevant_chunks.append(chunk)
        
        # Sort by enhanced score
        relevant_chunks.sort(key=lambda x: x.metadata.get('enhanced_score', 0), reverse=True)
        return relevant_chunks[:15]  # Return top 15
    
    async def _enhanced_has_sufficient_information(
        self,
        query: str,
        context_chunks: List[Document],
        reasoning_trace: List[Dict]
    ) -> bool:
        """Enhanced sufficiency check."""
        if not context_chunks:
            return False
        
        # Check quantity
        if len(context_chunks) >= 10:
            return True
        
        # Check quality - high scoring chunks
        high_quality_chunks = [
            chunk for chunk in context_chunks 
            if chunk.metadata.get('score', 0) > 0.5
        ]
        
        if len(high_quality_chunks) >= 5:
            return True
        
        # Check coverage - different documents
        unique_docs = set(chunk.metadata.get('document_id') for chunk in context_chunks)
        if len(unique_docs) >= 3 and len(context_chunks) >= 6:
            return True
        
        return False
    
    async def _enhanced_refine_plan(
        self,
        query: str,
        current_context: List[Document],
        current_plan: Dict[str, Any],
        reasoning_trace: List[Dict]
    ) -> Dict[str, Any]:
        """Enhanced plan refinement based on retrieved context."""
        try:
            if current_context:
                # Extract new terms from high-scoring chunks
                high_score_chunks = [
                    chunk for chunk in current_context 
                    if chunk.metadata.get('score', 0) > 0.3
                ]
                
                context_text = " ".join([
                    chunk.page_content[:300] for chunk in high_score_chunks[-3:]
                ])
                
                # Simple term extraction from context
                import re
                words = re.findall(r'\b\w{4,}\b', context_text.lower())
                
                # Filter and add new terms
                existing_terms = set(term.lower() for term in current_plan.get("search_terms", []))
                new_terms = [
                    word for word in words 
                    if (word not in existing_terms and 
                        len(word) > 4 and 
                        words.count(word) > 1)  # Appears multiple times
                ]
                
                # Update plan with new terms
                current_plan["search_terms"] = list(set(
                    current_plan.get("search_terms", []) + new_terms[:3]
                ))[:8]  # Limit total terms
                
                current_plan["refinement_iteration"] = current_plan.get("refinement_iteration", 0) + 1
            
            return current_plan
            
        except Exception as e:
            logger.warning(f"Enhanced plan refinement error: {e}")
            return current_plan
    
    def _assess_response_quality(
        self, 
        validation_result: str, 
        response: str, 
        context_chunks: List[Document]
    ) -> bool:
        """Assess response quality based on validation."""
        validation_lower = validation_result.lower()
        
        # Positive indicators
        positive_indicators = ["valid", "accurate", "correct", "good", "appropriate", "relevant"]
        negative_indicators = ["invalid", "inaccurate", "incorrect", "poor", "inappropriate", "irrelevant"]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in validation_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in validation_lower)
        
        return positive_count > negative_count
    
    def _calculate_response_quality_score(self, response: str, context_chunks: List[Document]) -> float:
        """Calculate a quality score for the response."""
        score = 0.5  # Base score
        
        # Length factor
        if 100 <= len(response) <= 1000:
            score += 0.2
        elif len(response) > 1000:
            score += 0.1
        
        # Context utilization
        if context_chunks:
            avg_score = sum(c.metadata.get('score', 0) for c in context_chunks) / len(context_chunks)
            score += min(0.3, avg_score)
        
        # Response structure (simple heuristics)
        if response.count('.') >= 2:  # Multiple sentences
            score += 0.1
        
        if any(word in response.lower() for word in ['because', 'therefore', 'however', 'additionally']):
            score += 0.1  # Explanatory language
        
        return min(1.0, score)
    
    def _format_sources(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """Enhanced source formatting with better metadata."""
        sources = []
        for i, chunk in enumerate(chunks):
            source = {
                "id": i + 1,
                "content": chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content,
                "metadata": {
                    "document_id": chunk.metadata.get("document_id"),
                    "chunk_index": chunk.metadata.get("chunk_index"),
                    "score": chunk.metadata.get("score", 0),
                    "enhanced_score": chunk.metadata.get("enhanced_score", 0)
                },
                "relevance_score": chunk.metadata.get('score', 0),
                "enhanced_relevance_score": chunk.metadata.get('enhanced_score', 0),
                "chunk_length": len(chunk.page_content),
                "source_document": chunk.metadata.get("document_id", "unknown")[:8] + "..." if chunk.metadata.get("document_id") else "unknown"
            }
            sources.append(source)
        return sources
    
    # Database logging methods (enhanced versions)
    
    async def _log_conversation_async(
        self,
        conversation_id: uuid.UUID,
        query: str,
        response: str,
        reasoning_trace: List[Dict],
        retrieved_chunks: List[Document]
    ):
        """Enhanced conversation logging."""
        db_session = None
        try:
            db_session = await create_db_session()
            
            # Create conversation record
            conversation = Conversation(
                id=conversation_id,
                user_message=query,
                ai_response=response,
                reasoning_trace=reasoning_trace,
                created_at=datetime.utcnow()
            )
            db_session.add(conversation)
            await db_session.flush()
            
            # Log agent steps with enhanced details
            for i, step in enumerate(reasoning_trace):
                try:
                    step_type_str = step.get("step", "unknown")
                    step_type_mapping = {
                        "planning": StepType.PLANNING,
                        "retrieval": StepType.RETRIEVAL,
                        "synthesis": StepType.SYNTHESIS,
                        "validation": StepType.VALIDATION,
                        "error": StepType.ERROR
                    }
                    step_type = step_type_mapping.get(step_type_str, StepType.ERROR)
                    
                    agent_step = AgentStep(
                        conversation_id=conversation_id,
                        step_number=i + 1,
                        step_type=step_type,
                        input_data=step.get("input"),
                        output_data=step.get("output"),
                        reasoning=str(step.get("reasoning", ""))[:1000],
                        duration_ms=step.get("duration_ms", 0),
                        error_message=step.get("error"),
                        created_at=datetime.fromisoformat(step["timestamp"]) if "timestamp" in step else datetime.utcnow()
                    )
                    db_session.add(agent_step)
                except Exception as step_error:
                    logger.warning(f"Failed to log enhanced step {i}: {step_error}")
            
            # Enhanced retrieval logging
            for chunk in retrieved_chunks[:50]:
                try:
                    retrieval_log = RetrievalLog(
                        conversation_id=conversation_id,
                        document_id=chunk.metadata.get('document_id'),
                        chunk_content=chunk.page_content[:2000],
                        chunk_index=chunk.metadata.get('chunk_index'),
                        relevance_score=float(chunk.metadata.get('score', 0.0)),
                        search_query=query[:500],
                        created_at=datetime.utcnow()
                    )
                    db_session.add(retrieval_log)
                except Exception as retrieval_error:
                    logger.warning(f"Failed to log enhanced retrieval: {retrieval_error}")
            
            await db_session.commit()
            logger.info(f"Enhanced conversation logged successfully: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Enhanced database logging error: {e}")
            if db_session:
                await db_session.rollback()
        finally:
            if db_session:
                await db_session.close()
    
    async def _log_error(self, conversation_id: uuid.UUID, query: str, error: str):
        """Enhanced error logging."""
        db_session = None
        try:
            db_session = await create_db_session()
            
            conversation = Conversation(
                id=conversation_id,
                user_message=query,
                ai_response=f"Error: {error}",
                reasoning_trace=[{
                    "step": "error",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": error,
                    "service_version": "enhanced"
                }],
                created_at=datetime.utcnow()
            )
            db_session.add(conversation)
            await db_session.commit()
            
        except Exception as e:
            logger.error(f"Enhanced error logging failed: {e}")
            if db_session:
                await db_session.rollback()
        finally:
            if db_session:
                await db_session.close()
    
    # Enhanced system prompts
    
    def _get_planner_prompt(self) -> str:
        """Enhanced planning prompt."""
        return """You are an AI assistant that creates detailed search strategies for answering questions about PDF documents.

Given a user question, provide a comprehensive analysis:

1. **Key Concepts**: Identify the main topics, entities, and concepts
2. **Search Terms**: Generate alternative terms that might appear in documents
3. **Question Type**: Classify the question (definition, procedural, comparative, analytical, etc.)
4. **Expected Answer**: Predict what type of answer the user expects
5. **Search Strategy**: Suggest how to approach finding relevant information

User Question: {query}

Provide your detailed analysis and search strategy:"""
    
    def _get_retriever_prompt(self) -> str:
        """Enhanced retrieval prompt."""
        return """You are helping to retrieve the most relevant information from documents.

Focus on finding content that directly addresses the user's question while considering:
- Multiple perspectives and related concepts
- Different ways the information might be expressed
- Context that supports understanding"""
    
    def _get_synthesizer_prompt(self) -> str:
        """Enhanced synthesis prompt."""
        return """You are an AI assistant that provides comprehensive, accurate answers based on provided document context.

User Question: {query}

Context from Documents:
{context}

Instructions:
1. **Direct Answer**: Start with a clear, direct answer to the question
2. **Supporting Evidence**: Use specific information from the context to support your answer
3. **Multiple Sources**: Synthesize information from different sources when available
4. **Clarity**: Explain complex concepts clearly and logically
5. **Limitations**: If the context doesn't fully address the question, state what's missing
6. **Structure**: Organize your response with clear paragraphs and logical flow

Provide a comprehensive, well-structured response:"""
    
    def _get_validator_prompt(self) -> str:
        """Enhanced validation prompt."""
        return """You are an AI assistant that validates whether responses accurately reflect source material and adequately answer user questions.

User Question: {query}
Response to Validate: {response}
Source Context: {context}

Evaluation Criteria:
1. **Accuracy**: Does the response accurately reflect the information in the context?
2. **Completeness**: Does it adequately address the user's question?
3. **Relevance**: Is the information provided relevant to the query?
4. **Clarity**: Is the response clear and well-structured?
5. **Source Fidelity**: Does it avoid adding information not present in the context?

Provide your assessment with specific feedback:"""


# Global enhanced agent service instance
enhanced_agent_service = EnhancedAgentService()