"""
Agentic reasoning service using free LLMs for PDF-based Q&A.
Implements iterative retrieval and reasoning with transparent logging.
Fixed with proper database session management and performance optimizations.
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
from app.services.free_llm_service import free_llm_service, free_embedding_service
from app.services.vector_service import vector_service

logger = logging.getLogger(__name__)
settings = get_settings()


class AgentService:
    """
    Agentic PDF reasoning service with iterative planning and execution.
    Uses completely free LLMs and embeddings.
    Optimized for production with timeouts and memory management.
    """
    
    def __init__(self):
        """Initialize the agent service."""
        self.llm_service = free_llm_service
        self.embedding_service = free_embedding_service
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Performance settings
        self.max_context_chunks = 20  # Limit total context chunks
        self.max_context_length = 4000  # Prevent runaway context
        self.timeout_per_step = 300000  # Timeout for each reasoning step
        self.total_timeout = 300000  # Total timeout for entire query
        
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
        Process a user query with agentic reasoning.
        
        Args:
            query: User's question
            conversation_id: ID of the conversation
            document_ids: List of document IDs to search
            max_iterations: Maximum reasoning iterations
        
        Returns:
            Dict containing response, reasoning trace, and metadata
        """
        start_time = time.time()
        logger.info(f"Processing query: {query[:100]}...")
        
        # Initialize conversation tracking
        conversation_uuid = uuid.UUID(conversation_id)
        reasoning_trace = []
        all_retrieved_chunks = []
        
        try:
            # Use asyncio.wait_for for total timeout control
            result = await asyncio.wait_for(
                self._process_with_limits(
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
            logger.warning(f"Query processing timed out after {processing_time:.2f}ms")
            
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
            logger.error(f"Error processing query: {e}", exc_info=True)
            
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
    
    async def _process_with_limits(
        self,
        query: str,
        conversation_uuid: uuid.UUID,
        document_ids: List[str],
        max_iterations: int,
        reasoning_trace: List[Dict],
        all_retrieved_chunks: List[Document]
    ) -> Dict[str, Any]:
        """Process query with context and iteration limits."""
        
        # Step 1: Planning Phase
        plan = await asyncio.wait_for(
            self._planning_phase(query, reasoning_trace),
            timeout=self.timeout_per_step
        )
        
        # Step 2: Iterative Retrieval and Reasoning
        context_chunks = []
        for iteration in range(max_iterations):
            logger.info(f"Reasoning iteration {iteration + 1}/{max_iterations}")
            
            try:
                # Retrieve relevant chunks with timeout
                retrieved_chunks = await asyncio.wait_for(
                    self._retrieval_phase(
                        query, plan, document_ids, context_chunks, reasoning_trace
                    ),
                    timeout=self.timeout_per_step
                )
                
                if retrieved_chunks:
                    # Limit context growth
                    remaining_slots = self.max_context_chunks - len(context_chunks)
                    if remaining_slots > 0:
                        retrieved_chunks = retrieved_chunks[:remaining_slots]
                        context_chunks.extend(retrieved_chunks)
                        all_retrieved_chunks.extend(retrieved_chunks)
                
                # Check if we have enough information or hit limits
                if (len(context_chunks) >= self.max_context_chunks or 
                    await self._has_sufficient_information(query, context_chunks, reasoning_trace)):
                    break
                
                # Refine search if needed and not the last iteration
                if iteration < max_iterations - 1:
                    plan = await asyncio.wait_for(
                        self._refine_plan(query, context_chunks, plan, reasoning_trace),
                        timeout=self.timeout_per_step
                    )
                    
            except asyncio.TimeoutError:
                logger.warning(f"Timeout in iteration {iteration + 1}")
                reasoning_trace.append({
                    "step": "retrieval",
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": f"Timeout in iteration {iteration + 1}",
                    "iteration": iteration + 1
                })
                break
        
        # Step 3: Synthesis Phase
        final_response = await asyncio.wait_for(
            self._synthesis_phase(query, context_chunks, reasoning_trace),
            timeout=self.timeout_per_step * 2  # Allow more time for synthesis
        )
        
        # Step 4: Validation Phase (optional, with timeout)
        try:
            validated_response = await asyncio.wait_for(
                self._validation_phase(query, final_response, context_chunks, reasoning_trace),
                timeout=self.timeout_per_step
            )
        except asyncio.TimeoutError:
            logger.warning("Validation phase timed out, using original response")
            validated_response = final_response
        
        # Log to database (background task to avoid blocking response)
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
    
    async def _planning_phase(self, query: str, reasoning_trace: List[Dict]) -> Dict[str, Any]:
        """
        Initial planning phase to understand the query and create search strategy.
        """
        step_start = datetime.utcnow()
        
        planning_prompt = self.system_prompts["planner"].format(query=query)
        
        try:
            plan_response = await self.llm_service.generate_response(
                planning_prompt, 
                max_tokens=500
            )
            
            plan = {
                "original_query": query,
                "search_terms": self._extract_search_terms(plan_response),
                "query_type": self._classify_query_type(query),
                "expected_answer_type": self._get_expected_answer_type(plan_response)
            }
            
            reasoning_trace.append({
                "step": "planning",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": query,
                "output": plan,
                "reasoning": plan_response[:500]  # Limit reasoning length
            })
            
            return plan
            
        except Exception as e:
            logger.error(f"Planning phase error: {e}")
            reasoning_trace.append({
                "step": "planning",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": query,
                "output": None,
                "error": str(e)
            })
            
            # Return a basic plan on error
            return {
                "original_query": query,
                "search_terms": query.split()[:5],
                "query_type": self._classify_query_type(query),
                "expected_answer_type": "descriptive"
            }
    
    async def _retrieval_phase(
        self,
        query: str,
        plan: Dict[str, Any],
        document_ids: List[str],
        existing_context: List[Document],
        reasoning_trace: List[Dict]
    ) -> List[Document]:
        """
        Retrieve relevant document chunks based on query and plan.
        """
        step_start = datetime.utcnow()
        
        try:
            # Generate search queries based on plan
            search_terms = plan.get("search_terms", [])
            search_queries = [query] + search_terms[:3]  # Limit search queries
            
            # Use vector service for search
            k_per_doc = max(1, 5 // len(document_ids)) if document_ids else 5
            similar_chunks = await vector_service.search_similar(
                document_ids=document_ids,
                query=query,
                k=k_per_doc,
                score_threshold=0.1  # Minimum relevance threshold
            )
            
            # Filter out chunks that are too similar to existing context
            unique_chunks = self._deduplicate_chunks(similar_chunks, existing_context)
            
            # Further filter by relevance
            relevant_chunks = await self._filter_relevant_chunks(
                query, unique_chunks, existing_context
            )
            
            reasoning_trace.append({
                "step": "retrieval",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": {
                    "query": query,
                    "search_terms": search_terms[:3],
                    "document_ids": document_ids
                },
                "output": {
                    "chunks_found": len(similar_chunks),
                    "unique_chunks": len(unique_chunks),
                    "relevant_chunks": len(relevant_chunks)
                },
                "reasoning": f"Retrieved {len(relevant_chunks)} relevant chunks from {len(document_ids)} documents"
            })
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Retrieval phase error: {e}")
            reasoning_trace.append({
                "step": "retrieval",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": {"query": query, "document_ids": document_ids},
                "output": None,
                "error": str(e)
            })
            return []
    
    async def _synthesis_phase(
        self,
        query: str,
        context_chunks: List[Document],
        reasoning_trace: List[Dict]
    ) -> str:
        """
        Synthesize final response from retrieved context.
        """
        step_start = datetime.utcnow()
        
        try:
            if not context_chunks:
                return "I couldn't find relevant information in the documents to answer your question."
            
            # Prepare context with length limits
            context_parts = []
            total_length = 0
            
            for i, chunk in enumerate(context_chunks):
                chunk_text = f"Source {i+1}: {chunk.page_content}"
                if total_length + len(chunk_text) > self.max_context_length:
                    break
                context_parts.append(chunk_text)
                total_length += len(chunk_text)
            
            context_text = "\n\n".join(context_parts)
            
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
                    "context_length": len(context_text)
                },
                "output": response[:500],  # Limit output length in trace
                "reasoning": f"Synthesized response from {len(context_chunks)} context chunks"
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Synthesis phase error: {e}")
            reasoning_trace.append({
                "step": "synthesis",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": {"query": query},
                "output": None,
                "error": str(e)
            })
            return "I apologize, but I couldn't generate a response based on the available information."
    
    async def _validation_phase(
        self,
        query: str,
        response: str,
        context_chunks: List[Document],
        reasoning_trace: List[Dict]
    ) -> str:
        """
        Validate and potentially refine the response.
        """
        step_start = datetime.utcnow()
        
        try:
            # Limit context for validation
            context_text = "\n\n".join([
                chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content
                for chunk in context_chunks[:3]  # Only use first 3 chunks for validation
            ])
            
            validation_prompt = self.system_prompts["validator"].format(
                query=query,
                response=response[:1000],  # Limit response length for validation
                context=context_text
            )
            
            validation_result = await self.llm_service.generate_response(
                validation_prompt,
                max_tokens=300
            )
            
            # Simple validation: if validation suggests changes, use original response
            # In a more sophisticated system, you might iteratively improve
            is_valid = "valid" in validation_result.lower()
            
            reasoning_trace.append({
                "step": "validation",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": {
                    "query": query,
                    "response_length": len(response)
                },
                "output": {
                    "is_valid": is_valid,
                    "validation_notes": validation_result[:300]  # Limit validation notes
                },
                "reasoning": f"Response validation: {'passed' if is_valid else 'flagged for review'}"
            })
            
            return response  # Return original response for now
            
        except Exception as e:
            logger.error(f"Validation phase error: {e}")
            reasoning_trace.append({
                "step": "validation",
                "timestamp": step_start.isoformat(),
                "duration_ms": (datetime.utcnow() - step_start).total_seconds() * 1000,
                "input": {"query": query},
                "output": None,
                "error": str(e)
            })
            return response  # Return original response on error
    
    # Helper methods for agent reasoning
    
    def _extract_search_terms(self, plan_response: str) -> List[str]:
        """Extract search terms from planning response."""
        # Simple extraction - in production, use more sophisticated NLP
        terms = []
        lines = plan_response.split('\n')
        for line in lines:
            if 'search' in line.lower() or 'term' in line.lower():
                # Extract terms between quotes or after colons
                words = line.split()
                terms.extend([word.strip('",.:') for word in words if len(word) > 3])
        return terms[:5]  # Limit to 5 terms
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            return 'definition'
        elif any(word in query_lower for word in ['how', 'step', 'process']):
            return 'procedural'
        elif any(word in query_lower for word in ['why', 'reason', 'cause']):
            return 'causal'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus']):
            return 'comparative'
        elif any(word in query_lower for word in ['list', 'enumerate', 'examples']):
            return 'enumerative'
        else:
            return 'general'
    
    def _get_expected_answer_type(self, plan_response: str) -> str:
        """Determine expected answer type from planning."""
        plan_lower = plan_response.lower()
        
        if 'number' in plan_lower or 'quantity' in plan_lower:
            return 'numerical'
        elif 'yes' in plan_lower or 'no' in plan_lower:
            return 'boolean'
        elif 'list' in plan_lower:
            return 'list'
        else:
            return 'descriptive'
    
    async def _has_sufficient_information(
        self,
        query: str,
        context_chunks: List[Document],
        reasoning_trace: List[Dict]
    ) -> bool:
        """Check if we have sufficient information to answer the query."""
        if not context_chunks:
            return False
        
        # Enhanced heuristics for sufficiency check
        if len(context_chunks) >= 5:
            return True
        
        # Check if any chunk has high relevance score
        high_relevance_chunks = [
            chunk for chunk in context_chunks 
            if chunk.metadata.get('score', 0) > 0.7
        ]
        
        return len(high_relevance_chunks) >= 2 or len(context_chunks) >= 3
    
    async def _refine_plan(
        self,
        query: str,
        current_context: List[Document],
        current_plan: Dict[str, Any],
        reasoning_trace: List[Dict]
    ) -> Dict[str, Any]:
        """Refine the search plan based on current context."""
        try:
            if current_context:
                # Extract key terms from current context
                context_text = " ".join([
                    chunk.page_content[:200] for chunk in current_context[-2:]  # Use last 2 chunks
                ])
                words = context_text.lower().split()
                
                # Simple keyword extraction (in production, use proper NLP)
                new_terms = [
                    word for word in words 
                    if len(word) > 5 and word not in current_plan.get("search_terms", [])
                ][:3]
                
                current_plan["search_terms"] = list(set(
                    current_plan.get("search_terms", []) + new_terms
                ))[:5]  # Limit total terms
            
            return current_plan
            
        except Exception as e:
            logger.warning(f"Plan refinement error: {e}")
            return current_plan
    
    def _deduplicate_chunks(
        self, 
        chunks: List[Document], 
        existing_context: List[Document] = None
    ) -> List[Document]:
        """Remove duplicate chunks based on content similarity."""
        if not chunks:
            return []
        
        unique_chunks = []
        seen_content = set()
        
        # Add existing context to seen content
        if existing_context:
            for chunk in existing_context:
                seen_content.add(hash(chunk.page_content[:100]))  # Use first 100 chars
        
        for chunk in chunks:
            # Simple deduplication by content hash
            content_hash = hash(chunk.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    async def _filter_relevant_chunks(
        self,
        query: str,
        chunks: List[Document],
        existing_context: List[Document]
    ) -> List[Document]:
        """Filter chunks by relevance to query."""
        if not chunks:
            return []
        
        # Use existing scores from vector search if available
        relevant_chunks = [
            chunk for chunk in chunks
            if chunk.metadata.get('score', 0) > 0.1  # Minimum relevance threshold
        ]
        
        # If no scored chunks, fall back to simple term matching
        if not relevant_chunks:
            query_terms = set(query.lower().split())
            for chunk in chunks:
                chunk_terms = set(chunk.page_content.lower().split())
                overlap = len(query_terms.intersection(chunk_terms))
                if overlap > 0:
                    chunk.metadata['relevance_score'] = overlap
                    relevant_chunks.append(chunk)
        
        # Sort by relevance and return top chunks
        relevant_chunks.sort(
            key=lambda x: x.metadata.get('score', x.metadata.get('relevance_score', 0)), 
            reverse=True
        )
        return relevant_chunks[:10]  # Limit to top 10
    
    def _format_sources(self, chunks: List[Document]) -> List[Dict[str, Any]]:
        """Format source citations for response."""
        sources = []
        for i, chunk in enumerate(chunks):
            sources.append({
                "id": i + 1,
                "content": chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content,
                "metadata": {
                    "document_id": chunk.metadata.get("document_id"),
                    "chunk_index": chunk.metadata.get("chunk_index"),
                    "score": chunk.metadata.get("score", chunk.metadata.get("relevance_score", 0))
                },
                "relevance_score": chunk.metadata.get('score', chunk.metadata.get('relevance_score', 0))
            })
        return sources
    
    # Database logging methods with proper session management
    
    async def _log_conversation_async(
        self,
        conversation_id: uuid.UUID,
        query: str,
        response: str,
        reasoning_trace: List[Dict],
        retrieved_chunks: List[Document]
    ):
        """Log conversation to database asynchronously."""
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
            await db_session.flush()  # Ensure conversation is saved before steps
            
            # Log agent steps
            for i, step in enumerate(reasoning_trace):
                try:
                    step_type_str = step.get("step", "unknown")
                    # Map step names to enum values
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
                        reasoning=str(step.get("reasoning", ""))[:1000],  # Limit length
                        duration_ms=step.get("duration_ms", 0),
                        error_message=step.get("error"),
                        created_at=datetime.fromisoformat(step["timestamp"]) if "timestamp" in step else datetime.utcnow()
                    )
                    db_session.add(agent_step)
                except Exception as step_error:
                    logger.warning(f"Failed to log step {i}: {step_error}")
            
            # Log retrieval information
            for chunk in retrieved_chunks[:50]:  # Limit to 50 chunks
                try:
                    retrieval_log = RetrievalLog(
                        conversation_id=conversation_id,
                        document_id=chunk.metadata.get('document_id'),
                        chunk_content=chunk.page_content[:2000],  # Limit content length
                        chunk_index=chunk.metadata.get('chunk_index'),
                        relevance_score=float(chunk.metadata.get('score', chunk.metadata.get('relevance_score', 0.0))),
                        search_query=query[:500],  # Limit query length
                        created_at=datetime.utcnow()
                    )
                    db_session.add(retrieval_log)
                except Exception as retrieval_error:
                    logger.warning(f"Failed to log retrieval: {retrieval_error}")
            
            await db_session.commit()
            logger.info(f"Conversation logged successfully: {conversation_id}")
            
        except Exception as e:
            logger.error(f"Database logging error: {e}")
            if db_session:
                await db_session.rollback()
        finally:
            if db_session:
                await db_session.close()
    
    async def _log_error(self, conversation_id: uuid.UUID, query: str, error: str):
        """Log error to database."""
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
                    "error": error
                }],
                created_at=datetime.utcnow()
            )
            db_session.add(conversation)
            await db_session.commit()
            
        except Exception as e:
            logger.error(f"Error logging failed: {e}")
            if db_session:
                await db_session.rollback()
        finally:
            if db_session:
                await db_session.close()
    
    # System prompts for different phases
    
    def _get_planner_prompt(self) -> str:
        """Get the system prompt for the planning phase."""
        return """You are an AI assistant that helps plan how to answer questions about PDF documents.

Given a user question, analyze it and create a search strategy:

1. Identify the key concepts and entities mentioned
2. Generate alternative search terms that might be used in the documents
3. Classify the type of question (factual, procedural, comparative, etc.)
4. Predict what type of answer is expected

User Question: {query}

Provide your analysis and search strategy:"""
    
    def _get_retriever_prompt(self) -> str:
        """Get the system prompt for the retrieval phase."""
        return """You are helping to retrieve relevant information from documents.

Focus on finding content that directly addresses the user's question.
Consider multiple perspectives and related concepts."""
    
    def _get_synthesizer_prompt(self) -> str:
        """Get the system prompt for the synthesis phase."""
        return """You are an AI assistant that provides accurate, helpful answers based on provided context.

User Question: {query}

Context from Documents:
{context}

Instructions:
1. Answer the question directly and accurately based on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite specific parts of the context when possible
4. Be concise but comprehensive
5. If there are multiple perspectives in the context, present them fairly

Provide your response:"""
    
    def _get_validator_prompt(self) -> str:
        """Get the system prompt for the validation phase."""
        return """You are an AI assistant that validates whether responses accurately reflect the source material.

User Question: {query}
Response to Validate: {response}
Source Context: {context}

Check if the response:
1. Accurately reflects the information in the context
2. Doesn't add information not present in the context
3. Appropriately acknowledges limitations
4. Is relevant to the user's question

Respond with "VALID" if the response is accurate, or suggest improvements:"""