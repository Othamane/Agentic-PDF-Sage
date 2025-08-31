-- Agentic PDF Sage Database Initialization
-- PostgreSQL initialization script

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable full-text search extension
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create database if it doesn't exist (this will be handled by Docker)
-- CREATE DATABASE agentsage;

-- Connect to the database
\c agentsage;

-- Create enum types
CREATE TYPE document_status AS ENUM ('uploaded', 'processing', 'processed', 'failed', 'deleted');
CREATE TYPE step_type AS ENUM ('planning', 'retrieval', 'synthesis', 'validation', 'refinement', 'error');

-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    title VARCHAR(500),
    description TEXT,
    file_size INTEGER NOT NULL,
    mime_type VARCHAR(100) NOT NULL DEFAULT 'application/pdf',
    file_hash VARCHAR(64) NOT NULL UNIQUE,
    storage_path VARCHAR(500) NOT NULL,
    status document_status NOT NULL DEFAULT 'uploaded',
    page_count INTEGER,
    word_count INTEGER,
    chunk_count INTEGER,
    extracted_text TEXT,
    summary TEXT,
    keywords TEXT,
    processing_started_at TIMESTAMP,
    processing_completed_at TIMESTAMP,
    processing_error TEXT,
    vector_store_path VARCHAR(500),
    embedding_model VARCHAR(100),
    user_id VARCHAR(255),
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_message TEXT NOT NULL,
    ai_response TEXT NOT NULL,
    reasoning_trace JSONB,
    session_id VARCHAR(255),
    user_id VARCHAR(255),
    response_time_ms INTEGER,
    token_count INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create agent_steps table
CREATE TABLE IF NOT EXISTS agent_steps (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    step_type step_type NOT NULL,
    input_data JSONB,
    output_data JSONB,
    reasoning TEXT,
    duration_ms REAL,
    token_count INTEGER,
    error_message TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create retrieval_logs table
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_content TEXT NOT NULL,
    chunk_index INTEGER,
    relevance_score REAL,
    embedding_distance REAL,
    search_query VARCHAR(1000),
    retrieval_method VARCHAR(50) DEFAULT 'vector_similarity',
    page_number INTEGER,
    character_start INTEGER,
    character_end INTEGER,
    was_used_in_response BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance

-- Documents indexes
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_is_public ON documents(is_public);

-- Full-text search index on documents
CREATE INDEX IF NOT EXISTS idx_documents_search ON documents USING gin(
    (title || ' ' || filename || ' ' || COALESCE(description, '') || ' ' || COALESCE(extracted_text, ''))
    gin_trgm_ops
);

-- Conversations indexes
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session_id ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC);

-- Agent steps indexes
CREATE INDEX IF NOT EXISTS idx_agent_steps_conversation_id ON agent_steps(conversation_id);
CREATE INDEX IF NOT EXISTS idx_agent_steps_step_type ON agent_steps(step_type);
CREATE INDEX IF NOT EXISTS idx_agent_steps_created_at ON agent_steps(created_at DESC);

-- Retrieval logs indexes
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_conversation_id ON retrieval_logs(conversation_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_document_id ON retrieval_logs(document_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_relevance_score ON retrieval_logs(relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_created_at ON retrieval_logs(created_at DESC);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries

-- Document summary view
CREATE OR REPLACE VIEW document_summary AS
SELECT 
    id,
    filename,
    title,
    description,
    file_size,
    status,
    page_count,
    word_count,
    chunk_count,
    summary,
    STRING_TO_ARRAY(keywords, ',') as keywords_array,
    user_id,
    is_public,
    created_at,
    updated_at
FROM documents
WHERE status != 'deleted';

-- Processing stats view
CREATE OR REPLACE VIEW processing_stats AS
SELECT 
    status,
    COUNT(*) as count,
    AVG(EXTRACT(EPOCH FROM (processing_completed_at - processing_started_at))) as avg_processing_time_seconds,
    SUM(file_size) as total_size_bytes,
    SUM(page_count) as total_pages,
    SUM(chunk_count) as total_chunks
FROM documents 
WHERE status != 'deleted'
GROUP BY status;

-- Recent conversations view
CREATE OR REPLACE VIEW recent_conversations AS
SELECT 
    c.id,
    c.user_message,
    c.ai_response,
    c.response_time_ms,
    c.created_at,
    COUNT(r.id) as source_count,
    ARRAY_AGG(DISTINCT d.title) FILTER (WHERE d.title IS NOT NULL) as source_documents
FROM conversations c
LEFT JOIN retrieval_logs r ON c.id = r.conversation_id
LEFT JOIN documents d ON r.document_id = d.id
GROUP BY c.id, c.user_message, c.ai_response, c.response_time_ms, c.created_at
ORDER BY c.created_at DESC;

-- Create functions for analytics

-- Function to get document stats
CREATE OR REPLACE FUNCTION get_document_stats()
RETURNS TABLE(
    total_documents BIGINT,
    total_size_bytes BIGINT,
    total_pages BIGINT,
    status_breakdown JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_documents,
        SUM(d.file_size) as total_size_bytes,
        SUM(d.page_count) as total_pages,
        JSON_OBJECT_AGG(d.status, d.status_count) as status_breakdown
    FROM (
        SELECT 
            file_size,
            page_count,
            status,
            COUNT(*) OVER (PARTITION BY status) as status_count
        FROM documents 
        WHERE status != 'deleted'
    ) d;
END;
$$ LANGUAGE plpgsql;

-- Function to get conversation analytics
CREATE OR REPLACE FUNCTION get_conversation_analytics(days_back INTEGER DEFAULT 30)
RETURNS TABLE(
    total_conversations BIGINT,
    avg_response_time_ms NUMERIC,
    total_tokens_used BIGINT,
    conversations_by_day JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_conversations,
        AVG(response_time_ms) as avg_response_time_ms,
        SUM(token_count) as total_tokens_used,
        JSON_OBJECT_AGG(
            DATE(created_at),
            daily_count
        ) as conversations_by_day
    FROM (
        SELECT 
            response_time_ms,
            token_count,
            created_at,
            COUNT(*) OVER (PARTITION BY DATE(created_at)) as daily_count
        FROM conversations 
        WHERE created_at >= CURRENT_DATE - INTERVAL '%s days' 
    ) c;
END;
$$ LANGUAGE plpgsql;

-- Insert sample data for testing (optional)
-- Uncomment the following lines if you want sample data

/*
-- Sample document
INSERT INTO documents (
    filename, title, description, file_size, file_hash, storage_path, status
) VALUES (
    'sample.pdf', 
    'Sample Document', 
    'A sample PDF document for testing', 
    1024000, 
    'sample_hash_123', 
    '/uploads/sample.pdf', 
    'processed'
);

-- Sample conversation
INSERT INTO conversations (
    user_message, ai_response, reasoning_trace
) VALUES (
    'What is this document about?',
    'This is a sample document for testing the system.',
    '[]'::jsonb
);
*/

-- Create a user for the application (if needed)
-- This is typically handled by the application, not the database init script

-- Grant permissions (adjust as needed for your security requirements)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_app_user;

-- Add any additional constraints or configurations

-- Constraint to ensure positive file sizes
ALTER TABLE documents ADD CONSTRAINT check_positive_file_size CHECK (file_size > 0);

-- Constraint to ensure valid relevance scores
ALTER TABLE retrieval_logs ADD CONSTRAINT check_relevance_score_range 
    CHECK (relevance_score IS NULL OR (relevance_score >= 0 AND relevance_score <= 1));

-- Constraint to ensure positive response times
ALTER TABLE conversations ADD CONSTRAINT check_positive_response_time 
    CHECK (response_time_ms IS NULL OR response_time_ms >= 0);

-- Print completion message
DO $$
BEGIN
    RAISE NOTICE 'Agentic PDF Sage database initialization completed successfully!';
    RAISE NOTICE 'Database: agentsage';
    RAISE NOTICE 'Tables created: documents, conversations, agent_steps, retrieval_logs';
    RAISE NOTICE 'Views created: document_summary, processing_stats, recent_conversations';
    RAISE NOTICE 'Functions created: get_document_stats(), get_conversation_analytics()';
END $$;