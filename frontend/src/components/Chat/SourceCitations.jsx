import React, { useState } from 'react';
import { FileText, ExternalLink, Star, ChevronDown, ChevronRight } from 'lucide-react';
import './SourceCitations.css'
const SourceCitations = ({ sources }) => {
  const [expandedSources, setExpandedSources] = useState(new Set());

  // Toggle source expansion
  const toggleSource = (index) => {
    const newExpanded = new Set(expandedSources);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSources(newExpanded);
  };

  // Get relevance score color
  const getScoreColor = (score) => {
    if (score >= 0.8) return 'source-score--high';
    if (score >= 0.6) return 'source-score--medium';
    return 'source-score--low';
  };

  // Format relevance score
  const formatScore = (score) => {
    if (typeof score !== 'number') return 'N/A';
    return Math.round(score * 100);
  };

  // Get source title
  const getSourceTitle = (source) => {
    if (source.metadata?.title) return source.metadata.title;
    if (source.metadata?.filename) return source.metadata.filename;
    return `Document ${source.id}`;
  };

  if (!sources || sources.length === 0) {
    return (
      <div className="source-citations source-citations--empty">
        <p>No sources available</p>
      </div>
    );
  }

  return (
    <div className="source-citations">
      <div className="source-citations__header">
        <h4 className="source-citations__title">
          <FileText size={16} />
          Source Documents
        </h4>
        <span className="source-citations__count">
          {sources.length} source{sources.length !== 1 ? 's' : ''}
        </span>
      </div>

      <div className="source-citations__list">
        {sources.map((source, index) => {
          const isExpanded = expandedSources.has(index);
          const relevanceScore = source.relevance_score || source.score || 0;
          const scoreColorClass = getScoreColor(relevanceScore);

          return (
            <div key={index} className="source-citation">
              {/* Source Header */}
              <div
                className="source-citation__header"
                onClick={() => toggleSource(index)}
              >
                <div className="source-citation__header-left">
                  <div className="source-citation__icon">
                    <FileText size={16} />
                  </div>
                  <div className="source-citation__title-section">
                    <h5 className="source-citation__title">
                      {getSourceTitle(source)}
                    </h5>
                    <p className="source-citation__preview">
                      {source.content?.substring(0, 120)}
                      {source.content?.length > 120 ? '...' : ''}
                    </p>
                  </div>
                </div>

                <div className="source-citation__header-right">
                  {/* Relevance Score */}
                  <div className={`source-citation__score ${scoreColorClass}`}>
                    <Star size={12} />
                    <span>{formatScore(relevanceScore)}%</span>
                  </div>

                  <button className="source-citation__toggle">
                    {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                  </button>
                </div>
              </div>

              {/* Source Content */}
              {isExpanded && (
                <div className="source-citation__content">
                  {/* Full Content */}
                  <div className="source-citation__section">
                    <h6 className="source-citation__section-title">Content</h6>
                    <div className="source-citation__text">
                      {source.content}
                    </div>
                  </div>

                  {/* Metadata */}
                  {source.metadata && (
                    <div className="source-citation__section">
                      <h6 className="source-citation__section-title">Document Information</h6>
                      <div className="source-citation__metadata">
                        {source.metadata.filename && (
                          <div className="source-citation__metadata-item">
                            <strong>Filename:</strong> {source.metadata.filename}
                          </div>
                        )}
                        
                        {source.metadata.page_number && (
                          <div className="source-citation__metadata-item">
                            <strong>Page:</strong> {source.metadata.page_number}
                          </div>
                        )}
                        
                        {source.chunk_index !== undefined && (
                          <div className="source-citation__metadata-item">
                            <strong>Section:</strong> {source.chunk_index + 1}
                          </div>
                        )}
                        
                        {source.metadata.document_id && (
                          <div className="source-citation__metadata-item">
                            <strong>Document ID:</strong> 
                            <code className="source-citation__code">
                              {source.metadata.document_id.substring(0, 8)}...
                            </code>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Retrieval Information */}
                  <div className="source-citation__section">
                    <h6 className="source-citation__section-title">Retrieval Details</h6>
                    <div className="source-citation__retrieval">
                      <div className="source-citation__retrieval-item">
                        <strong>Relevance Score:</strong>
                        <div className={`source-citation__score-detail ${scoreColorClass}`}>
                          {formatScore(relevanceScore)}%
                          <div className="source-citation__score-bar">
                            <div 
                              className="source-citation__score-fill"
                              style={{ width: `${formatScore(relevanceScore)}%` }}
                            />
                          </div>
                        </div>
                      </div>
                      
                      {source.retrieval_method && (
                        <div className="source-citation__retrieval-item">
                          <strong>Method:</strong> {source.retrieval_method}
                        </div>
                      )}
                      
                      {source.embedding_distance && (
                        <div className="source-citation__retrieval-item">
                          <strong>Distance:</strong> {source.embedding_distance.toFixed(4)}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="source-citation__actions">
                    <button className="btn btn--ghost btn--sm">
                      <ExternalLink size={14} />
                      View Document
                    </button>
                    
                    <button className="btn btn--ghost btn--sm">
                      Copy Citation
                    </button>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Summary Statistics */}
      <div className="source-citations__summary">
        <div className="source-citations__stat">
          <span className="source-citations__stat-label">Avg. Relevance:</span>
          <span className="source-citations__stat-value">
            {Math.round(
              sources.reduce((sum, s) => sum + (s.relevance_score || s.score || 0), 0) / sources.length * 100
            )}%
          </span>
        </div>
        
        <div className="source-citations__stat">
          <span className="source-citations__stat-label">High Quality:</span>
          <span className="source-citations__stat-value">
            {sources.filter(s => (s.relevance_score || s.score || 0) >= 0.7).length} / {sources.length}
          </span>
        </div>
      </div>
    </div>
  );
};

export default SourceCitations;