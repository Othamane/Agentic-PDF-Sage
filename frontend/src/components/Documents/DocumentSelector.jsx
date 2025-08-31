import React, { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Search, 
  FileText, 
  CheckCircle, 
  Clock, 
  AlertCircle, 
  Upload,
  Loader2,
  Filter,
  X
} from 'lucide-react';
import toast from 'react-hot-toast';

import LoadingSpinner from '../UI/LoadingSpinner';
import DocumentUpload from './DocumentUpload';
import { documentsAPI } from '../../services/api';

// Import styles
import './DocumentSelector.css';

const DocumentSelector = ({ selectedDocuments, onSelectionChange }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('processed');
  const [showUpload, setShowUpload] = useState(false);
  const [localSelected, setLocalSelected] = useState(new Set(selectedDocuments.map(doc => doc.id)));

  // Fetch documents
  const { 
    data: documents = [], 
    isLoading, 
    error, 
    refetch 
  } = useQuery({
    queryKey: ['documents', statusFilter],
    queryFn: () => documentsAPI.listDocuments({ status: statusFilter }),
    refetchInterval: 30000, // Refetch every 30 seconds for status updates
  });

  // Update local selection when props change
  useEffect(() => {
    setLocalSelected(new Set(selectedDocuments.map(doc => doc.id)));
  }, [selectedDocuments]);

  // Handle document selection
  const handleDocumentToggle = (document) => {
    const newSelected = new Set(localSelected);
    
    if (newSelected.has(document.id)) {
      newSelected.delete(document.id);
    } else {
      newSelected.add(document.id);
    }
    
    setLocalSelected(newSelected);
    
    // Convert back to array of documents
    const selectedDocs = documents.filter(doc => newSelected.has(doc.id));
    onSelectionChange(selectedDocs);
  };

  // Handle select all
  const handleSelectAll = () => {
    const processedDocs = filteredDocuments.filter(doc => doc.status === 'processed');
    const allSelected = processedDocs.every(doc => localSelected.has(doc.id));
    
    if (allSelected) {
      // Deselect all
      const newSelected = new Set();
      setLocalSelected(newSelected);
      onSelectionChange([]);
    } else {
      // Select all processed documents
      const newSelected = new Set(processedDocs.map(doc => doc.id));
      setLocalSelected(newSelected);
      onSelectionChange(processedDocs);
    }
  };

  // Filter documents based on search and status
  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = !searchQuery || 
      doc.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.filename?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.description?.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesStatus = !statusFilter || doc.status === statusFilter;
    
    return matchesSearch && matchesStatus;
  });

  // Get status icon and color
  const getStatusInfo = (status) => {
    switch (status) {
      case 'processed':
        return { icon: CheckCircle, color: 'text-green-600', label: 'Ready' };
      case 'processing':
        return { icon: Clock, color: 'text-yellow-600', label: 'Processing' };
      case 'uploaded':
        return { icon: Clock, color: 'text-blue-600', label: 'Queued' };
      case 'failed':
        return { icon: AlertCircle, color: 'text-red-600', label: 'Failed' };
      default:
        return { icon: FileText, color: 'text-gray-600', label: status };
    }
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (error) {
    return (
      <div className="document-selector document-selector--error">
        <div className="document-selector__error">
          <AlertCircle size={24} />
          <h3>Failed to load documents</h3>
          <p>{error.message}</p>
          <button className="btn btn--primary btn--sm" onClick={refetch}>
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="document-selector">
      {/* Header */}
      <div className="document-selector__header">
        <div className="document-selector__header-top">
          <h3 className="document-selector__title">Select Documents</h3>
          <button
            className="btn btn--primary btn--sm"
            onClick={() => setShowUpload(true)}
          >
            <Upload size={16} />
            Upload
          </button>
        </div>

        {/* Search and Filter */}
        <div className="document-selector__controls">
          <div className="document-selector__search">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="document-selector__search-input"
            />
            {searchQuery && (
              <button
                className="document-selector__search-clear"
                onClick={() => setSearchQuery('')}
              >
                <X size={14} />
              </button>
            )}
          </div>

          <div className="document-selector__filter">
            <Filter size={16} />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="document-selector__filter-select"
            >
              <option value="">All Status</option>
              <option value="processed">Ready</option>
              <option value="processing">Processing</option>
              <option value="uploaded">Queued</option>
              <option value="failed">Failed</option>
            </select>
          </div>
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="document-selector__loading">
          <LoadingSpinner />
          <p>Loading documents...</p>
        </div>
      )}

      {/* Document List */}
      {!isLoading && (
        <>
          {/* Selection Controls */}
          {filteredDocuments.length > 0 && (
            <div className="document-selector__selection-controls">
              <button
                className="btn btn--ghost btn--sm"
                onClick={handleSelectAll}
              >
                {filteredDocuments.filter(doc => doc.status === 'processed').every(doc => localSelected.has(doc.id))
                  ? 'Deselect All'
                  : 'Select All Ready'
                }
              </button>
              
              <span className="document-selector__selection-count">
                {localSelected.size} selected
              </span>
            </div>
          )}

          {/* Document Grid */}
          <div className="document-selector__grid">
            {filteredDocuments.length === 0 ? (
              <div className="document-selector__empty">
                <FileText size={48} />
                <h3>No documents found</h3>
                <p>
                  {searchQuery 
                    ? `No documents match "${searchQuery}"`
                    : 'Upload your first PDF document to get started'
                  }
                </p>
                <button
                  className="btn btn--primary"
                  onClick={() => setShowUpload(true)}
                >
                  <Upload size={16} />
                  Upload Document
                </button>
              </div>
            ) : (
              filteredDocuments.map((document) => {
                const isSelected = localSelected.has(document.id);
                const isProcessed = document.status === 'processed';
                const statusInfo = getStatusInfo(document.status);
                const StatusIcon = statusInfo.icon;

                return (
                  <div
                    key={document.id}
                    className={`document-card ${isSelected ? 'document-card--selected' : ''} ${
                      !isProcessed ? 'document-card--disabled' : ''
                    }`}
                    onClick={() => isProcessed && handleDocumentToggle(document)}
                  >
                    {/* Selection Checkbox */}
                    <div className="document-card__checkbox">
                      {isProcessed && (
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => handleDocumentToggle(document)}
                          className="checkbox"
                        />
                      )}
                    </div>

                    {/* Document Info */}
                    <div className="document-card__content">
                      <div className="document-card__header">
                        <div className="document-card__icon">
                          <FileText size={24} />
                        </div>
                        
                        <div className="document-card__status">
                          <StatusIcon size={16} className={statusInfo.color} />
                          <span className={`document-card__status-text ${statusInfo.color}`}>
                            {statusInfo.label}
                          </span>
                        </div>
                      </div>

                      <div className="document-card__body">
                        <h4 className="document-card__title">
                          {document.title || document.filename}
                        </h4>
                        
                        {document.description && (
                          <p className="document-card__description">
                            {document.description}
                          </p>
                        )}

                        {document.summary && (
                          <p className="document-card__summary">
                            {document.summary.length > 120 
                              ? `${document.summary.substring(0, 120)}...`
                              : document.summary
                            }
                          </p>
                        )}

                        {/* Keywords */}
                        {document.keywords && document.keywords.length > 0 && (
                          <div className="document-card__keywords">
                            {document.keywords.slice(0, 3).map((keyword, idx) => (
                              <span key={idx} className="document-card__keyword">
                                {keyword}
                              </span>
                            ))}
                            {document.keywords.length > 3 && (
                              <span className="document-card__keyword-more">
                                +{document.keywords.length - 3} more
                              </span>
                            )}
                          </div>
                        )}
                      </div>

                      <div className="document-card__footer">
                        <div className="document-card__meta">
                          <span className="document-card__meta-item">
                            {formatFileSize(document.file_size)}
                          </span>
                          
                          {document.page_count && (
                            <span className="document-card__meta-item">
                              {document.page_count} pages
                            </span>
                          )}
                          
                          {document.chunk_count && (
                            <span className="document-card__meta-item">
                              {document.chunk_count} chunks
                            </span>
                          )}
                        </div>

                        <div className="document-card__date">
                          {new Date(document.created_at).toLocaleDateString()}
                        </div>
                      </div>

                      {/* Processing Status Details */}
                      {document.status === 'processing' && (
                        <div className="document-card__processing">
                          <Loader2 size={16} className="animate-spin" />
                          <span>Processing document...</span>
                        </div>
                      )}

                      {document.status === 'failed' && document.processing_error && (
                        <div className="document-card__error">
                          <AlertCircle size={16} />
                          <span>Error: {document.processing_error}</span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </>
      )}
      </div>
      )}

      export default DocumentSelector ; 