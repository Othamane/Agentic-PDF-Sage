import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  Upload, 
  Search, 
  Filter, 
  MoreVertical, 
  Eye, 
  Download, 
  Trash2,
  RefreshCw,
  Plus,
  FileText,
  BarChart3
} from 'lucide-react';
import toast from 'react-hot-toast';

import DocumentUpload from '../../components/Documents/DocumentUpload';
import LoadingSpinner from '../../components/UI/LoadingSpinner';
import { documentsAPI } from '../../services/api';
import './DocumentManager.css'
const DocumentManager = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [showUpload, setShowUpload] = useState(false);
  const [selectedDocuments, setSelectedDocuments] = useState(new Set());
  
  const queryClient = useQueryClient();

  // Fetch documents
  const { 
    data: documents = [], 
    isLoading, 
    error, 
    refetch 
  } = useQuery({
    queryKey: ['documents', statusFilter],
    queryFn: () => documentsAPI.listDocuments({ status: statusFilter }),
    refetchInterval: 30000,
  });

  // Fetch overview stats
  const { data: overview } = useQuery({
    queryKey: ['documents-overview'],
    queryFn: documentsAPI.getDocumentsOverview,
    refetchInterval: 60000,
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: documentsAPI.deleteDocument,
    onSuccess: () => {
      toast.success('Document deleted successfully');
      queryClient.invalidateQueries(['documents']);
      queryClient.invalidateQueries(['documents-overview']);
    },
    onError: (error) => {
      toast.error(`Failed to delete document: ${error.message}`);
    }
  });

  // Filter documents
  const filteredDocuments = documents.filter(doc => {
    const matchesSearch = !searchQuery || 
      doc.title?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      doc.filename?.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

  // Handle document selection
  const handleSelectDocument = (documentId) => {
    const newSelected = new Set(selectedDocuments);
    if (newSelected.has(documentId)) {
      newSelected.delete(documentId);
    } else {
      newSelected.add(documentId);
    }
    setSelectedDocuments(newSelected);
  };

  // Handle select all
  const handleSelectAll = () => {
    if (selectedDocuments.size === filteredDocuments.length) {
      setSelectedDocuments(new Set());
    } else {
      setSelectedDocuments(new Set(filteredDocuments.map(doc => doc.id)));
    }
  };

  // Handle delete
  const handleDelete = (documentId) => {
    if (confirm('Are you sure you want to delete this document?')) {
      deleteMutation.mutate(documentId);
    }
  };

  // Handle bulk delete
  const handleBulkDelete = () => {
    if (selectedDocuments.size === 0) return;
    
    if (confirm(`Are you sure you want to delete ${selectedDocuments.size} document(s)?`)) {
      Array.from(selectedDocuments).forEach(documentId => {
        deleteMutation.mutate(documentId);
      });
      setSelectedDocuments(new Set());
    }
  };

  // Get status badge
  const getStatusBadge = (status) => {
    const statusConfig = {
      processed: { label: 'Ready', class: 'badge--success' },
      processing: { label: 'Processing', class: 'badge--warning' },
      uploaded: { label: 'Queued', class: 'badge--info' },
      failed: { label: 'Failed', class: 'badge--error' }
    };
    
    const config = statusConfig[status] || { label: status, class: 'badge--neutral' };
    return <span className={`badge ${config.class}`}>{config.label}</span>;
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Format date
  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (error) {
    return (
      <div className="document-manager__error">
        <h2>Failed to load documents</h2>
        <p>{error.message}</p>
        <button className="btn btn--primary" onClick={refetch}>
          <RefreshCw size={16} />
          Try Again
        </button>
      </div>
    );
  }

  return (
    <div className="document-manager">
      {/* Header */}
      <div className="document-manager__header">
        <div className="document-manager__header-content">
          <h1 className="document-manager__title">Document Manager</h1>
          <p className="document-manager__description">
            Upload, manage, and organize your PDF documents for AI analysis
          </p>
        </div>
        
        <button
          className="btn btn--primary"
          onClick={() => setShowUpload(true)}
        >
          <Upload size={16} />
          Upload Documents
        </button>
      </div>

      {/* Stats Overview */}
      {overview && (
        <div className="document-manager__stats">
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-card__icon">
                <FileText size={24} />
              </div>
              <div className="stat-card__content">
                <div className="stat-card__value">{overview.total_documents}</div>
                <div className="stat-card__label">Total Documents</div>
              </div>
            </div>
            
            <div className="stat-card">
              <div className="stat-card__icon">
                <BarChart3 size={24} />
              </div>
              <div className="stat-card__content">
                <div className="stat-card__value">{overview.total_pages || 0}</div>
                <div className="stat-card__label">Total Pages</div>
              </div>
            </div>
            
            <div className="stat-card">
              <div className="stat-card__icon">
                <Upload size={24} />
              </div>
              <div className="stat-card__content">
                <div className="stat-card__value">{formatFileSize(overview.total_size_bytes || 0)}</div>
                <div className="stat-card__label">Total Size</div>
              </div>
            </div>
            
            <div className="stat-card">
              <div className="stat-card__icon">
                <FileText size={24} />
              </div>
              <div className="stat-card__content">
                <div className="stat-card__value">
                  {overview.status_breakdown?.processed || 0}
                </div>
                <div className="stat-card__label">Ready for Chat</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="document-manager__controls">
        <div className="document-manager__controls-left">
          {/* Search */}
          <div className="search-input">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search documents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          {/* Filter */}
          <div className="filter-select">
            <Filter size={16} />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
            >
              <option value="">All Status</option>
              <option value="processed">Ready</option>
              <option value="processing">Processing</option>
              <option value="uploaded">Queued</option>
              <option value="failed">Failed</option>
            </select>
          </div>
        </div>

        <div className="document-manager__controls-right">
          {selectedDocuments.size > 0 && (
            <>
              <span className="selection-count">
                {selectedDocuments.size} selected
              </span>
              <button
                className="btn btn--danger btn--sm"
                onClick={handleBulkDelete}
                disabled={deleteMutation.isPending}
              >
                <Trash2 size={16} />
                Delete Selected
              </button>
            </>
          )}
          
          <button
            className="btn btn--ghost btn--sm"
            onClick={refetch}
            disabled={isLoading}
          >
            <RefreshCw size={16} className={isLoading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* Document List */}
      {isLoading ? (
        <div className="document-manager__loading">
          <LoadingSpinner size="lg" text="Loading documents..." centered />
        </div>
      ) : filteredDocuments.length === 0 ? (
        <div className="document-manager__empty">
          <FileText size={64} />
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
            <Plus size={16} />
            Upload Document
          </button>
        </div>
      ) : (
        <div className="document-table-container">
          <table className="document-table">
            <thead>
              <tr>
                <th className="document-table__checkbox-col">
                  <input
                    type="checkbox"
                    checked={selectedDocuments.size === filteredDocuments.length && filteredDocuments.length > 0}
                    onChange={handleSelectAll}
                    className="checkbox"
                  />
                </th>
                <th>Document</th>
                <th>Status</th>
                <th>Size</th>
                <th>Pages</th>
                <th>Uploaded</th>
                <th className="document-table__actions-col">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredDocuments.map((document) => (
                <tr
                  key={document.id}
                  className={selectedDocuments.has(document.id) ? 'document-table__row--selected' : ''}
                >
                  <td>
                    <input
                      type="checkbox"
                      checked={selectedDocuments.has(document.id)}
                      onChange={() => handleSelectDocument(document.id)}
                      className="checkbox"
                    />
                  </td>
                  
                  <td className="document-table__document-cell">
                    <div className="document-info">
                      <div className="document-info__icon">
                        <FileText size={20} />
                      </div>
                      <div className="document-info__content">
                        <div className="document-info__title">
                          {document.title || document.filename}
                        </div>
                        {document.description && (
                          <div className="document-info__description">
                            {document.description}
                          </div>
                        )}
                        <div className="document-info__filename">
                          {document.filename}
                        </div>
                      </div>
                    </div>
                  </td>
                  
                  <td>
                    {getStatusBadge(document.status)}
                  </td>
                  
                  <td>
                    {formatFileSize(document.file_size)}
                  </td>
                  
                  <td>
                    {document.page_count || '-'}
                  </td>
                  
                  <td>
                    {formatDate(document.created_at)}
                  </td>
                  
                  <td>
                    <div className="document-actions">
                      <button
                        className="btn btn--ghost btn--sm"
                        title="View Details"
                      >
                        <Eye size={16} />
                      </button>
                      
                      <button
                        className="btn btn--ghost btn--sm"
                        title="Download"
                      >
                        <Download size={16} />
                      </button>
                      
                      <button
                        className="btn btn--ghost btn--sm btn--danger"
                        onClick={() => handleDelete(document.id)}
                        disabled={deleteMutation.isPending}
                        title="Delete"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Upload Modal */}
      {showUpload && (
        <DocumentUpload
          onClose={() => setShowUpload(false)}
          onUploadSuccess={() => {
            setShowUpload(false);
            refetch();
            queryClient.invalidateQueries(['documents-overview']);
          }}
        />
      )}
    </div>
  );
};

export default DocumentManager;