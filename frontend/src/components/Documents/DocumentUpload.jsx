import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation } from '@tanstack/react-query';
import { 
  Upload, 
  FileText, 
  X, 
  CheckCircle, 
  AlertCircle, 
  Loader2,
  Plus
} from 'lucide-react';
import toast from 'react-hot-toast';

import { documentsAPI } from '../../services/api';

// Import styles
import './DocumentUpload.css';

const DocumentUpload = ({ onClose, onUploadSuccess }) => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [currentFile, setCurrentFile] = useState(null);

  // Upload mutation
  const uploadMutation = useMutation({
    mutationFn: documentsAPI.uploadDocument,
    onSuccess: (data, variables) => {
      // Update file status
      setUploadedFiles(prev => prev.map(file => 
        file.id === variables.fileId 
          ? { ...file, status: 'uploaded', documentId: data.id }
          : file
      ));
      
      // Check if all files are uploaded
      const allUploaded = uploadedFiles.every(f => 
        f.id === variables.fileId || f.status === 'uploaded'
      );
      
      if (allUploaded) {
        setTimeout(() => {
          onUploadSuccess();
        }, 1000);
      }
    },
    onError: (error, variables) => {
      setUploadedFiles(prev => prev.map(file => 
        file.id === variables.fileId 
          ? { ...file, status: 'error', error: error.message }
          : file
      ));
      toast.error(`Upload failed: ${error.message}`);
    }
  });

  // Handle file drop
  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    // Handle rejected files
    rejectedFiles.forEach(({ file, errors }) => {
      errors.forEach(error => {
        if (error.code === 'file-too-large') {
          toast.error(`File "${file.name}" is too large. Maximum size is 50MB.`);
        } else if (error.code === 'file-invalid-type') {
          toast.error(`File "${file.name}" is not a PDF file.`);
        } else {
          toast.error(`File "${file.name}": ${error.message}`);
        }
      });
    });

    // Add accepted files
    const newFiles = acceptedFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      name: file.name,
      size: file.size,
      status: 'pending',
      title: '',
      description: ''
    }));

    setUploadedFiles(prev => [...prev, ...newFiles]);
  }, []);

  // Configure dropzone
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxSize: 50 * 1024 * 1024, // 50MB
    multiple: true
  });

  // Remove file
  const removeFile = (fileId) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  // Update file metadata
  const updateFileMetadata = (fileId, field, value) => {
    setUploadedFiles(prev => prev.map(file => 
      file.id === fileId ? { ...file, [field]: value } : file
    ));
  };

  // Upload single file
  const uploadFile = (file) => {
    setCurrentFile(file.id);
    
    const formData = new FormData();
    formData.append('file', file.file);
    formData.append('title', file.title || file.name.replace('.pdf', ''));
    formData.append('description', file.description || '');

    uploadMutation.mutate({
      formData,
      fileId: file.id
    });
  };

  // Upload all files
  const uploadAllFiles = () => {
    const pendingFiles = uploadedFiles.filter(f => f.status === 'pending');
    
    if (pendingFiles.length === 0) {
      toast.error('No files to upload');
      return;
    }

    // Update all files to uploading status
    setUploadedFiles(prev => prev.map(file => 
      file.status === 'pending' ? { ...file, status: 'uploading' } : file
    ));

    // Upload files sequentially
    pendingFiles.forEach((file, index) => {
      setTimeout(() => {
        uploadFile(file);
      }, index * 1000); // Stagger uploads by 1 second
    });
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Get file status info
  const getFileStatusInfo = (status) => {
    switch (status) {
      case 'pending':
        return { icon: FileText, color: 'text-gray-500', label: 'Ready to upload' };
      case 'uploading':
        return { icon: Loader2, color: 'text-blue-500', label: 'Uploading...', animate: true };
      case 'uploaded':
        return { icon: CheckCircle, color: 'text-green-500', label: 'Uploaded successfully' };
      case 'error':
        return { icon: AlertCircle, color: 'text-red-500', label: 'Upload failed' };
      default:
        return { icon: FileText, color: 'text-gray-500', label: status };
    }
  };

  const pendingFiles = uploadedFiles.filter(f => f.status === 'pending');
  const isUploading = uploadMutation.isPending;
  const allUploaded = uploadedFiles.length > 0 && uploadedFiles.every(f => f.status === 'uploaded');

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal modal--large" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="modal__header">
          <h2 className="modal__title">
            <Upload size={24} />
            Upload Documents
          </h2>
          <button className="modal__close" onClick={onClose}>
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="modal__content">
          {/* Dropzone */}
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? 'dropzone--active' : ''}`}
          >
            <input {...getInputProps()} />
            <div className="dropzone__icon">
              <Upload size={48} />
            </div>
            <div className="dropzone__content">
              <h3 className="dropzone__title">
                {isDragActive ? 'Drop PDF files here' : 'Upload PDF Documents'}
              </h3>
              <p className="dropzone__subtitle">
                Drag and drop PDF files here, or click to select files
              </p>
              <div className="dropzone__specs">
                <span>• PDF files only</span>
                <span>• Maximum 50MB per file</span>
                <span>• Multiple files supported</span>
              </div>
            </div>
          </div>

          {/* File List */}
          {uploadedFiles.length > 0 && (
            <div className="upload-files">
              <div className="upload-files__header">
                <h3 className="upload-files__title">
                  Files ({uploadedFiles.length})
                </h3>
                
                {pendingFiles.length > 0 && (
                  <button
                    className="btn btn--primary btn--sm"
                    onClick={uploadAllFiles}
                    disabled={isUploading}
                  >
                    {isUploading ? (
                      <>
                        <Loader2 size={16} className="animate-spin" />
                        Uploading...
                      </>
                    ) : (
                      <>
                        <Upload size={16} />
                        Upload All ({pendingFiles.length})
                      </>
                    )}
                  </button>
                )}
              </div>

              <div className="upload-files__list">
                {uploadedFiles.map((file) => {
                  const statusInfo = getFileStatusInfo(file.status);
                  const StatusIcon = statusInfo.icon;

                  return (
                    <div key={file.id} className="upload-file">
                      <div className="upload-file__main">
                        <div className="upload-file__icon">
                          <FileText size={24} />
                        </div>

                        <div className="upload-file__info">
                          <div className="upload-file__name">
                            {file.name}
                          </div>
                          <div className="upload-file__meta">
                            <span>{formatFileSize(file.size)}</span>
                            <span className={statusInfo.color}>
                              <StatusIcon 
                                size={16} 
                                className={statusInfo.animate ? 'animate-spin' : ''} 
                              />
                              {statusInfo.label}
                            </span>
                          </div>
                        </div>

                        <div className="upload-file__actions">
                          {file.status === 'pending' && (
                            <>
                              <button
                                className="btn btn--ghost btn--sm"
                                onClick={() => uploadFile(file)}
                                disabled={isUploading}
                              >
                                <Upload size={16} />
                                Upload
                              </button>
                              <button
                                className="btn btn--ghost btn--sm"
                                onClick={() => removeFile(file.id)}
                              >
                                <X size={16} />
                              </button>
                            </>
                          )}
                        </div>
                      </div>

                      {/* Metadata Form */}
                      {file.status === 'pending' && (
                        <div className="upload-file__metadata">
                          <div className="input-group">
                            <label className="input-label">Title (optional)</label>
                            <input
                              type="text"
                              className="input input--sm"
                              placeholder="Document title"
                              value={file.title}
                              onChange={(e) => updateFileMetadata(file.id, 'title', e.target.value)}
                            />
                          </div>

                          <div className="input-group">
                            <label className="input-label">Description (optional)</label>
                            <textarea
                              className="input textarea input--sm"
                              placeholder="Brief description of the document"
                              value={file.description}
                              onChange={(e) => updateFileMetadata(file.id, 'description', e.target.value)}
                              rows={2}
                            />
                          </div>
                        </div>
                      )}

                      {/* Error Message */}
                      {file.status === 'error' && file.error && (
                        <div className="upload-file__error">
                          <AlertCircle size={16} />
                          <span>{file.error}</span>
                          <button
                            className="btn btn--ghost btn--sm"
                            onClick={() => {
                              setUploadedFiles(prev => prev.map(f => 
                                f.id === file.id ? { ...f, status: 'pending', error: null } : f
                              ));
                            }}
                          >
                            Retry
                          </button>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Success Message */}
          {allUploaded && (
            <div className="upload-success">
              <CheckCircle size={48} />
              <h3>Upload Complete!</h3>
              <p>All documents have been uploaded successfully and are being processed.</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="modal__footer">
          <button className="btn btn--ghost" onClick={onClose}>
            {allUploaded ? 'Close' : 'Cancel'}
          </button>
          
          {!allUploaded && (
            <button
              className="btn btn--ghost btn--sm"
              onClick={() => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.pdf';
                input.multiple = true;
                input.onchange = (e) => {
                  const files = Array.from(e.target.files);
                  onDrop(files, []);
                };
                input.click();
              }}
            >
              <Plus size={16} />
              Add More Files
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default DocumentUpload;