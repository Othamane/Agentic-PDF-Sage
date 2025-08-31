/**
 * API service for Agentic PDF Sage frontend
 * Handles all communication with the backend API
 */

import axios from 'axios';
import toast from 'react-hot-toast';

// Get backend URL from environment
const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

// Create axios instance with default configuration
const api = axios.create({
  baseURL: `${BACKEND_URL}/api/v1`,
  timeout: 120000, // 2 minutes timeout for large file uploads
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth headers here if needed in the future
    // config.headers.Authorization = `Bearer ${token}`;
    
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API] Request error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    console.log(`[API] Response ${response.status} from ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('[API] Response error:', error);
    
    // Handle common error scenarios
    if (error.response) {
      // Server responded with error status
      const { status, data } = error.response;
      
      switch (status) {
        case 400:
          toast.error(data.detail || 'Bad request');
          break;
        case 401:
          toast.error('Unauthorized access');
          break;
        case 403:
          toast.error('Access forbidden');
          break;
        case 404:
          toast.error('Resource not found');
          break;
        case 413:
          toast.error('File too large');
          break;
        case 422:
          toast.error(data.detail || 'Validation error');
          break;
        case 429:
          toast.error('Too many requests. Please slow down.');
          break;
        case 500:
          toast.error('Internal server error');
          break;
        default:
          toast.error(data.detail || `Error ${status}`);
      }
    } else if (error.request) {
      // Network error
      toast.error('Network error. Please check your connection.');
    } else {
      // Other error
      toast.error('An unexpected error occurred');
    }
    
    return Promise.reject(error);
  }
);

// Helper function to handle API errors
const handleApiError = (error, defaultMessage = 'An error occurred') => {
  if (error.response?.data?.detail) {
    throw new Error(error.response.data.detail);
  }
  throw new Error(defaultMessage);
};

// ===== CHAT API =====
export const chatAPI = {
  /**
   * Send a message to the chat API
   */
  sendMessage: async ({ message, document_ids, conversation_id, max_iterations = 3 }) => {
    try {
      const response = await api.post('/chat/send', {
        message,
        document_ids,
        conversation_id,
        max_iterations , 
      },
   { timeout : 300000}
    );
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to send message');
    }
  },

  /**
   * Get chat status
   */
  getStatus: async () => {
    try {
      const response = await api.get('/chat/status');
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to get chat status');
    }
  },

  /**
   * List conversations
   */
  listConversations: async ({ limit = 50, offset = 0, user_id } = {}) => {
    try {
      const params = { limit, offset };
      if (user_id) params.user_id = user_id;
      
      const response = await api.get('/chat/conversations', { params });
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to fetch conversations');
    }
  },

  /**
   * Get a specific conversation
   */
  getConversation: async (conversationId) => {
    try {
      const response = await api.get(`/chat/conversations/${conversationId}`);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to fetch conversation');
    }
  },

  /**
   * Delete a conversation
   */
  deleteConversation: async (conversationId) => {
    try {
      const response = await api.delete(`/chat/conversations/${conversationId}`);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to delete conversation');
    }
  }
};

// ===== DOCUMENTS API =====
export const documentsAPI = {
  /**
   * Upload a document
   */
  uploadDocument: async ({ formData }) => {
    try {
      const response = await api.post('/documents/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minutes for large uploads
      });
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to upload document');
    }
  },

  /**
   * List documents
   */
  listDocuments: async ({ 
    user_id, 
    status, 
    limit = 50, 
    offset = 0 
  } = {}) => {
    try {
      const params = { limit, offset };
      if (user_id) params.user_id = user_id;
      if (status) params.status = status;
      
      const response = await api.get('/documents/', { params });
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to fetch documents');
    }
  },

  /**
   * Get a specific document
   */
  getDocument: async (documentId) => {
    try {
      const response = await api.get(`/documents/${documentId}`);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to fetch document');
    }
  },

  /**
   * Delete a document
   */
  deleteDocument: async (documentId) => {
    try {
      const response = await api.delete(`/documents/${documentId}`);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to delete document');
    }
  },

  /**
   * Get document content
   */
  getDocumentContent: async (documentId) => {
    try {
      const response = await api.get(`/documents/${documentId}/content`);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to fetch document content');
    }
  },

  /**
   * Search documents
   */
  searchDocuments: async ({ query, document_ids, k = 10 }) => {
    try {
      const response = await api.post('/documents/search', {
        query,
        document_ids,
        k
      });
      return response.data;
    } catch (error) {
      handleApiError(error, 'Search failed');
    }
  },

  /**
   * Get document status
   */
  getDocumentStatus: async (documentId) => {
    try {
      const response = await api.get(`/documents/${documentId}/status`);
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to fetch document status');
    }
  },

  /**
   * Get documents overview/stats
   */
  getDocumentsOverview: async () => {
    try {
      const response = await api.get('/documents/stats/overview');
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to fetch documents overview');
    }
  }
};

// ===== HEALTH API =====
export const healthAPI = {
  /**
   * Basic health check
   */
  checkHealth: async () => {
    try {
      const response = await api.get('/health/');
      return response.data;
    } catch (error) {
      handleApiError(error, 'Health check failed');
    }
  },

  /**
   * Detailed health check
   */
  checkDetailedHealth: async () => {
    try {
      const response = await api.get('/health/detailed');
      return response.data;
    } catch (error) {
      handleApiError(error, 'Detailed health check failed');
    }
  },

  /**
   * Get metrics
   */
  getMetrics: async () => {
    try {
      const response = await api.get('/health/metrics');
      return response.data;
    } catch (error) {
      handleApiError(error, 'Failed to fetch metrics');
    }
  }
};

// ===== UTILITY FUNCTIONS =====

/**
 * Test backend connectivity
 */
export const testConnection = async () => {
  try {
    const response = await axios.get(`${BACKEND_URL}/health`, {
      timeout: 5000
    });
    return response.status === 200;
  } catch (error) {
    console.error('Backend connectivity test failed:', error);
    return false;
  }
};

/**
 * Get backend URL
 */
export const getBackendUrl = () => BACKEND_URL;

/**
 * Format API error for display
 */
export const formatApiError = (error) => {
  if (error.response?.data?.detail) {
    return error.response.data.detail;
  }
  if (error.message) {
    return error.message;
  }
  return 'An unexpected error occurred';
};

// Export the configured axios instance for direct use if needed
export { api };

// Default export with all APIs
export default {
  chat: chatAPI,
  documents: documentsAPI,
  health: healthAPI,
  testConnection,
  getBackendUrl,
  formatApiError
};