import React, { useState, useRef, useEffect } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Send, FileText, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import toast from 'react-hot-toast';

// Components
import MessageBubble from '../../components/Chat/MessageBubble';
import ReasoningTrace from '../../components/Chat/ReasoningTrace';
import DocumentSelector from '../../components/Documents/DocumentSelector';
import LoadingSpinner from '../../components/UI/LoadingSpinner';

// Services
import { chatAPI } from '../../services/api';

// Styles
import './ChatInterface.css';

const ChatInterface = () => {
  // State management
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [showDocumentSelector, setShowDocumentSelector] = useState(false);
  const [conversationId, setConversationId] = useState(null);

  // Refs
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom when new messages arrive
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Chat mutation
  const chatMutation = useMutation({
    mutationFn: chatAPI.sendMessage,
    onSuccess: (data) => {
      // Add AI response to messages
      const aiMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: data.response,
        reasoning_trace: data.reasoning_trace,
        sources: data.sources,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, aiMessage]);
      
      // Update conversation ID if it's a new conversation
      if (data.conversation_id && !conversationId) {
        setConversationId(data.conversation_id);
      }

      toast.success('Response generated successfully');
    },
    onError: (error) => {
      console.error('Chat error:', error);
      toast.error('Failed to get response. Please try again.');
      
      // Add error message
      const errorMessage = {
        id: Date.now() + 1,
        type: 'ai',
        content: 'I apologize, but I encountered an error. Please try again.',
        error: true,
        timestamp: new Date().toISOString(),
      };
      
      setMessages(prev => [...prev, errorMessage]);
    },
  });

  // Handle form submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!inputValue.trim()) return;
    
    if (selectedDocuments.length === 0) {
      toast.error('Please select at least one document to chat with');
      setShowDocumentSelector(true);
      return;
    }

    // Add user message to state
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);

    // Send to API
    chatMutation.mutate({
      message: inputValue.trim(),
      document_ids: selectedDocuments.map(doc => doc.id),
      conversation_id: conversationId,
    });

    // Clear input
    setInputValue('');
  };

  // Handle input changes
  const handleInputChange = (e) => {
    setInputValue(e.target.value);
  };

  // Handle key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Handle document selection
  const handleDocumentSelection = (documents) => {
    setSelectedDocuments(documents);
    setShowDocumentSelector(false);
    
    if (documents.length > 0) {
      toast.success(`Selected ${documents.length} document(s)`);
    }
  };

  return (
    <div className="chat-container">
      {/* Document Selector Sidebar */}
      <div className={`chat-sidebar ${showDocumentSelector ? 'chat-sidebar--visible' : ''}`}>
        <div className="chat-sidebar__header">
          <h3 className="chat-sidebar__title">
            <FileText size={20} />
            Documents
          </h3>
          <button
            className="btn btn--ghost btn--sm"
            onClick={() => setShowDocumentSelector(!showDocumentSelector)}
          >
            {showDocumentSelector ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>
        
        {showDocumentSelector && (
          <div className="chat-sidebar__content">
            <DocumentSelector
              selectedDocuments={selectedDocuments}
              onSelectionChange={handleDocumentSelection}
            />
          </div>
        )}
        
        {/* Selected Documents Summary */}
        {selectedDocuments.length > 0 && (
          <div className="chat-sidebar__selected">
            <h4 className="chat-sidebar__selected-title">
              Selected ({selectedDocuments.length})
            </h4>
            <div className="chat-sidebar__selected-list">
              {selectedDocuments.map((doc) => (
                <div key={doc.id} className="chat-sidebar__selected-item">
                  <span className="chat-sidebar__selected-name">
                    {doc.title || doc.filename}
                  </span>
                  <button
                    className="btn btn--ghost btn--sm"
                    onClick={() => {
                      setSelectedDocuments(prev => 
                        prev.filter(d => d.id !== doc.id)
                      );
                    }}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Main Chat Area */}
      <div className="chat-main">
        {/* Chat Header */}
        <div className="chat-header">
          <div className="chat-header__actions">
            <button
              className="btn btn--outline btn--sm"
              onClick={() => setShowDocumentSelector(!showDocumentSelector)}
            >
              <FileText size={16} />
              {selectedDocuments.length > 0 
                ? `${selectedDocuments.length} Selected` 
                : 'Select Documents'
              }
            </button>
            
            <button
              className="btn btn--ghost btn--sm"
              onClick={() => {
                setMessages([]);
                setConversationId(null);
                toast.success('Chat cleared');
              }}
            >
              Clear Chat
            </button>
          </div>
        </div>

        {/* Chat Messages */}
        <div className="chat-messages">
          {messages.length === 0 ? (
            <div className="chat-welcome">
              <div className="chat-welcome__content">
                <h2 className="chat-welcome__title">Welcome to PDF Sage</h2>
                <p className="chat-welcome__description">
                  Select documents and start asking questions to get intelligent, 
                  source-backed answers with transparent reasoning.
                </p>
                
                {selectedDocuments.length === 0 ? (
                  <button
                    className="btn btn--primary"
                    onClick={() => setShowDocumentSelector(true)}
                  >
                    <FileText size={20} />
                    Select Documents to Get Started
                  </button>
                ) : (
                  <div className="chat-welcome__suggestions">
                    <h3>Try asking:</h3>
                    <ul>
                      <li>"What is the main topic of these documents?"</li>
                      <li>"Summarize the key findings"</li>
                      <li>"What are the most important points?"</li>
                      <li>"How do these documents relate to each other?"</li>
                    </ul>
                  </div>
                )}
              </div>
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <MessageBubble
                  key={message.id}
                  message={message}
                />
              ))}
              
              {/* Loading indicator */}
              {chatMutation.isPending && (
                <div className="message message--ai">
                  <div className="message__avatar message__avatar--ai">
                    AI
                  </div>
                  <div className="message__content">
                    <div className="message__bubble">
                      <div className="message__typing">
                        <LoadingSpinner size="sm" />
                        <span>Thinking and searching...</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Chat Input */}
        <div className="chat-input-container">
          <form onSubmit={handleSubmit} className="chat-input-wrapper">
            <textarea
              ref={inputRef}
              className="chat-input"
              placeholder={
                selectedDocuments.length === 0
                  ? "Select documents first, then ask your question..."
                  : "Ask a question about your documents..."
              }
              value={inputValue}
              onChange={handleInputChange}
              onKeyPress={handleKeyPress}
              disabled={chatMutation.isPending || selectedDocuments.length === 0}
              rows={1}
              style={{
                minHeight: 'var(--chat-input-height)',
                height: 'auto',
                resize: 'none',
              }}
              onInput={(e) => {
                // Auto-resize textarea
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
              }}
            />
            
            <button
              type="submit"
              className="btn btn--primary chat-send-btn"
              disabled={
                !inputValue.trim() || 
                chatMutation.isPending || 
                selectedDocuments.length === 0
              }
            >
              {chatMutation.isPending ? (
                <Loader2 size={20} className="animate-spin" />
              ) : (
                <Send size={20} />
              )}
            </button>
          </form>
          
          {/* Input hints */}
          <div className="chat-input-hints">
            <span className="chat-input-hint">
              Press Enter to send, Shift+Enter for new line
            </span>
            {selectedDocuments.length > 0 && (
              <span className="chat-input-hint">
                Searching in {selectedDocuments.length} document(s)
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;