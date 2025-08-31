import React, { useState } from 'react';
import { User, Bot, ChevronDown, ChevronUp, Clock, CheckCircle, AlertCircle } from 'lucide-react';
import { format } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';

import ReasoningTrace from './ReasoningTrace';
import SourceCitations from './SourceCitations';
import './MessageBubble.css';

const MessageBubble = ({ message }) => {
  const [showReasoningTrace, setShowReasoningTrace] = useState(false);
  const [showSources, setShowSources] = useState(false);

  const isUser = message.type === 'user';
  const isAI = message.type === 'ai';
  const hasError = message.error;
  const hasReasoningTrace = message.reasoning_trace && message.reasoning_trace.length > 0;
  const hasSources = message.sources && message.sources.length > 0;

  // Format timestamp
  const formatTimestamp = (timestamp) => {
    try {
      return format(new Date(timestamp), 'HH:mm');
    } catch {
      return '';
    }
  };

  // Custom markdown components
  const markdownComponents = {
    code({ node, inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter
          style={tomorrow}
          language={match[1]}
          PreTag="div"
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className={className} {...props}>
          {children}
        </code>
      );
    },
  };

  return (
    <div className={`message ${isUser ? 'message--user' : 'message--ai'}`}>
      {/* Avatar */}
      <div className={`message__avatar ${isAI ? 'message__avatar--ai' : ''}`}>
        {isUser ? <User size={16} /> : <Bot size={16} />}
      </div>

      {/* Message Content */}
      <div className="message__content">
        {/* Main Message Bubble */}
        <div className={`message__bubble ${hasError ? 'message__bubble--error' : ''}`}>
          {isUser ? (
            // User message - plain text
            <div className="message__text">
              {message.content}
            </div>
          ) : (
            // AI message - markdown support
            <div className="message__text">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={markdownComponents}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}

          {/* Error indicator */}
          {hasError && (
            <div className="message__error">
              <AlertCircle size={16} />
              <span>This response may contain errors</span>
            </div>
          )}
        </div>

        {/* AI Message Controls */}
        {isAI && (
          <div className="message__controls">
            {/* Reasoning Trace Toggle */}
            {hasReasoningTrace && (
              <button
                className="message__control-btn"
                onClick={() => setShowReasoningTrace(!showReasoningTrace)}
              >
                <Clock size={14} />
                <span>Reasoning ({message.reasoning_trace.length} steps)</span>
                {showReasoningTrace ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
            )}

            {/* Sources Toggle */}
            {hasSources && (
              <button
                className="message__control-btn"
                onClick={() => setShowSources(!showSources)}
              >
                <CheckCircle size={14} />
                <span>Sources ({message.sources.length})</span>
                {showSources ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
            )}
          </div>
        )}

        {/* Expandable Sections */}
        {isAI && (
          <>
            {/* Reasoning Trace */}
            {showReasoningTrace && hasReasoningTrace && (
              <div className="message__reasoning">
                <ReasoningTrace steps={message.reasoning_trace} />
              </div>
            )}

            {/* Source Citations */}
            {showSources && hasSources && (
              <div className="message__sources">
                <SourceCitations sources={message.sources} />
              </div>
            )}
          </>
        )}

        {/* Message Metadata */}
        <div className="message__meta">
          <span className="message__timestamp">
            {formatTimestamp(message.timestamp)}
          </span>

          {/* Status indicators */}
          {isAI && (
            <div className="message__status">
              {hasError ? (
                <span className="message__status-item message__status-item--error">
                  <AlertCircle size={12} />
                  Error
                </span>
              ) : (
                <span className="message__status-item message__status-item--success">
                  <CheckCircle size={12} />
                  Generated
                </span>
              )}

              {/* Performance info */}
              {message.response_time_ms && (
                <span className="message__status-item">
                  {Math.round(message.response_time_ms)}ms
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MessageBubble;