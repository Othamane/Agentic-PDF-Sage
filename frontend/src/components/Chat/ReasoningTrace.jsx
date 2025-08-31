import React, { useState } from 'react';
import { ChevronRight, ChevronDown, Clock, Search, Lightbulb, CheckCircle, AlertTriangle } from 'lucide-react';
import { format } from 'date-fns';
import './ReasoningTrace.css'

const ReasoningTrace = ({ steps }) => {
  const [expandedSteps, setExpandedSteps] = useState(new Set());

  // Toggle step expansion
  const toggleStep = (index) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSteps(newExpanded);
  };

  // Get icon for step type
  const getStepIcon = (stepType) => {
    switch (stepType) {
      case 'planning':
        return <Lightbulb size={16} />;
      case 'retrieval':
        return <Search size={16} />;
      case 'synthesis':
        return <CheckCircle size={16} />;
      case 'validation':
        return <CheckCircle size={16} />;
      case 'error':
        return <AlertTriangle size={16} />;
      default:
        return <Clock size={16} />;
    }
  };

  // Get step color class
  const getStepColorClass = (stepType) => {
    switch (stepType) {
      case 'planning':
        return 'reasoning-step--planning';
      case 'retrieval':
        return 'reasoning-step--retrieval';
      case 'synthesis':
        return 'reasoning-step--synthesis';
      case 'validation':
        return 'reasoning-step--validation';
      case 'error':
        return 'reasoning-step--error';
      default:
        return 'reasoning-step--default';
    }
  };

  // Format duration
  const formatDuration = (durationMs) => {
    if (!durationMs) return '';
    if (durationMs < 1000) {
      return `${Math.round(durationMs)}ms`;
    }
    return `${(durationMs / 1000).toFixed(1)}s`;
  };

  // Format step title
  const formatStepTitle = (step) => {
    const titles = {
      planning: 'Planning & Analysis',
      retrieval: 'Information Retrieval',
      synthesis: 'Response Generation',
      validation: 'Quality Validation',
      error: 'Error Handling'
    };
    return titles[step.step] || step.step.charAt(0).toUpperCase() + step.step.slice(1);
  };

  if (!steps || steps.length === 0) {
    return (
      <div className="reasoning-trace reasoning-trace--empty">
        <p>No reasoning steps available</p>
      </div>
    );
  }

  return (
    <div className="reasoning-trace">
      <div className="reasoning-trace__header">
        <h4 className="reasoning-trace__title">
          <Clock size={16} />
          Agent Reasoning Process
        </h4>
        <span className="reasoning-trace__summary">
          {steps.length} steps completed
        </span>
      </div>

      <div className="reasoning-trace__steps">
        {steps.map((step, index) => {
          const isExpanded = expandedSteps.has(index);
          const hasError = step.error;
          const stepColorClass = getStepColorClass(step.step);

          return (
            <div
              key={index}
              className={`reasoning-step ${stepColorClass} ${hasError ? 'reasoning-step--error' : ''}`}
            >
              {/* Step Header */}
              <div
                className="reasoning-step__header"
                onClick={() => toggleStep(index)}
              >
                <div className="reasoning-step__header-left">
                  <div className="reasoning-step__icon">
                    {getStepIcon(step.step)}
                  </div>
                  <div className="reasoning-step__title-section">
                    <h5 className="reasoning-step__title">
                      {formatStepTitle(step)}
                    </h5>
                    {step.reasoning && (
                      <p className="reasoning-step__summary">
                        {step.reasoning.substring(0, 100)}
                        {step.reasoning.length > 100 ? '...' : ''}
                      </p>
                    )}
                  </div>
                </div>

                <div className="reasoning-step__header-right">
                  {step.duration_ms && (
                    <span className="reasoning-step__duration">
                      {formatDuration(step.duration_ms)}
                    </span>
                  )}
                  <button className="reasoning-step__toggle">
                    {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                  </button>
                </div>
              </div>

              {/* Step Content */}
              {isExpanded && (
                <div className="reasoning-step__content">
                  {/* Full reasoning */}
                  {step.reasoning && (
                    <div className="reasoning-step__section">
                      <h6 className="reasoning-step__section-title">Reasoning</h6>
                      <div className="reasoning-step__text">
                        {step.reasoning}
                      </div>
                    </div>
                  )}

                  {/* Input data */}
                  {step.input && (
                    <div className="reasoning-step__section">
                      <h6 className="reasoning-step__section-title">Input</h6>
                      <div className="reasoning-step__data">
                        <pre>{JSON.stringify(step.input, null, 2)}</pre>
                      </div>
                    </div>
                  )}

                  {/* Output data */}
                  {step.output && (
                    <div className="reasoning-step__section">
                      <h6 className="reasoning-step__section-title">Output</h6>
                      <div className="reasoning-step__data">
                        {typeof step.output === 'string' ? (
                          <div className="reasoning-step__text">{step.output}</div>
                        ) : (
                          <pre>{JSON.stringify(step.output, null, 2)}</pre>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Error information */}
                  {hasError && (
                    <div className="reasoning-step__section reasoning-step__section--error">
                      <h6 className="reasoning-step__section-title">Error</h6>
                      <div className="reasoning-step__error">
                        <AlertTriangle size={16} />
                        <span>{step.error}</span>
                      </div>
                    </div>
                  )}

                  {/* Metadata */}
                  <div className="reasoning-step__metadata">
                    <div className="reasoning-step__metadata-item">
                      <strong>Step:</strong> {index + 1} of {steps.length}
                    </div>
                    <div className="reasoning-step__metadata-item">
                      <strong>Type:</strong> {step.step}
                    </div>
                    {step.duration_ms && (
                      <div className="reasoning-step__metadata-item">
                        <strong>Duration:</strong> {formatDuration(step.duration_ms)}
                      </div>
                    )}
                    {step.timestamp && (
                      <div className="reasoning-step__metadata-item">
                        <strong>Time:</strong> {format(new Date(step.timestamp), 'HH:mm:ss.SSS')}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Summary Statistics */}
      <div className="reasoning-trace__summary-stats">
        <div className="reasoning-trace__stat">
          <span className="reasoning-trace__stat-label">Total Steps:</span>
          <span className="reasoning-trace__stat-value">{steps.length}</span>
        </div>
        
        <div className="reasoning-trace__stat">
          <span className="reasoning-trace__stat-label">Total Time:</span>
          <span className="reasoning-trace__stat-value">
            {formatDuration(
              steps.reduce((total, step) => total + (step.duration_ms || 0), 0)
            )}
          </span>
        </div>

        <div className="reasoning-trace__stat">
          <span className="reasoning-trace__stat-label">Success Rate:</span>
          <span className="reasoning-trace__stat-value">
            {Math.round((steps.filter(s => !s.error).length / steps.length) * 100)}%
          </span>
        </div>
      </div>
    </div>
  );
};

export default ReasoningTrace;