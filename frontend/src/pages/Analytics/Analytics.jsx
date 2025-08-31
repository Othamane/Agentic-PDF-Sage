import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  BarChart3, 
  TrendingUp, 
  Clock, 
  MessageSquare,
  FileText,
  Users,
  Activity
} from 'lucide-react';

import LoadingSpinner from '../../components/UI/LoadingSpinner';
import { documentsAPI, healthAPI } from '../../services/api';
import './Analytics.css'
const Analytics = () => {
  // Fetch documents overview
  const { data: documentsOverview, isLoading: documentsLoading } = useQuery({
    queryKey: ['documents-overview'],
    queryFn: documentsAPI.getDocumentsOverview,
    refetchInterval: 60000,
  });

  // Fetch system metrics
  const { data: systemMetrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['system-metrics'],
    queryFn: healthAPI.getMetrics,
    refetchInterval: 30000,
  });

  // Fetch detailed health
  const { data: healthStatus, isLoading: healthLoading } = useQuery({
    queryKey: ['health-detailed'],
    queryFn: healthAPI.checkDetailedHealth,
    refetchInterval: 30000,
  });

  const isLoading = documentsLoading || metricsLoading || healthLoading;

  // Calculate processing success rate
  const getProcessingSuccessRate = () => {
    if (!documentsOverview?.status_breakdown) return 0;
    const total = documentsOverview.total_documents;
    const processed = documentsOverview.status_breakdown.processed || 0;
    const failed = documentsOverview.status_breakdown.failed || 0;
    const completed = processed + failed;
    
    if (completed === 0) return 0;
    return Math.round((processed / completed) * 100);
  };

  // Format bytes
  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Format percentage
  const formatPercentage = (value) => {
    return `${Math.round(value)}%`;
  };

  if (isLoading) {
    return (
      <div className="analytics__loading">
        <LoadingSpinner size="lg" text="Loading analytics..." centered />
      </div>
    );
  }

  return (
    <div className="analytics">
      {/* Header */}
      <div className="analytics__header">
        <h1 className="analytics__title">Analytics Dashboard</h1>
        <p className="analytics__description">
          Monitor system performance, document processing, and usage statistics
        </p>
      </div>

      {/* Key Metrics */}
      <div className="analytics__section">
        <h2 className="analytics__section-title">Key Metrics</h2>
        
        <div className="metrics-grid">
          <div className="metric-card metric-card--primary">
            <div className="metric-card__icon">
              <FileText size={32} />
            </div>
            <div className="metric-card__content">
              <div className="metric-card__value">
                {documentsOverview?.total_documents || 0}
              </div>
              <div className="metric-card__label">Total Documents</div>
              <div className="metric-card__change">
                {documentsOverview?.status_breakdown?.processed || 0} ready for chat
              </div>
            </div>
          </div>

          <div className="metric-card metric-card--success">
            <div className="metric-card__icon">
              <TrendingUp size={32} />
            </div>
            <div className="metric-card__content">
              <div className="metric-card__value">
                {getProcessingSuccessRate()}%
              </div>
              <div className="metric-card__label">Success Rate</div>
              <div className="metric-card__change">
                Processing success rate
              </div>
            </div>
          </div>

          <div className="metric-card metric-card--info">
            <div className="metric-card__icon">
              <BarChart3 size={32} />
            </div>
            <div className="metric-card__content">
              <div className="metric-card__value">
                {documentsOverview?.total_pages || 0}
              </div>
              <div className="metric-card__label">Total Pages</div>
              <div className="metric-card__change">
                Across all documents
              </div>
            </div>
          </div>

          <div className="metric-card metric-card--warning">
            <div className="metric-card__icon">
              <Activity size={32} />
            </div>
            <div className="metric-card__content">
              <div className="metric-card__value">
                {formatBytes(documentsOverview?.total_size_bytes || 0)}
              </div>
              <div className="metric-card__label">Storage Used</div>
              <div className="metric-card__change">
                Total file storage
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Document Status Breakdown */}
      <div className="analytics__section">
        <h2 className="analytics__section-title">Document Status</h2>
        
        <div className="status-breakdown">
          <div className="status-breakdown__chart">
            {documentsOverview?.status_breakdown && (
              <div className="status-chart">
                {Object.entries(documentsOverview.status_breakdown).map(([status, count]) => {
                  const total = documentsOverview.total_documents;
                  const percentage = total > 0 ? (count / total) * 100 : 0;
                  
                  const statusConfig = {
                    processed: { label: 'Ready', color: 'var(--color-success-500)' },
                    processing: { label: 'Processing', color: 'var(--color-warning-500)' },
                    uploaded: { label: 'Queued', color: 'var(--color-info-500)' },
                    failed: { label: 'Failed', color: 'var(--color-error-500)' }
                  };
                  
                  const config = statusConfig[status] || { label: status, color: 'var(--color-neutral-500)' };
                  
                  return (
                    <div key={status} className="status-item">
                      <div className="status-item__bar">
                        <div 
                          className="status-item__fill"
                          style={{ 
                            width: `${percentage}%`,
                            backgroundColor: config.color
                          }}
                        />
                      </div>
                      <div className="status-item__info">
                        <span className="status-item__label">{config.label}</span>
                        <span className="status-item__value">{count} ({Math.round(percentage)}%)</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* System Health */}
      <div className="analytics__section">
        <h2 className="analytics__section-title">System Health</h2>
        
        <div className="health-grid">
          {/* System Resources */}
          <div className="health-card">
            <h3 className="health-card__title">
              <Activity size={20} />
              System Resources
            </h3>
            
            {systemMetrics && (
              <div className="health-metrics">
                <div className="health-metric">
                  <span className="health-metric__label">CPU Usage</span>
                  <div className="health-metric__bar">
                    <div 
                      className="health-metric__fill"
                      style={{ width: `${systemMetrics.system_cpu_usage_percent || 0}%` }}
                    />
                  </div>
                  <span className="health-metric__value">
                    {formatPercentage(systemMetrics.system_cpu_usage_percent || 0)}
                  </span>
                </div>
                
                <div className="health-metric">
                  <span className="health-metric__label">Memory Usage</span>
                  <div className="health-metric__bar">
                    <div 
                      className="health-metric__fill"
                      style={{ width: `${systemMetrics.system_memory_usage_percent || 0}%` }}
                    />
                  </div>
                  <span className="health-metric__value">
                    {formatPercentage(systemMetrics.system_memory_usage_percent || 0)}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Component Status */}
          <div className="health-card">
            <h3 className="health-card__title">
              <Users size={20} />
              Component Status
            </h3>
            
            {healthStatus?.components && (
              <div className="component-status">
                {Object.entries(healthStatus.components).map(([component, status]) => (
                  <div key={component} className="component-item">
                    <div className={`component-status-dot component-status-dot--${status.status}`} />
                    <span className="component-name">{component.replace('_', ' ')}</span>
                    <span className={`component-status-text component-status-text--${status.status}`}>
                      {status.status}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Performance Metrics */}
          <div className="health-card">
            <h3 className="health-card__title">
              <Clock size={20} />
              Performance
            </h3>
            
            {healthStatus?.components && (
              <div className="performance-metrics">
                {Object.entries(healthStatus.components).map(([component, status]) => (
                  status.response_time_ms && (
                    <div key={component} className="performance-item">
                      <span className="performance-label">{component}</span>
                      <span className="performance-value">
                        {Math.round(status.response_time_ms)}ms
                      </span>
                    </div>
                  )
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="analytics__section">
        <h2 className="analytics__section-title">Recent Activity</h2>
        
        <div className="activity-card">
          <div className="activity-placeholder">
            <MessageSquare size={48} />
            <h3>Activity Tracking</h3>
            <p>
              Detailed activity logs and conversation analytics will be available here.
              This feature tracks user interactions, popular documents, and usage patterns.
            </p>
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="analytics__footer">
        <div className="analytics__footer-content">
          <p className="analytics__footer-text">
            Analytics data is refreshed every 30-60 seconds. 
            All processing happens locally with no external data sharing.
          </p>
          <div className="analytics__footer-status">
            <div className="status-indicator status-indicator--online">
              <div className="status-dot"></div>
              <span>System Online</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;