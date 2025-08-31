import React from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';
import './ErrorBoundary.css'
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to an error reporting service
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    this.setState({
      error: error,
      errorInfo: errorInfo
    });

    // In production, you would send this to an error monitoring service
    if (process.env.NODE_ENV === 'production') {
      // Example: Sentry.captureException(error, { extra: errorInfo });
    }
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  handleGoHome = () => {
    window.location.href = '/';
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <div className="error-boundary__container">
            <div className="error-boundary__icon">
              <AlertTriangle size={64} />
            </div>
            
            <div className="error-boundary__content">
              <h1 className="error-boundary__title">
                Something went wrong
              </h1>
              
              <p className="error-boundary__description">
                We're sorry, but something unexpected happened. 
                The error has been logged and we'll look into it.
              </p>

              <div className="error-boundary__actions">
                <button 
                  className="btn btn--primary"
                  onClick={this.handleRetry}
                >
                  <RefreshCw size={16} />
                  Try Again
                </button>
                
                <button 
                  className="btn btn--outline"
                  onClick={this.handleGoHome}
                >
                  <Home size={16} />
                  Go Home
                </button>
              </div>

              {/* Error Details (only in development) */}
              {process.env.NODE_ENV === 'development' && this.state.error && (
                <details className="error-boundary__details">
                  <summary className="error-boundary__details-summary">
                    Error Details (Development Only)
                  </summary>
                  
                  <div className="error-boundary__error-info">
                    <h3>Error:</h3>
                    <pre className="error-boundary__error-text">
                      {this.state.error && this.state.error.toString()}
                    </pre>
                    
                    <h3>Component Stack:</h3>
                    <pre className="error-boundary__error-text">
                      {this.state.errorInfo.componentStack}
                    </pre>
                  </div>
                </details>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;