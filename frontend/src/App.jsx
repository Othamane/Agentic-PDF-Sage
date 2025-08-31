import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';

// Styles
import './styles/variables.css';
import './styles/globals.css';
import './styles/components.css';

// Components
import Layout from './components/Layout/Layout';
import ChatInterface from './pages/ChatInterface/ChatInterface';
import DocumentManager from './pages/DocumentManager/DocumentManager';
import Analytics from './pages/Analytics/Analytics';
import ErrorBoundary from './components/ErrorBoundary/ErrorBoundary';

// Create QueryClient for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 3,
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      staleTime: 5 * 60 * 1000, // 5 minutes
      cacheTime: 10 * 60 * 1000, // 10 minutes
    },
    mutations: {
      retry: 1,
    },
  },
});

function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <Router>
          <div className="app" data-theme="light">
            <Layout>
              <Routes>
                <Route path="/" element={<ChatInterface />} />
                <Route path="/chat" element={<ChatInterface />} />
                <Route path="/documents" element={<DocumentManager />} />
                <Route path="/analytics" element={<Analytics />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Layout>
            
            {/* Global Toast Notifications */}
            <Toaster
              position="top-right"
              reverseOrder={false}
              gutter={8}
              containerClassName=""
              containerStyle={{}}
              toastOptions={{
                // Default options for all toasts
                className: '',
                duration: 4000,
                style: {
                  background: 'var(--color-bg-primary)',
                  color: 'var(--color-text-primary)',
                  border: '1px solid var(--color-border)',
                  borderRadius: 'var(--radius-lg)',
                  fontSize: 'var(--font-size-sm)',
                  padding: 'var(--space-3) var(--space-4)',
                  boxShadow: 'var(--shadow-lg)',
                },
                
                // Individual toast type styles
                success: {
                  iconTheme: {
                    primary: 'var(--color-success-500)',
                    secondary: 'var(--color-success-50)',
                  },
                },
                error: {
                  iconTheme: {
                    primary: 'var(--color-error-500)',
                    secondary: 'var(--color-error-50)',
                  },
                },
                loading: {
                  iconTheme: {
                    primary: 'var(--color-primary-500)',
                    secondary: 'var(--color-primary-50)',
                  },
                },
              }}
            />
          </div>
        </Router>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

// 404 Not Found Component
function NotFound() {
  return (
    <div className="not-found">
      <div className="not-found__content">
        <h1 className="not-found__title">404</h1>
        <p className="not-found__description">
          The page you're looking for doesn't exist.
        </p>
        <a href="/" className="btn btn--primary">
          Go Home
        </a>
      </div>
    </div>
  );
}

export default App;