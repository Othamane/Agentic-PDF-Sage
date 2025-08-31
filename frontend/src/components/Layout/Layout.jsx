import React from 'react';
import { useLocation } from 'react-router-dom';

import Header from './Header';
import Sidebar from './Sidebar';
import ErrorBoundary from '../ErrorBoundary/ErrorBoundary';
import './Layout.css'

const Layout = ({ children }) => {
  const location = useLocation();
  
  // Determine if we should show the sidebar based on the current route
  const showSidebar = location.pathname !== '/' && location.pathname !== '/chat';

  return (
    <div className="layout">
      <Header />
      
      <div className="layout__main">
        {showSidebar && <Sidebar />}
        
        <main className={`layout__content ${showSidebar ? 'layout__content--with-sidebar' : ''}`}>
          <ErrorBoundary>
            {children}
          </ErrorBoundary>
        </main>
      </div>
    </div>
  );
};

export default Layout;