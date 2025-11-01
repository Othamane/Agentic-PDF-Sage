import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { MessageSquare, FileText, BarChart3, Bot } from 'lucide-react';
import './Header.css'
const Header = () => {
  const location = useLocation();

  const navigation = [
    { name: 'Chat', href: '/chat', icon: MessageSquare },
    { name: 'Documents', href: '/documents', icon: FileText },
    { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  ];

  const isActive = (href) => {
    if (href === '/chat') {
      return location.pathname === '/' || location.pathname === '/chat';
    }
    return location.pathname === href;
  };

  return (
    <header className="header">
      <div className="header__container">
        {/* Logo */}
        <Link to="/" className="header__logo">
          <div className="header__logo-icon">
            <Bot size={24} />
          </div>
          <div className="header__logo-text">
            <span className="header__logo-title">PDF Sage</span>
            <span className="header__logo-subtitle">Intelligent Document Assistant</span>
          </div>
        </Link>

        {/* Navigation */}
        <nav className="header__nav">
          {navigation.map((item) => {
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`header__nav-item ${isActive(item.href) ? 'header__nav-item--active' : ''}`}
              >
                <Icon size={20} />
                <span>{item.name}</span>
              </Link>
            );
          })}
        </nav>

        {/* Status Indicator */}
        <div className="header__status">
          <div className="header__status-indicator header__status-indicator--online">
            <div className="header__status-dot"></div>
            <span className="header__status-text">Online</span>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;