import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  MessageSquare, 
  FileText, 
  BarChart3, 
  Settings, 
  HelpCircle,
  ChevronRight
} from 'lucide-react';
import './Sidebar.css';
const Sidebar = () => {
  const location = useLocation();

  const primaryNav = [
    { name: 'Chat', href: '/chat', icon: MessageSquare },
    { name: 'Documents', href: '/documents', icon: FileText },
    { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  ];

  const secondaryNav = [
    { name: 'Settings', href: '/settings', icon: Settings },
    { name: 'Help', href: '/help', icon: HelpCircle },
  ];

  const isActive = (href) => {
    if (href === '/chat') {
      return location.pathname === '/' || location.pathname === '/chat';
    }
    return location.pathname === href;
  };

  return (
    <aside className="sidebar">
      <div className="sidebar__content">
        {/* Primary Navigation */}
        <nav className="sidebar__nav">
          <div className="sidebar__nav-section">
            <h3 className="sidebar__nav-title">Main</h3>
            <ul className="sidebar__nav-list">
              {primaryNav.map((item) => {
                const Icon = item.icon;
                return (
                  <li key={item.name}>
                    <Link
                      to={item.href}
                      className={`sidebar__nav-item ${isActive(item.href) ? 'sidebar__nav-item--active' : ''}`}
                    >
                      <Icon size={20} />
                      <span>{item.name}</span>
                      <ChevronRight size={16} className="sidebar__nav-arrow" />
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>

          {/* Secondary Navigation */}
          <div className="sidebar__nav-section">
            <h3 className="sidebar__nav-title">More</h3>
            <ul className="sidebar__nav-list">
              {secondaryNav.map((item) => {
                const Icon = item.icon;
                return (
                  <li key={item.name}>
                    <Link
                      to={item.href}
                      className={`sidebar__nav-item ${isActive(item.href) ? 'sidebar__nav-item--active' : ''}`}
                    >
                      <Icon size={20} />
                      <span>{item.name}</span>
                      <ChevronRight size={16} className="sidebar__nav-arrow" />
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        </nav>
      </div>

      {/* Footer */}
      <div className="sidebar__footer">
        <div className="sidebar__version">
          <span className="sidebar__version-text">
            Version 1.0.0
          </span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;