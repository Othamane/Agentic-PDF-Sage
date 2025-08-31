import React from 'react';
import { Loader2 } from 'lucide-react';

const LoadingSpinner = ({ 
  size = 'md', 
  className = '', 
  text = '',
  centered = false 
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  const containerClasses = centered 
    ? 'flex flex-col items-center justify-center gap-2'
    : 'flex items-center gap-2';

  return (
    <div className={`loading-spinner ${containerClasses} ${className}`}>
      <Loader2 
        className={`animate-spin ${sizeClasses[size]} text-primary-500`}
      />
      {text && (
        <span className="loading-spinner__text text-sm text-gray-600">
          {text}
        </span>
      )}
    </div>
  );
};

export default LoadingSpinner;