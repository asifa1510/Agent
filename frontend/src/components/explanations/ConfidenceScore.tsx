import React from 'react';

interface ConfidenceScoreProps {
  confidence: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  showPercentage?: boolean;
  className?: string;
}

export function ConfidenceScore({ 
  confidence, 
  size = 'md', 
  showLabel = true,
  showPercentage = true,
  className = '' 
}: ConfidenceScoreProps) {
  const percentage = Math.round(confidence * 100);
  


  const getRingColorClasses = () => {
    if (confidence >= 0.8) return 'text-green-500';
    if (confidence >= 0.6) return 'text-yellow-500';
    if (confidence >= 0.4) return 'text-orange-500';
    return 'text-red-500';
  };

  const getSizeClasses = () => {
    switch (size) {
      case 'sm':
        return {
          container: 'w-12 h-12',
          text: 'text-xs',
          strokeWidth: '3'
        };
      case 'lg':
        return {
          container: 'w-20 h-20',
          text: 'text-sm',
          strokeWidth: '2'
        };
      default:
        return {
          container: 'w-16 h-16',
          text: 'text-xs',
          strokeWidth: '2.5'
        };
    }
  };

  const sizeClasses = getSizeClasses();
  const circumference = 2 * Math.PI * 18; // radius = 18
  const strokeDasharray = circumference;
  const strokeDashoffset = circumference - (confidence * circumference);

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      {/* Circular Progress */}
      <div className={`relative ${sizeClasses.container}`}>
        <svg className="transform -rotate-90 w-full h-full" viewBox="0 0 40 40">
          {/* Background circle */}
          <circle
            cx="20"
            cy="20"
            r="18"
            stroke="currentColor"
            strokeWidth={sizeClasses.strokeWidth}
            fill="none"
            className="text-gray-200"
          />
          {/* Progress circle */}
          <circle
            cx="20"
            cy="20"
            r="18"
            stroke="currentColor"
            strokeWidth={sizeClasses.strokeWidth}
            fill="none"
            strokeLinecap="round"
            strokeDasharray={strokeDasharray}
            strokeDashoffset={strokeDashoffset}
            className={getRingColorClasses()}
            style={{
              transition: 'stroke-dashoffset 0.5s ease-in-out'
            }}
          />
        </svg>
        
        {/* Percentage text */}
        {showPercentage && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className={`font-semibold ${sizeClasses.text} text-gray-700`}>
              {percentage}%
            </span>
          </div>
        )}
      </div>

      {/* Label */}
      {showLabel && (
        <div className="flex flex-col">
          <span className="text-xs font-medium text-gray-700">Confidence</span>
          <span className={`text-xs ${getRingColorClasses().replace('text-', 'text-')}`}>
            {confidence >= 0.8 ? 'High' : 
             confidence >= 0.6 ? 'Medium' : 
             confidence >= 0.4 ? 'Low' : 'Very Low'}
          </span>
        </div>
      )}
    </div>
  );
}

export default ConfidenceScore;