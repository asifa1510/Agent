import React, { useState } from 'react';

interface SupportingDataVisualizationProps {
  data: Record<string, any>;
  compact?: boolean;
  className?: string;
}

export function SupportingDataVisualization({ 
  data, 
  compact = false, 
  className = '' 
}: SupportingDataVisualizationProps) {
  const [expandedSection, setExpandedSection] = useState<string | null>(null);

  // Categorize data by type for better visualization
  const categorizeData = (data: Record<string, any>) => {
    const categories: Record<string, Record<string, any>> = {
      sentiment: {},
      prediction: {},
      market: {},
      risk: {},
      other: {}
    };

    Object.entries(data).forEach(([key, value]) => {
      const lowerKey = key.toLowerCase();
      if (lowerKey.includes('sentiment') || lowerKey.includes('social')) {
        categories.sentiment[key] = value;
      } else if (lowerKey.includes('prediction') || lowerKey.includes('forecast') || lowerKey.includes('price')) {
        categories.prediction[key] = value;
      } else if (lowerKey.includes('market') || lowerKey.includes('volume') || lowerKey.includes('volatility')) {
        categories.market[key] = value;
      } else if (lowerKey.includes('risk') || lowerKey.includes('stop') || lowerKey.includes('position')) {
        categories.risk[key] = value;
      } else {
        categories.other[key] = value;
      }
    });

    // Filter out empty categories
    return Object.fromEntries(
      Object.entries(categories).filter(([_, values]) => Object.keys(values).length > 0)
    );
  };

  const formatValue = (value: any): string => {
    if (typeof value === 'number') {
      if (value > 1000 || value < -1000) {
        return value.toLocaleString();
      }
      return value.toFixed(3);
    }
    if (typeof value === 'boolean') {
      return value ? 'Yes' : 'No';
    }
    if (Array.isArray(value)) {
      return `[${value.length} items]`;
    }
    if (typeof value === 'object' && value !== null) {
      return `{${Object.keys(value).length} fields}`;
    }
    return String(value);
  };

  const formatKey = (key: string): string => {
    return key
      .replace(/_/g, ' ')
      .replace(/([A-Z])/g, ' $1')
      .replace(/^./, str => str.toUpperCase())
      .trim();
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'sentiment':
        return '💭';
      case 'prediction':
        return '📈';
      case 'market':
        return '📊';
      case 'risk':
        return '⚠️';
      default:
        return '📋';
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'sentiment':
        return 'bg-blue-50 border-blue-200 text-blue-800';
      case 'prediction':
        return 'bg-green-50 border-green-200 text-green-800';
      case 'market':
        return 'bg-purple-50 border-purple-200 text-purple-800';
      case 'risk':
        return 'bg-red-50 border-red-200 text-red-800';
      default:
        return 'bg-gray-50 border-gray-200 text-gray-800';
    }
  };

  const categorizedData = categorizeData(data);

  if (Object.keys(categorizedData).length === 0) {
    return (
      <div className={`p-4 text-center text-gray-500 text-sm ${className}`}>
        No supporting data available
      </div>
    );
  }

  return (
    <div className={`p-4 ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <h4 className={`font-medium text-gray-700 ${compact ? 'text-sm' : 'text-base'}`}>
          Supporting Data
        </h4>
        <span className="text-xs text-gray-500">
          {Object.keys(data).length} data points
        </span>
      </div>

      <div className="space-y-3">
        {Object.entries(categorizedData).map(([category, categoryData]) => (
          <div key={category} className="border border-gray-200 rounded-lg overflow-hidden">
            {/* Category Header */}
            <button
              onClick={() => setExpandedSection(
                expandedSection === category ? null : category
              )}
              className={`w-full px-3 py-2 text-left flex items-center justify-between hover:bg-gray-50 transition-colors ${
                compact ? 'text-sm' : 'text-base'
              }`}
            >
              <div className="flex items-center space-x-2">
                <span className="text-lg">{getCategoryIcon(category)}</span>
                <span className="font-medium text-gray-700 capitalize">
                  {category}
                </span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getCategoryColor(category)}`}>
                  {Object.keys(categoryData).length}
                </span>
              </div>
              <svg
                className={`w-4 h-4 text-gray-400 transition-transform ${
                  expandedSection === category ? 'rotate-180' : ''
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>

            {/* Category Content */}
            {expandedSection === category && (
              <div className="border-t border-gray-200 bg-gray-50">
                <div className="p-3 space-y-2">
                  {Object.entries(categoryData).map(([key, value]) => (
                    <div key={key} className="flex justify-between items-center">
                      <span className={`text-gray-600 ${compact ? 'text-xs' : 'text-sm'}`}>
                        {formatKey(key)}:
                      </span>
                      <span className={`font-medium text-gray-900 ${compact ? 'text-xs' : 'text-sm'}`}>
                        {formatValue(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Quick Summary for Compact Mode */}
      {compact && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <div className="grid grid-cols-2 gap-2 text-xs">
            {Object.entries(data).slice(0, 4).map(([key, value]) => (
              <div key={key} className="flex justify-between">
                <span className="text-gray-500 truncate">{formatKey(key)}:</span>
                <span className="text-gray-700 font-medium ml-1">
                  {formatValue(value)}
                </span>
              </div>
            ))}
          </div>
          {Object.keys(data).length > 4 && (
            <div className="text-center mt-2">
              <span className="text-xs text-gray-400">
                +{Object.keys(data).length - 4} more data points
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default SupportingDataVisualization;