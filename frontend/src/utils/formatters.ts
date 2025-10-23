/**
 * Utility functions for formatting data in the UI
 */

// Format currency values
export const formatCurrency = (value: number, decimals: number = 2): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

// Format percentage values
export const formatPercentage = (value: number, decimals: number = 2): string => {
  return `${(value * 100).toFixed(decimals)}%`;
};

// Format large numbers with K, M, B suffixes
export const formatLargeNumber = (value: number): string => {
  if (value >= 1e9) {
    return `${(value / 1e9).toFixed(1)}B`;
  }
  if (value >= 1e6) {
    return `${(value / 1e6).toFixed(1)}M`;
  }
  if (value >= 1e3) {
    return `${(value / 1e3).toFixed(1)}K`;
  }
  return value.toString();
};

// Format timestamps
export const formatTimestamp = (timestamp: number, includeDate: boolean = false): string => {
  // Handle both seconds and milliseconds timestamps
  const date = new Date(timestamp > 1e12 ? timestamp : timestamp * 1000);
  
  if (includeDate) {
    return date.toLocaleString();
  }
  
  return date.toLocaleTimeString();
};

// Format relative time (e.g., "2 minutes ago")
export const formatRelativeTime = (timestamp: number): string => {
  const now = Date.now();
  const diff = now - (timestamp * 1000);
  
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (days > 0) {
    return `${days} day${days > 1 ? 's' : ''} ago`;
  }
  if (hours > 0) {
    return `${hours} hour${hours > 1 ? 's' : ''} ago`;
  }
  if (minutes > 0) {
    return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
  }
  return `${seconds} second${seconds > 1 ? 's' : ''} ago`;
};

// Format sentiment score with color class
export const formatSentimentScore = (score: number): { 
  formatted: string; 
  colorClass: string; 
} => {
  const formatted = `${score > 0 ? '+' : ''}${score.toFixed(2)}`;
  let colorClass = 'text-gray-600';
  
  if (score > 0.1) {
    colorClass = 'text-green-600';
  } else if (score < -0.1) {
    colorClass = 'text-red-600';
  }
  
  return { formatted, colorClass };
};

// Format confidence score as percentage
export const formatConfidence = (confidence: number): string => {
  return `${(confidence * 100).toFixed(0)}%`;
};

// Get status color class based on P&L
export const getPnLColorClass = (pnl: number): string => {
  if (pnl > 0) return 'text-green-600';
  if (pnl < 0) return 'text-red-600';
  return 'text-gray-600';
};

// Truncate text with ellipsis
export const truncateText = (text: string, maxLength: number): string => {
  if (text.length <= maxLength) return text;
  return `${text.substring(0, maxLength)}...`;
};