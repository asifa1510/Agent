/**
 * Application constants
 */

// Popular stock symbols for quick selection
export const POPULAR_SYMBOLS = [
  'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
  'META', 'NVDA', 'NFLX', 'SPY', 'QQQ'
];

// Chart colors for consistent theming
export const CHART_COLORS = {
  primary: '#3B82F6',
  success: '#10B981',
  danger: '#EF4444',
  warning: '#F59E0B',
  secondary: '#6B7280',
  background: '#F9FAFB',
  grid: '#E5E7EB',
};

// Sentiment score thresholds
export const SENTIMENT_THRESHOLDS = {
  VERY_POSITIVE: 0.5,
  POSITIVE: 0.1,
  NEUTRAL: 0.0,
  NEGATIVE: -0.1,
  VERY_NEGATIVE: -0.5,
};

// Confidence score thresholds
export const CONFIDENCE_THRESHOLDS = {
  HIGH: 0.8,
  MEDIUM: 0.6,
  LOW: 0.4,
};

// Trade action types
export const TRADE_ACTIONS = {
  BUY: 'buy',
  SELL: 'sell',
} as const;

// Prediction horizons
export const PREDICTION_HORIZONS = {
  ONE_DAY: '1d',
  THREE_DAY: '3d',
  SEVEN_DAY: '7d',
} as const;

// Model types
export const MODEL_TYPES = {
  LSTM: 'lstm',
  XGBOOST: 'xgboost',
  BERT: 'bert',
} as const;

// Data sources
export const DATA_SOURCES = {
  TWITTER: 'twitter',
  REDDIT: 'reddit',
} as const;

// Risk metrics thresholds
export const RISK_THRESHOLDS = {
  MAX_POSITION_SIZE: 0.05, // 5% max per position
  STOP_LOSS: 0.02, // 2% stop loss
  MAX_DRAWDOWN_WARNING: 0.1, // 10% drawdown warning
  MIN_SHARPE_RATIO: 1.0,
};

// UI constants
export const UI_CONSTANTS = {
  MAX_CHART_DATA_POINTS: 100,
  DEFAULT_PAGE_SIZE: 20,
  DEBOUNCE_DELAY: 300,
  TOAST_DURATION: 5000,
};

// API endpoints (relative to base URL)
export const API_ENDPOINTS = {
  SENTIMENT: '/sentiment',
  PREDICTIONS: '/predictions',
  TRADES: '/trades',
  EXPLANATIONS: '/explanations',
  PORTFOLIO: '/portfolio',
  SIMULATION: '/portfolio/simulate',
  BENCHMARK: '/portfolio/benchmark',
} as const;