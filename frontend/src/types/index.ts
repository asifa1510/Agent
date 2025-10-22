// TypeScript interfaces for frontend data models

export interface SentimentScore {
  symbol: string;
  timestamp: number;
  score: number; // -1 to 1
  confidence: number; // 0 to 1
  volume: number; // post count
  source: 'twitter' | 'reddit';
}

export interface PricePrediction {
  symbol: string;
  timestamp: number;
  horizon: '1d' | '3d' | '7d';
  predicted_price: number;
  confidence_lower: number;
  confidence_upper: number;
  model: 'lstm' | 'xgboost';
}

export interface Trade {
  id: string;
  symbol: string;
  timestamp: number;
  action: 'buy' | 'sell';
  quantity: number;
  price: number;
  signal_strength: number;
  explanation_id?: string;
}

export interface TradeExplanation {
  id: string;
  trade_id: string;
  explanation: string;
  confidence: number;
  supporting_data: Record<string, any>;
  timestamp: number;
}

export interface PortfolioPosition {
  symbol: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  unrealized_pnl: number;
  allocation_percent: number;
  last_updated: number;
}

export interface RiskMetrics {
  total_value: number;
  total_pnl: number;
  max_drawdown: number;
  sharpe_ratio?: number;
  volatility: number;
  var_95: number;
  positions_count: number;
  timestamp: number;
}

// API Response types
export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
}

// Chart data types
export interface ChartDataPoint {
  x: number; // timestamp
  y: number; // value
}

export interface SentimentChartData {
  symbol: string;
  data: ChartDataPoint[];
  source: 'twitter' | 'reddit';
}

export interface PredictionChartData {
  symbol: string;
  predictions: ChartDataPoint[];
  confidence_upper: ChartDataPoint[];
  confidence_lower: ChartDataPoint[];
  model: 'lstm' | 'xgboost';
}