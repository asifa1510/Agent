from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime

class SentimentScore(BaseModel):
    """Sentiment analysis result for a stock symbol"""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    timestamp: int = Field(..., description="Unix timestamp")
    score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score from -1 to 1")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0 to 1")
    volume: int = Field(..., ge=0, description="Number of posts analyzed")
    source: Literal['twitter', 'reddit'] = Field(..., description="Data source")

class PricePrediction(BaseModel):
    """Price prediction for a stock symbol"""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    timestamp: int = Field(..., description="Unix timestamp")
    horizon: Literal['1d', '3d', '7d'] = Field(..., description="Prediction time horizon")
    predicted_price: float = Field(..., gt=0, description="Predicted stock price")
    confidence_lower: float = Field(..., gt=0, description="Lower confidence bound")
    confidence_upper: float = Field(..., gt=0, description="Upper confidence bound")
    model: Literal['lstm', 'xgboost'] = Field(..., description="ML model used")

class Trade(BaseModel):
    """Trading transaction record"""
    id: str = Field(..., description="Unique trade identifier")
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    timestamp: int = Field(..., description="Unix timestamp")
    action: Literal['buy', 'sell'] = Field(..., description="Trade action")
    quantity: int = Field(..., gt=0, description="Number of shares")
    price: float = Field(..., gt=0, description="Execution price per share")
    signal_strength: float = Field(..., ge=0.0, le=1.0, description="Trading signal strength")
    explanation_id: Optional[str] = Field(None, description="Reference to explanation record")

class TradeExplanation(BaseModel):
    """AI-generated explanation for a trade decision"""
    id: str = Field(..., description="Unique explanation identifier")
    trade_id: str = Field(..., description="Associated trade identifier")
    explanation: str = Field(..., description="Human-readable trade explanation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Explanation confidence")
    supporting_data: dict = Field(default_factory=dict, description="Supporting data and metrics")
    timestamp: int = Field(..., description="Unix timestamp")

class PortfolioPosition(BaseModel):
    """Current portfolio position for a symbol"""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    quantity: int = Field(..., description="Current shares held (can be negative for short)")
    avg_price: float = Field(..., gt=0, description="Average purchase price")
    current_price: float = Field(..., gt=0, description="Current market price")
    unrealized_pnl: float = Field(..., description="Unrealized profit/loss")
    allocation_percent: float = Field(..., ge=0.0, le=100.0, description="Portfolio allocation percentage")
    last_updated: int = Field(..., description="Unix timestamp of last update")

class RiskMetrics(BaseModel):
    """Portfolio risk metrics"""
    total_value: float = Field(..., description="Total portfolio value")
    total_pnl: float = Field(..., description="Total profit/loss")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    volatility: float = Field(..., ge=0.0, description="Portfolio volatility")
    var_95: float = Field(..., description="Value at Risk (95% confidence)")
    positions_count: int = Field(..., ge=0, description="Number of active positions")
    timestamp: int = Field(..., description="Unix timestamp")