"""
FastAPI router for sentiment analysis endpoints.
Provides API endpoints for retrieving and aggregating sentiment data.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime, timedelta

from ..models.data_models import SentimentScore
from ..services.sentiment_service import SentimentAggregationService
from ..repositories.sentiment_repository import SentimentRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sentiment", tags=["sentiment"])

# Initialize services
sentiment_service = SentimentAggregationService()
sentiment_repo = SentimentRepository()

# Response models
class SentimentResponse(BaseModel):
    """Response model for sentiment data."""
    data: List[SentimentScore]
    total_count: int
    status: str = "success"

class SentimentAggregateResponse(BaseModel):
    """Response model for aggregated sentiment data."""
    symbol: str
    avg_score: float
    avg_confidence: float
    total_volume: int
    time_window_hours: int
    timestamp: int
    status: str = "success"

class SentimentTrendResponse(BaseModel):
    """Response model for sentiment trend analysis."""
    symbol: str
    trend: str
    trend_strength: float
    volatility: float
    current_sentiment: float
    sentiment_change_24h: float
    momentum: float
    data_points: int
    hours_analyzed: int
    timestamp: int
    status: str = "success"

class SentimentSummaryResponse(BaseModel):
    """Response model for comprehensive sentiment summary."""
    symbol: str
    real_time: Dict[str, Any]
    hourly: Dict[str, Any]
    daily: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    timestamp: int
    status: str = "success"

@router.get("/", response_model=SentimentResponse)
async def get_sentiment_data(
    symbol: Optional[str] = Query(None, description="Stock symbol filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    hours_back: int = Query(24, ge=1, le=168, description="Hours to look back")
):
    """
    Retrieve sentiment data with optional filtering.
    
    Args:
        symbol: Optional stock symbol filter
        limit: Maximum number of records to return
        hours_back: Number of hours to look back from now
        
    Returns:
        List of sentiment scores matching the criteria
    """
    try:
        if symbol:
            # Get data for specific symbol
            data = await sentiment_repo.get_latest_by_symbol(symbol, limit)
        else:
            # Get recent data across all symbols
            data = await sentiment_repo.get_recent_sentiment(hours_back, limit)
        
        return SentimentResponse(
            data=data,
            total_count=len(data),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving sentiment data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve sentiment data: {str(e)}"
        )

@router.get("/{symbol}/aggregate", response_model=SentimentAggregateResponse)
async def get_sentiment_aggregate(
    symbol: str,
    hours: int = Query(1, ge=1, le=168, description="Time window in hours")
):
    """
    Get aggregated sentiment data for a specific symbol.
    
    Args:
        symbol: Stock symbol to aggregate
        hours: Time window for aggregation in hours
        
    Returns:
        Aggregated sentiment metrics for the symbol
    """
    try:
        result = await sentiment_repo.calculate_sentiment_aggregate(symbol.upper(), hours)
        
        return SentimentAggregateResponse(
            symbol=symbol.upper(),
            avg_score=result['avg_score'],
            avg_confidence=result['avg_confidence'],
            total_volume=result['total_volume'],
            time_window_hours=hours,
            timestamp=int(datetime.now().timestamp()),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error aggregating sentiment for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to aggregate sentiment for {symbol}: {str(e)}"
        )
@ro
uter.get("/{symbol}/realtime")
async def get_realtime_sentiment(
    symbol: str,
    minutes: int = Query(5, ge=1, le=60, description="Minutes to look back for real-time data")
):
    """
    Get real-time sentiment aggregation for a symbol.
    Updates every 5 minutes as per requirement 1.3.
    
    Args:
        symbol: Stock symbol
        minutes: Minutes to look back for aggregation
        
    Returns:
        Real-time sentiment metrics
    """
    try:
        result = await sentiment_service.get_real_time_sentiment(symbol.upper(), minutes)
        return {**result, "status": "success"}
        
    except Exception as e:
        logger.error(f"Error getting real-time sentiment for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get real-time sentiment for {symbol}: {str(e)}"
        )

@router.get("/{symbol}/windows")
async def get_time_window_sentiment(
    symbol: str,
    hours_back: int = Query(24, ge=1, le=168, description="Total hours to analyze"),
    window_size: int = Query(60, ge=15, le=240, description="Window size in minutes")
):
    """
    Get sentiment data aggregated over time windows.
    
    Args:
        symbol: Stock symbol
        hours_back: Total hours to look back
        window_size: Size of each time window in minutes
        
    Returns:
        List of sentiment aggregations for each time window
    """
    try:
        result = await sentiment_service.get_time_window_sentiment(
            symbol.upper(), hours_back, window_size
        )
        return {
            "symbol": symbol.upper(),
            "windows": result,
            "total_windows": len(result),
            "hours_analyzed": hours_back,
            "window_size_minutes": window_size,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting time window sentiment for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get time window sentiment for {symbol}: {str(e)}"
        )

@router.get("/{symbol}/trend", response_model=SentimentTrendResponse)
async def get_sentiment_trend(
    symbol: str,
    hours: int = Query(24, ge=6, le=168, description="Hours to analyze for trend")
):
    """
    Analyze sentiment trends for a symbol.
    
    Args:
        symbol: Stock symbol
        hours: Hours to analyze for trend
        
    Returns:
        Sentiment trend analysis results
    """
    try:
        result = await sentiment_service.analyze_sentiment_trend(symbol.upper(), hours)
        
        return SentimentTrendResponse(
            symbol=result['symbol'],
            trend=result['trend'],
            trend_strength=result['trend_strength'],
            volatility=result['volatility'],
            current_sentiment=result['current_sentiment'],
            sentiment_change_24h=result['sentiment_change_24h'],
            momentum=result.get('momentum', 0.0),
            data_points=result.get('data_points', 0),
            hours_analyzed=result['hours_analyzed'],
            timestamp=result['timestamp'],
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment trend for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze sentiment trend for {symbol}: {str(e)}"
        )

@router.get("/{symbol}/summary", response_model=SentimentSummaryResponse)
async def get_sentiment_summary(symbol: str):
    """
    Get comprehensive sentiment summary for a symbol.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Comprehensive sentiment analysis including real-time, hourly, daily, and trend data
    """
    try:
        result = await sentiment_service.get_sentiment_summary(symbol.upper())
        
        return SentimentSummaryResponse(
            symbol=result['symbol'],
            real_time=result['real_time'],
            hourly=result['hourly'],
            daily=result['daily'],
            trend_analysis=result['trend_analysis'],
            timestamp=result['timestamp'],
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error getting sentiment summary for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get sentiment summary for {symbol}: {str(e)}"
        )

@router.post("/multi-symbol")
async def get_multi_symbol_sentiment(
    symbols: List[str],
    hours_back: int = Query(1, ge=1, le=24, description="Hours to look back")
):
    """
    Get sentiment aggregation for multiple symbols.
    
    Args:
        symbols: List of stock symbols (max 20)
        hours_back: Hours to look back for aggregation
        
    Returns:
        Dictionary mapping symbols to their sentiment metrics
    """
    try:
        if len(symbols) > 20:
            raise HTTPException(
                status_code=400,
                detail="Maximum 20 symbols allowed per request"
            )
        
        # Convert to uppercase
        symbols = [symbol.upper() for symbol in symbols]
        
        result = await sentiment_service.get_multi_symbol_sentiment(symbols, hours_back)
        
        return {
            "symbols": result,
            "total_symbols": len(symbols),
            "hours_analyzed": hours_back,
            "timestamp": int(datetime.now().timestamp()),
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting multi-symbol sentiment: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get multi-symbol sentiment: {str(e)}"
        )