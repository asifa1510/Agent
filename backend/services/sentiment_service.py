"""
Sentiment aggregation service.
Provides real-time sentiment score aggregation and trend analysis.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from statistics import mean, stdev
import asyncio

from ..repositories.sentiment_repository import SentimentRepository
from ..models.data_models import SentimentScore

logger = logging.getLogger(__name__)

class SentimentAggregationService:
    """Service for aggregating and analyzing sentiment data."""
    
    def __init__(self):
        self.sentiment_repo = SentimentRepository()
    
    async def get_real_time_sentiment(
        self, 
        symbol: str, 
        minutes_back: int = 5
    ) -> Dict[str, float]:
        """
        Get real-time sentiment aggregation for a symbol.
        Updates every 5 minutes as per requirement 1.3.
        
        Args:
            symbol: Stock symbol
            minutes_back: Minutes to look back for aggregation
            
        Returns:
            Dictionary with aggregated sentiment metrics
        """
        try:
            end_timestamp = int(datetime.now().timestamp())
            start_timestamp = end_timestamp - (minutes_back * 60)
            
            scores = await self.sentiment_repo.get_by_symbol_and_timerange(
                symbol, start_timestamp, end_timestamp
            )
            
            if not scores:
                return {
                    'symbol': symbol,
                    'avg_score': 0.0,
                    'avg_confidence': 0.0,
                    'total_volume': 0,
                    'count': 0,
                    'timestamp': end_timestamp,
                    'time_window_minutes': minutes_back
                }
            
            # Calculate aggregated metrics
            total_score = sum(score.score for score in scores)
            total_confidence = sum(score.confidence for score in scores)
            total_volume = sum(score.volume for score in scores)
            count = len(scores)
            
            return {
                'symbol': symbol,
                'avg_score': total_score / count,
                'avg_confidence': total_confidence / count,
                'total_volume': total_volume,
                'count': count,
                'timestamp': end_timestamp,
                'time_window_minutes': minutes_back
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time sentiment for {symbol}: {e}")
            return {
                'symbol': symbol,
                'avg_score': 0.0,
                'avg_confidence': 0.0,
                'total_volume': 0,
                'count': 0,
                'timestamp': int(datetime.now().timestamp()),
                'time_window_minutes': minutes_back,
                'error': str(e)
            }
    
    async def get_time_window_sentiment(
        self, 
        symbol: str, 
        hours_back: int = 24,
        window_size_minutes: int = 60
    ) -> List[Dict[str, float]]:
        """
        Get sentiment data aggregated over time windows.
        
        Args:
            symbol: Stock symbol
            hours_back: Total hours to look back
            window_size_minutes: Size of each time window in minutes
            
        Returns:
            List of sentiment aggregations for each time window
        """
        try:
            end_timestamp = int(datetime.now().timestamp())
            start_timestamp = end_timestamp - (hours_back * 3600)
            
            # Get all sentiment data for the time period
            all_scores = await self.sentiment_repo.get_by_symbol_and_timerange(
                symbol, start_timestamp, end_timestamp
            )
            
            if not all_scores:
                return []
            
            # Create time windows
            window_size_seconds = window_size_minutes * 60
            windows = []
            
            current_window_start = start_timestamp
            while current_window_start < end_timestamp:
                current_window_end = min(current_window_start + window_size_seconds, end_timestamp)
                
                # Filter scores for this window
                window_scores = [
                    score for score in all_scores
                    if current_window_start <= score.timestamp < current_window_end
                ]
                
                if window_scores:
                    total_score = sum(score.score for score in window_scores)
                    total_confidence = sum(score.confidence for score in window_scores)
                    total_volume = sum(score.volume for score in window_scores)
                    count = len(window_scores)
                    
                    windows.append({
                        'symbol': symbol,
                        'window_start': current_window_start,
                        'window_end': current_window_end,
                        'avg_score': total_score / count,
                        'avg_confidence': total_confidence / count,
                        'total_volume': total_volume,
                        'count': count,
                        'window_size_minutes': window_size_minutes
                    })
                else:
                    # Add empty window for continuity
                    windows.append({
                        'symbol': symbol,
                        'window_start': current_window_start,
                        'window_end': current_window_end,
                        'avg_score': 0.0,
                        'avg_confidence': 0.0,
                        'total_volume': 0,
                        'count': 0,
                        'window_size_minutes': window_size_minutes
                    })
                
                current_window_start = current_window_end
            
            return windows
            
        except Exception as e:
            logger.error(f"Error getting time window sentiment for {symbol}: {e}")
            return []    asy
nc def analyze_sentiment_trend(
        self, 
        symbol: str, 
        hours_back: int = 24
    ) -> Dict[str, any]:
        """
        Analyze sentiment trends for a symbol.
        
        Args:
            symbol: Stock symbol
            hours_back: Hours to analyze for trend
            
        Returns:
            Dictionary with trend analysis results
        """
        try:
            # Get hourly sentiment windows
            hourly_data = await self.get_time_window_sentiment(
                symbol, hours_back, window_size_minutes=60
            )
            
            if len(hourly_data) < 2:
                return {
                    'symbol': symbol,
                    'trend': 'insufficient_data',
                    'trend_strength': 0.0,
                    'volatility': 0.0,
                    'current_sentiment': 0.0,
                    'sentiment_change_24h': 0.0,
                    'hours_analyzed': hours_back
                }
            
            # Extract sentiment scores for trend analysis
            sentiment_scores = [window['avg_score'] for window in hourly_data if window['count'] > 0]
            
            if len(sentiment_scores) < 2:
                return {
                    'symbol': symbol,
                    'trend': 'insufficient_data',
                    'trend_strength': 0.0,
                    'volatility': 0.0,
                    'current_sentiment': 0.0,
                    'sentiment_change_24h': 0.0,
                    'hours_analyzed': hours_back
                }
            
            # Calculate trend metrics
            current_sentiment = sentiment_scores[-1]
            initial_sentiment = sentiment_scores[0]
            sentiment_change = current_sentiment - initial_sentiment
            
            # Calculate trend strength using linear regression slope
            x_values = list(range(len(sentiment_scores)))
            n = len(sentiment_scores)
            
            sum_x = sum(x_values)
            sum_y = sum(sentiment_scores)
            sum_xy = sum(x * y for x, y in zip(x_values, sentiment_scores))
            sum_x2 = sum(x * x for x in x_values)
            
            # Linear regression slope (trend strength)
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            else:
                slope = 0.0
            
            # Determine trend direction
            if abs(slope) < 0.01:
                trend = 'neutral'
            elif slope > 0:
                trend = 'bullish'
            else:
                trend = 'bearish'
            
            # Calculate volatility (standard deviation)
            volatility = stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0.0
            
            # Calculate momentum (recent vs older sentiment)
            recent_avg = mean(sentiment_scores[-6:]) if len(sentiment_scores) >= 6 else current_sentiment
            older_avg = mean(sentiment_scores[:6]) if len(sentiment_scores) >= 6 else initial_sentiment
            momentum = recent_avg - older_avg
            
            return {
                'symbol': symbol,
                'trend': trend,
                'trend_strength': abs(slope),
                'slope': slope,
                'volatility': volatility,
                'current_sentiment': current_sentiment,
                'sentiment_change_24h': sentiment_change,
                'momentum': momentum,
                'data_points': len(sentiment_scores),
                'hours_analyzed': hours_back,
                'timestamp': int(datetime.now().timestamp())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment trend for {symbol}: {e}")
            return {
                'symbol': symbol,
                'trend': 'error',
                'trend_strength': 0.0,
                'volatility': 0.0,
                'current_sentiment': 0.0,
                'sentiment_change_24h': 0.0,
                'hours_analyzed': hours_back,
                'error': str(e)
            }
    
    async def get_multi_symbol_sentiment(
        self, 
        symbols: List[str], 
        hours_back: int = 1
    ) -> Dict[str, Dict[str, float]]:
        """
        Get sentiment aggregation for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            hours_back: Hours to look back for aggregation
            
        Returns:
            Dictionary mapping symbols to their sentiment metrics
        """
        try:
            tasks = [
                self.sentiment_repo.calculate_sentiment_aggregate(symbol, hours_back)
                for symbol in symbols
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            sentiment_data = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.error(f"Error getting sentiment for {symbol}: {result}")
                    sentiment_data[symbol] = {
                        'avg_score': 0.0,
                        'avg_confidence': 0.0,
                        'total_volume': 0,
                        'count': 0,
                        'error': str(result)
                    }
                else:
                    sentiment_data[symbol] = result
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error getting multi-symbol sentiment: {e}")
            return {symbol: {'error': str(e)} for symbol in symbols}
    
    async def get_sentiment_summary(
        self, 
        symbol: str
    ) -> Dict[str, any]:
        """
        Get comprehensive sentiment summary for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with comprehensive sentiment analysis
        """
        try:
            # Get real-time sentiment (5 minutes)
            real_time = await self.get_real_time_sentiment(symbol, minutes_back=5)
            
            # Get hourly aggregation
            hourly = await self.sentiment_repo.calculate_sentiment_aggregate(symbol, hours_back=1)
            
            # Get daily aggregation
            daily = await self.sentiment_repo.calculate_sentiment_aggregate(symbol, hours_back=24)
            
            # Get trend analysis
            trend = await self.analyze_sentiment_trend(symbol, hours_back=24)
            
            return {
                'symbol': symbol,
                'real_time': real_time,
                'hourly': hourly,
                'daily': daily,
                'trend_analysis': trend,
                'timestamp': int(datetime.now().timestamp())
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment summary for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': int(datetime.now().timestamp())
            }