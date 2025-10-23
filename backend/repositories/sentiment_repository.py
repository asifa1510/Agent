"""
Repository for sentiment score data operations.
Handles DynamoDB operations for sentiment analysis results.
"""

import logging
from typing import List, Optional, Type
from datetime import datetime, timedelta

from .base_repository import BaseRepository
from ..models.data_models import SentimentScore
from ..config import settings

logger = logging.getLogger(__name__)

class SentimentRepository(BaseRepository):
    """Repository for sentiment score data."""
    
    def __init__(self):
        super().__init__(settings.sentiment_table)
    
    def get_model_class(self) -> Type[SentimentScore]:
        """Return the SentimentScore model class."""
        return SentimentScore
    
    async def get_by_symbol_and_timerange(
        self, 
        symbol: str, 
        start_timestamp: int, 
        end_timestamp: int,
        limit: Optional[int] = None
    ) -> List[SentimentScore]:
        """
        Get sentiment scores for a symbol within a time range.
        
        Args:
            symbol: Stock symbol
            start_timestamp: Start time (Unix timestamp)
            end_timestamp: End time (Unix timestamp)
            limit: Maximum number of records to return
            
        Returns:
            List of sentiment scores
        """
        key_condition = "symbol = :symbol AND #ts BETWEEN :start_ts AND :end_ts"
        expression_values = {
            ':symbol': symbol,
            ':start_ts': start_timestamp,
            ':end_ts': end_timestamp
        }
        
        # Use expression attribute names for reserved keywords
        expression_names = {'#ts': 'timestamp'}
        
        try:
            query_params = {
                'KeyConditionExpression': key_condition,
                'ExpressionAttributeValues': expression_values,
                'ExpressionAttributeNames': expression_names,
                'ScanIndexForward': False  # Most recent first
            }
            
            if limit:
                query_params['Limit'] = limit
            
            response = self.table.query(**query_params)
            return [SentimentScore(**item) for item in response.get('Items', [])]
        except Exception as e:
            logger.error(f"Error querying sentiment by symbol and time: {e}")
            return []
    
    async def get_latest_by_symbol(self, symbol: str, limit: int = 10) -> List[SentimentScore]:
        """
        Get the latest sentiment scores for a symbol.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of records to return
            
        Returns:
            List of latest sentiment scores
        """
        key_condition = "symbol = :symbol"
        expression_values = {':symbol': symbol}
        
        return await self.query_items(
            key_condition=key_condition,
            expression_values=expression_values,
            limit=limit,
            scan_forward=False  # Most recent first
        )
    
    async def get_recent_sentiment(self, hours_back: int = 24, limit: int = 1000) -> List[SentimentScore]:
        """
        Get recent sentiment scores across all symbols.
        
        Args:
            hours_back: Number of hours to look back
            limit: Maximum number of records to return
            
        Returns:
            List of recent sentiment scores
        """
        cutoff_timestamp = int((datetime.now() - timedelta(hours=hours_back)).timestamp())
        
        filter_expression = "#ts >= :cutoff_ts"
        expression_values = {':cutoff_ts': cutoff_timestamp}
        expression_names = {'#ts': 'timestamp'}
        
        try:
            scan_params = {
                'FilterExpression': filter_expression,
                'ExpressionAttributeValues': expression_values,
                'ExpressionAttributeNames': expression_names,
                'Limit': limit
            }
            
            response = self.table.scan(**scan_params)
            return [SentimentScore(**item) for item in response.get('Items', [])]
        except Exception as e:
            logger.error(f"Error scanning recent sentiment: {e}")
            return []
    
    async def calculate_sentiment_aggregate(
        self, 
        symbol: str, 
        hours_back: int = 1
    ) -> dict:
        """
        Calculate aggregated sentiment metrics for a symbol.
        
        Args:
            symbol: Stock symbol
            hours_back: Number of hours to aggregate over
            
        Returns:
            Dictionary with aggregated metrics
        """
        end_timestamp = int(datetime.now().timestamp())
        start_timestamp = end_timestamp - (hours_back * 3600)
        
        scores = await self.get_by_symbol_and_timerange(
            symbol, start_timestamp, end_timestamp
        )
        
        if not scores:
            return {
                'avg_score': 0.0,
                'avg_confidence': 0.0,
                'total_volume': 0,
                'count': 0
            }
        
        total_score = sum(score.score for score in scores)
        total_confidence = sum(score.confidence for score in scores)
        total_volume = sum(score.volume for score in scores)
        count = len(scores)
        
        return {
            'avg_score': total_score / count,
            'avg_confidence': total_confidence / count,
            'total_volume': total_volume,
            'count': count
        }