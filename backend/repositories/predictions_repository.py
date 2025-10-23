"""
Repository for price prediction data operations.
Handles DynamoDB operations for ML model predictions.
"""

import logging
from typing import List, Optional, Type, Dict, Any
from datetime import datetime

from .base_repository import BaseRepository
from ..models.data_models import PricePrediction
from ..config import settings

logger = logging.getLogger(__name__)

class PredictionsRepository(BaseRepository):
    """Repository for price prediction data."""
    
    def __init__(self):
        super().__init__(settings.predictions_table)
    
    def get_model_class(self) -> Type[PricePrediction]:
        """Return the PricePrediction model class."""
        return PricePrediction
    
    async def get_latest_predictions(
        self, 
        symbol: str, 
        limit: int = 10
    ) -> List[PricePrediction]:
        """
        Get the latest predictions for a symbol.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of records to return
            
        Returns:
            List of latest predictions
        """
        key_condition = "symbol = :symbol"
        expression_values = {':symbol': symbol}
        
        return await self.query_items(
            key_condition=key_condition,
            expression_values=expression_values,
            limit=limit,
            scan_forward=False  # Most recent first
        )
    
    async def get_predictions_by_horizon(
        self, 
        symbol: str, 
        horizon: str,
        limit: int = 5
    ) -> List[PricePrediction]:
        """
        Get predictions for a specific symbol and horizon.
        
        Args:
            symbol: Stock symbol
            horizon: Prediction horizon (1d, 3d, 7d)
            limit: Maximum number of records to return
            
        Returns:
            List of predictions for the specified horizon
        """
        filter_expression = "symbol = :symbol AND horizon = :horizon"
        expression_values = {
            ':symbol': symbol,
            ':horizon': horizon
        }
        
        return await self.scan_items(
            filter_expression=filter_expression,
            expression_values=expression_values,
            limit=limit
        )
    
    async def get_predictions_by_model(
        self, 
        symbol: str, 
        model: str,
        limit: int = 5
    ) -> List[PricePrediction]:
        """
        Get predictions for a specific symbol and model.
        
        Args:
            symbol: Stock symbol
            model: Model type (lstm, xgboost)
            limit: Maximum number of records to return
            
        Returns:
            List of predictions from the specified model
        """
        filter_expression = "symbol = :symbol AND model = :model"
        expression_values = {
            ':symbol': symbol,
            ':model': model
        }
        
        return await self.scan_items(
            filter_expression=filter_expression,
            expression_values=expression_values,
            limit=limit
        )
    
    async def get_consensus_prediction(self, symbol: str, horizon: str) -> Optional[Dict[str, Any]]:
        """
        Calculate consensus prediction from multiple models.
        
        Args:
            symbol: Stock symbol
            horizon: Prediction horizon
            
        Returns:
            Dictionary with consensus prediction data
        """
        predictions = await self.get_predictions_by_horizon(symbol, horizon, limit=10)
        
        if not predictions:
            return None
        
        # Group by model and get latest prediction from each
        model_predictions = {}
        for pred in predictions:
            if pred.model not in model_predictions:
                model_predictions[pred.model] = pred
            elif pred.timestamp > model_predictions[pred.model].timestamp:
                model_predictions[pred.model] = pred
        
        if not model_predictions:
            return None
        
        # Calculate consensus
        prices = [pred.predicted_price for pred in model_predictions.values()]
        lower_bounds = [pred.confidence_lower for pred in model_predictions.values()]
        upper_bounds = [pred.confidence_upper for pred in model_predictions.values()]
        
        consensus_price = sum(prices) / len(prices)
        consensus_lower = sum(lower_bounds) / len(lower_bounds)
        consensus_upper = sum(upper_bounds) / len(upper_bounds)
        
        return {
            'symbol': symbol,
            'horizon': horizon,
            'consensus_price': consensus_price,
            'confidence_lower': consensus_lower,
            'confidence_upper': consensus_upper,
            'model_count': len(model_predictions),
            'individual_predictions': list(model_predictions.values()),
            'timestamp': int(datetime.now().timestamp())
        }
    
    async def get_recent_predictions(
        self, 
        hours_back: int = 24, 
        limit: int = 1000
    ) -> List[PricePrediction]:
        """
        Get recent predictions across all symbols.
        
        Args:
            hours_back: Number of hours to look back
            limit: Maximum number of records to return
            
        Returns:
            List of recent predictions
        """
        cutoff_timestamp = int(datetime.now().timestamp()) - (hours_back * 3600)
        
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
            return [PricePrediction(**item) for item in response.get('Items', [])]
        except Exception as e:
            logger.error(f"Error scanning recent predictions: {e}")
            return []