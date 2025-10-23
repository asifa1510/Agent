"""
Prediction service for aggregating ML model predictions.
Handles prediction aggregation, confidence intervals, and caching.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import numpy as np
from dataclasses import dataclass, asdict
import boto3
from botocore.exceptions import ClientError

from ..models.data_models import PricePrediction
from ..repositories.predictions_repository import PredictionsRepository
from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Individual model prediction result."""
    model: str
    symbol: str
    horizon: str
    predicted_price: float
    confidence_lower: float
    confidence_upper: float
    timestamp: int
    confidence_score: float = 0.0


@dataclass
class AggregatedPrediction:
    """Aggregated prediction from multiple models."""
    symbol: str
    horizon: str
    consensus_price: float
    confidence_lower: float
    confidence_upper: float
    model_count: int
    individual_predictions: List[ModelPrediction]
    timestamp: int
    confidence_score: float
    prediction_variance: float


class PredictionService:
    """
    Service for aggregating ML model predictions and managing prediction cache.
    Implements prediction aggregation, confidence interval calculations, and caching.
    """
    
    def __init__(self):
        """Initialize prediction service."""
        self.predictions_repo = PredictionsRepository()
        self.sagemaker_client = boto3.client('sagemaker-runtime')
        self.cache = {}  # In-memory cache for recent predictions
        self.cache_ttl = 300  # 5 minutes cache TTL
        
        # SageMaker endpoint names
        self.lstm_endpoint = settings.lstm_endpoint_name
        self.xgboost_endpoint = settings.xgboost_endpoint_name
        
    async def get_model_prediction(
        self, 
        model: str, 
        symbol: str, 
        input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get prediction from a specific ML model.
        
        Args:
            model: Model type ('lstm' or 'xgboost')
            symbol: Stock symbol
            input_data: Input data for the model
            
        Returns:
            Model prediction result or None if failed
        """
        try:
            endpoint_name = self.lstm_endpoint if model == 'lstm' else self.xgboost_endpoint
            
            if not endpoint_name:
                logger.warning(f"No endpoint configured for model: {model}")
                return None
            
            # Prepare request payload
            payload = json.dumps(input_data)
            
            # Call SageMaker endpoint
            response = self.sagemaker_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=payload
            )
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            
            if 'error' in result:
                logger.error(f"Model {model} prediction error: {result['error']}")
                return None
            
            logger.info(f"Successfully got prediction from {model} model for {symbol}")
            return result
            
        except ClientError as e:
            logger.error(f"SageMaker client error for {model}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting {model} prediction: {e}")
            return None
    
    async def aggregate_predictions(
        self, 
        symbol: str, 
        horizon: str,
        input_data: Dict[str, Any]
    ) -> Optional[AggregatedPrediction]:
        """
        Aggregate predictions from multiple models.
        
        Args:
            symbol: Stock symbol
            horizon: Prediction horizon ('1d', '3d', '7d')
            input_data: Input data for models
            
        Returns:
            Aggregated prediction result
        """
        try:
            # Get predictions from both models concurrently
            lstm_task = self.get_model_prediction('lstm', symbol, input_data)
            xgboost_task = self.get_model_prediction('xgboost', symbol, input_data)
            
            lstm_result, xgboost_result = await asyncio.gather(
                lstm_task, xgboost_task, return_exceptions=True
            )
            
            individual_predictions = []
            current_timestamp = int(datetime.now().timestamp())
            
            # Process LSTM prediction
            if isinstance(lstm_result, dict) and 'predictions' in lstm_result:
                lstm_pred = lstm_result['predictions'].get(horizon.replace('d', ''))
                if lstm_pred:
                    individual_predictions.append(ModelPrediction(
                        model='lstm',
                        symbol=symbol,
                        horizon=horizon,
                        predicted_price=lstm_pred['predicted_price'],
                        confidence_lower=lstm_pred['confidence_lower'],
                        confidence_upper=lstm_pred['confidence_upper'],
                        timestamp=current_timestamp,
                        confidence_score=self._calculate_confidence_score(lstm_pred)
                    ))
            
            # Process XGBoost prediction
            if isinstance(xgboost_result, dict) and 'predictions' in xgboost_result:
                xgb_pred = xgboost_result['predictions'].get(horizon.replace('d', ''))
                if xgb_pred:
                    individual_predictions.append(ModelPrediction(
                        model='xgboost',
                        symbol=symbol,
                        horizon=horizon,
                        predicted_price=xgb_pred['predicted_price'],
                        confidence_lower=xgb_pred['confidence_lower'],
                        confidence_upper=xgb_pred['confidence_upper'],
                        timestamp=current_timestamp,
                        confidence_score=self._calculate_confidence_score(xgb_pred)
                    ))
            
            if not individual_predictions:
                logger.warning(f"No valid predictions obtained for {symbol} {horizon}")
                return None
            
            # Calculate aggregated prediction
            aggregated = self._calculate_consensus(individual_predictions, symbol, horizon)
            
            logger.info(f"Aggregated {len(individual_predictions)} predictions for {symbol} {horizon}")
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating predictions for {symbol}: {e}")
            return None
    
    def _calculate_confidence_score(self, prediction: Dict[str, float]) -> float:
        """
        Calculate confidence score based on prediction interval width.
        
        Args:
            prediction: Prediction dictionary with price and bounds
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            price = prediction['predicted_price']
            lower = prediction['confidence_lower']
            upper = prediction['confidence_upper']
            
            if price <= 0:
                return 0.0
            
            # Calculate relative interval width
            interval_width = (upper - lower) / price
            
            # Convert to confidence score (narrower interval = higher confidence)
            confidence = max(0.0, min(1.0, 1.0 - interval_width))
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0
    
    def _calculate_consensus(
        self, 
        predictions: List[ModelPrediction], 
        symbol: str, 
        horizon: str
    ) -> AggregatedPrediction:
        """
        Calculate consensus prediction from individual model predictions.
        
        Args:
            predictions: List of individual model predictions
            symbol: Stock symbol
            horizon: Prediction horizon
            
        Returns:
            Aggregated consensus prediction
        """
        try:
            if not predictions:
                raise ValueError("No predictions to aggregate")
            
            # Weight predictions by confidence score
            total_weight = sum(pred.confidence_score for pred in predictions)
            
            if total_weight == 0:
                # Equal weighting if no confidence scores
                weights = [1.0 / len(predictions)] * len(predictions)
            else:
                weights = [pred.confidence_score / total_weight for pred in predictions]
            
            # Calculate weighted consensus
            consensus_price = sum(
                pred.predicted_price * weight 
                for pred, weight in zip(predictions, weights)
            )
            
            confidence_lower = sum(
                pred.confidence_lower * weight 
                for pred, weight in zip(predictions, weights)
            )
            
            confidence_upper = sum(
                pred.confidence_upper * weight 
                for pred, weight in zip(predictions, weights)
            )
            
            # Calculate prediction variance
            price_variance = np.var([pred.predicted_price for pred in predictions])
            
            # Calculate overall confidence score
            overall_confidence = sum(
                pred.confidence_score * weight 
                for pred, weight in zip(predictions, weights)
            )
            
            return AggregatedPrediction(
                symbol=symbol,
                horizon=horizon,
                consensus_price=consensus_price,
                confidence_lower=confidence_lower,
                confidence_upper=confidence_upper,
                model_count=len(predictions),
                individual_predictions=predictions,
                timestamp=int(datetime.now().timestamp()),
                confidence_score=overall_confidence,
                prediction_variance=float(price_variance)
            )
            
        except Exception as e:
            logger.error(f"Error calculating consensus: {e}")
            raise
    
    async def get_cached_prediction(
        self, 
        symbol: str, 
        horizon: str
    ) -> Optional[AggregatedPrediction]:
        """
        Get cached prediction if available and not expired.
        
        Args:
            symbol: Stock symbol
            horizon: Prediction horizon
            
        Returns:
            Cached prediction or None if not available/expired
        """
        try:
            cache_key = f"{symbol}_{horizon}"
            
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                
                # Check if cache is still valid
                if datetime.now().timestamp() - cached_data['timestamp'] < self.cache_ttl:
                    logger.info(f"Returning cached prediction for {symbol} {horizon}")
                    return cached_data['prediction']
                else:
                    # Remove expired cache entry
                    del self.cache[cache_key]
            
            return None
            
        except Exception as e:
            logger.error(f"Error accessing cache for {symbol}: {e}")
            return None
    
    def _cache_prediction(self, prediction: AggregatedPrediction) -> None:
        """
        Cache prediction result.
        
        Args:
            prediction: Aggregated prediction to cache
        """
        try:
            cache_key = f"{prediction.symbol}_{prediction.horizon}"
            
            self.cache[cache_key] = {
                'prediction': prediction,
                'timestamp': datetime.now().timestamp()
            }
            
            # Clean up old cache entries (simple cleanup)
            current_time = datetime.now().timestamp()
            expired_keys = [
                key for key, value in self.cache.items()
                if current_time - value['timestamp'] > self.cache_ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            logger.debug(f"Cached prediction for {prediction.symbol} {prediction.horizon}")
            
        except Exception as e:
            logger.error(f"Error caching prediction: {e}")
    
    async def store_predictions(self, prediction: AggregatedPrediction) -> None:
        """
        Store individual predictions in database.
        
        Args:
            prediction: Aggregated prediction containing individual predictions
        """
        try:
            # Store each individual prediction
            for individual_pred in prediction.individual_predictions:
                price_prediction = PricePrediction(
                    symbol=individual_pred.symbol,
                    timestamp=individual_pred.timestamp,
                    horizon=individual_pred.horizon,
                    predicted_price=individual_pred.predicted_price,
                    confidence_lower=individual_pred.confidence_lower,
                    confidence_upper=individual_pred.confidence_upper,
                    model=individual_pred.model
                )
                
                await self.predictions_repo.create_item(price_prediction)
            
            logger.info(f"Stored {len(prediction.individual_predictions)} predictions for {prediction.symbol}")
            
        except Exception as e:
            logger.error(f"Error storing predictions: {e}")
            raise
    
    async def get_prediction(
        self, 
        symbol: str, 
        horizon: str,
        input_data: Dict[str, Any],
        use_cache: bool = True
    ) -> Optional[AggregatedPrediction]:
        """
        Get prediction for symbol and horizon with caching.
        
        Args:
            symbol: Stock symbol
            horizon: Prediction horizon ('1d', '3d', '7d')
            input_data: Input data for models
            use_cache: Whether to use cached predictions
            
        Returns:
            Aggregated prediction result
        """
        try:
            # Check cache first if enabled
            if use_cache:
                cached_prediction = await self.get_cached_prediction(symbol, horizon)
                if cached_prediction:
                    return cached_prediction
            
            # Generate new prediction
            prediction = await self.aggregate_predictions(symbol, horizon, input_data)
            
            if prediction:
                # Cache the result
                self._cache_prediction(prediction)
                
                # Store in database
                await self.store_predictions(prediction)
                
                logger.info(f"Generated new prediction for {symbol} {horizon}")
                return prediction
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}: {e}")
            return None
    
    async def get_multiple_predictions(
        self, 
        symbol: str,
        horizons: List[str] = ['1d', '3d', '7d'],
        input_data: Dict[str, Any] = None,
        use_cache: bool = True
    ) -> Dict[str, Optional[AggregatedPrediction]]:
        """
        Get predictions for multiple horizons.
        
        Args:
            symbol: Stock symbol
            horizons: List of prediction horizons
            input_data: Input data for models
            use_cache: Whether to use cached predictions
            
        Returns:
            Dictionary mapping horizons to predictions
        """
        try:
            if input_data is None:
                input_data = {}
            
            # Get predictions for all horizons concurrently
            tasks = [
                self.get_prediction(symbol, horizon, input_data, use_cache)
                for horizon in horizons
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Map results to horizons
            predictions = {}
            for horizon, result in zip(horizons, results):
                if isinstance(result, Exception):
                    logger.error(f"Error getting prediction for {symbol} {horizon}: {result}")
                    predictions[horizon] = None
                else:
                    predictions[horizon] = result
            
            logger.info(f"Retrieved predictions for {symbol} across {len(horizons)} horizons")
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting multiple predictions for {symbol}: {e}")
            return {horizon: None for horizon in horizons}
    
    async def get_historical_predictions(
        self, 
        symbol: str, 
        hours_back: int = 24
    ) -> List[PricePrediction]:
        """
        Get historical predictions for analysis.
        
        Args:
            symbol: Stock symbol
            hours_back: Number of hours to look back
            
        Returns:
            List of historical predictions
        """
        try:
            return await self.predictions_repo.get_recent_predictions(hours_back)
            
        except Exception as e:
            logger.error(f"Error getting historical predictions: {e}")
            return []
    
    async def calculate_prediction_accuracy(
        self, 
        symbol: str, 
        actual_prices: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate prediction accuracy metrics.
        
        Args:
            symbol: Stock symbol
            actual_prices: Dictionary mapping horizons to actual prices
            
        Returns:
            Dictionary with accuracy metrics by horizon
        """
        try:
            accuracy_metrics = {}
            
            for horizon, actual_price in actual_prices.items():
                # Get recent predictions for this horizon
                predictions = await self.predictions_repo.get_predictions_by_horizon(
                    symbol, horizon, limit=10
                )
                
                if not predictions:
                    continue
                
                # Calculate metrics
                predicted_prices = [pred.predicted_price for pred in predictions]
                errors = [abs(pred - actual_price) for pred in predicted_prices]
                
                if errors:
                    mae = sum(errors) / len(errors)
                    mape = sum(abs(error / actual_price) for error in errors) / len(errors) * 100
                    
                    accuracy_metrics[horizon] = {
                        'mae': mae,
                        'mape': mape,
                        'prediction_count': len(predictions),
                        'actual_price': actual_price
                    }
            
            return accuracy_metrics
            
        except Exception as e:
            logger.error(f"Error calculating prediction accuracy: {e}")
            return {}
    
    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self.cache.clear()
        logger.info("Prediction cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        current_time = datetime.now().timestamp()
        
        valid_entries = sum(
            1 for value in self.cache.values()
            if current_time - value['timestamp'] < self.cache_ttl
        )
        
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.cache) - valid_entries,
            'cache_ttl_seconds': self.cache_ttl
        }