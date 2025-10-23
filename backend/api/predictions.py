"""
FastAPI router for price prediction endpoints.
Provides API endpoints for retrieving and managing price predictions.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel
from datetime import datetime

from ..models.data_models import PricePrediction
from ..services.prediction_service import PredictionService, AggregatedPrediction
from ..repositories.predictions_repository import PredictionsRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])

# Initialize services
prediction_service = PredictionService()
predictions_repo = PredictionsRepository()

# Response models
class PredictionsResponse(BaseModel):
    """Response model for prediction data."""
    data: List[PricePrediction]
    total_count: int
    status: str = "success"

class AggregatedPredictionResponse(BaseModel):
    """Response model for aggregated prediction."""
    symbol: str
    horizon: str
    consensus_price: float
    confidence_lower: float
    confidence_upper: float
    model_count: int
    confidence_score: float
    prediction_variance: float
    individual_predictions: List[Dict[str, Any]]
    timestamp: int
    status: str = "success"

class MultiplePredictionsResponse(BaseModel):
    """Response model for multiple horizon predictions."""
    symbol: str
    predictions: Dict[str, Optional[AggregatedPredictionResponse]]
    timestamp: int
    status: str = "success"

class PredictionSummaryResponse(BaseModel):
    """Response model for prediction summary."""
    symbol: str
    latest_predictions: List[PricePrediction]
    consensus_price: Optional[float]
    confidence_range: Optional[tuple[float, float]]
    timestamp: int
    status: str = "success"

class PredictionRequest(BaseModel):
    """Request model for generating predictions."""
    symbol: str
    horizons: Optional[List[str]] = ['1d', '3d', '7d']
    input_data: Optional[Dict[str, Any]] = None
    use_cache: bool = True

@router.get("/", response_model=PredictionsResponse)
async def get_predictions(
    symbol: Optional[str] = Query(None, description="Stock symbol filter"),
    horizon: Optional[str] = Query(None, description="Prediction horizon filter"),
    model: Optional[str] = Query(None, description="Model type filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records")
):
    """
    Retrieve price predictions with optional filtering.
    
    Args:
        symbol: Optional stock symbol filter
        horizon: Optional prediction horizon filter (1d, 3d, 7d)
        model: Optional model type filter (lstm, xgboost)
        limit: Maximum number of records to return
        
    Returns:
        List of price predictions matching the criteria
    """
    try:
        predictions = []
        
        if symbol and horizon:
            # Get predictions for specific symbol and horizon
            predictions = await predictions_repo.get_predictions_by_horizon(
                symbol.upper(), horizon, limit
            )
        elif symbol and model:
            # Get predictions for specific symbol and model
            predictions = await predictions_repo.get_predictions_by_model(
                symbol.upper(), model, limit
            )
        elif symbol:
            # Get latest predictions for symbol
            predictions = await predictions_repo.get_latest_predictions(
                symbol.upper(), limit
            )
        else:
            # Get recent predictions across all symbols
            predictions = await predictions_repo.get_recent_predictions(
                hours_back=24, limit=limit
            )
        
        return PredictionsResponse(
            data=predictions,
            total_count=len(predictions),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve predictions: {str(e)}"
        )

@router.post("/generate", response_model=MultiplePredictionsResponse)
async def generate_predictions(request: PredictionRequest):
    """
    Generate new predictions for a symbol across multiple horizons.
    
    Args:
        request: Prediction request with symbol, horizons, and input data
        
    Returns:
        Aggregated predictions for all requested horizons
    """
    try:
        symbol = request.symbol.upper()
        
        # Get predictions for multiple horizons
        predictions = await prediction_service.get_multiple_predictions(
            symbol=symbol,
            horizons=request.horizons,
            input_data=request.input_data or {},
            use_cache=request.use_cache
        )
        
        # Convert to response format
        response_predictions = {}
        for horizon, prediction in predictions.items():
            if prediction:
                response_predictions[horizon] = AggregatedPredictionResponse(
                    symbol=prediction.symbol,
                    horizon=prediction.horizon,
                    consensus_price=prediction.consensus_price,
                    confidence_lower=prediction.confidence_lower,
                    confidence_upper=prediction.confidence_upper,
                    model_count=prediction.model_count,
                    confidence_score=prediction.confidence_score,
                    prediction_variance=prediction.prediction_variance,
                    individual_predictions=[
                        {
                            'model': pred.model,
                            'predicted_price': pred.predicted_price,
                            'confidence_lower': pred.confidence_lower,
                            'confidence_upper': pred.confidence_upper,
                            'confidence_score': pred.confidence_score
                        }
                        for pred in prediction.individual_predictions
                    ],
                    timestamp=prediction.timestamp,
                    status="success"
                )
            else:
                response_predictions[horizon] = None
        
        return MultiplePredictionsResponse(
            symbol=symbol,
            predictions=response_predictions,
            timestamp=int(datetime.now().timestamp()),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error generating predictions for {request.symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate predictions: {str(e)}"
        )

@router.get("/{symbol}/consensus/{horizon}")
async def get_consensus_prediction(
    symbol: str,
    horizon: str
):
    """
    Get consensus prediction for a specific symbol and horizon.
    
    Args:
        symbol: Stock symbol
        horizon: Prediction horizon (1d, 3d, 7d)
        
    Returns:
        Consensus prediction from multiple models
    """
    try:
        symbol = symbol.upper()
        
        # Validate horizon
        if horizon not in ['1d', '3d', '7d']:
            raise HTTPException(
                status_code=400,
                detail="Invalid horizon. Must be one of: 1d, 3d, 7d"
            )
        
        # Get consensus from repository
        consensus = await predictions_repo.get_consensus_prediction(symbol, horizon)
        
        if not consensus:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for {symbol} {horizon}"
            )
        
        return {
            "symbol": symbol,
            "horizon": horizon,
            "consensus_price": consensus['consensus_price'],
            "confidence_lower": consensus['confidence_lower'],
            "confidence_upper": consensus['confidence_upper'],
            "model_count": consensus['model_count'],
            "timestamp": consensus['timestamp'],
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting consensus prediction for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get consensus prediction: {str(e)}"
        )

@router.get("/{symbol}/summary", response_model=PredictionSummaryResponse)
async def get_prediction_summary(
    symbol: str
):
    """
    Get prediction summary for a specific symbol.
    
    Args:
        symbol: Stock symbol to get predictions for
        
    Returns:
        Summary of latest predictions with consensus analysis
    """
    try:
        symbol = symbol.upper()
        
        # Get latest predictions
        latest_predictions = await predictions_repo.get_latest_predictions(symbol, limit=10)
        
        # Calculate consensus if predictions exist
        consensus_price = None
        confidence_range = None
        
        if latest_predictions:
            # Get consensus for 1d horizon as default
            consensus = await predictions_repo.get_consensus_prediction(symbol, '1d')
            if consensus:
                consensus_price = consensus['consensus_price']
                confidence_range = (consensus['confidence_lower'], consensus['confidence_upper'])
        
        return PredictionSummaryResponse(
            symbol=symbol,
            latest_predictions=latest_predictions,
            consensus_price=consensus_price,
            confidence_range=confidence_range,
            timestamp=int(datetime.now().timestamp()),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error getting prediction summary for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get prediction summary for {symbol}: {str(e)}"
        )

@router.get("/{symbol}/accuracy")
async def get_prediction_accuracy(
    symbol: str,
    actual_prices: Dict[str, float] = Body(...)
):
    """
    Calculate prediction accuracy metrics for a symbol.
    
    Args:
        symbol: Stock symbol
        actual_prices: Dictionary mapping horizons to actual prices
        
    Returns:
        Accuracy metrics by horizon
    """
    try:
        symbol = symbol.upper()
        
        accuracy_metrics = await prediction_service.calculate_prediction_accuracy(
            symbol, actual_prices
        )
        
        return {
            "symbol": symbol,
            "accuracy_metrics": accuracy_metrics,
            "timestamp": int(datetime.now().timestamp()),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error calculating prediction accuracy for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate prediction accuracy: {str(e)}"
        )

@router.delete("/cache")
async def clear_prediction_cache():
    """
    Clear the prediction cache.
    
    Returns:
        Success message
    """
    try:
        prediction_service.clear_cache()
        
        return {
            "message": "Prediction cache cleared successfully",
            "timestamp": int(datetime.now().timestamp()),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error clearing prediction cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear prediction cache: {str(e)}"
        )

@router.get("/cache/stats")
async def get_cache_stats():
    """
    Get prediction cache statistics.
    
    Returns:
        Cache statistics
    """
    try:
        stats = prediction_service.get_cache_stats()
        
        return {
            "cache_stats": stats,
            "timestamp": int(datetime.now().timestamp()),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache stats: {str(e)}"
        )