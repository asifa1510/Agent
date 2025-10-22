"""
FastAPI router for explanation endpoints.
Provides API endpoints for generating and retrieving trade and portfolio explanations.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel

from ..models.data_models import Trade, TradeExplanation, PortfolioPosition, RiskMetrics
from ..services.explanation_service import ExplanationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explanations", tags=["explanations"])

# Dependency to get explanation service
def get_explanation_service() -> ExplanationService:
    """Dependency to provide explanation service instance."""
    return ExplanationService()


# Request/Response models
class ExplanationRequest(BaseModel):
    """Request model for generating trade explanations."""
    trade: Trade
    include_context: bool = True
    sentiment_data: Optional[Dict[str, Any]] = None
    prediction_data: Optional[Dict[str, Any]] = None
    market_data: Optional[Dict[str, Any]] = None


class PortfolioExplanationRequest(BaseModel):
    """Request model for generating portfolio explanations."""
    positions: List[PortfolioPosition]
    risk_metrics: Optional[RiskMetrics] = None


class BatchExplanationRequest(BaseModel):
    """Request model for batch explanation generation."""
    trades: List[Trade]
    include_context: bool = True


class ExplanationResponse(BaseModel):
    """Response model for explanation generation."""
    explanation: TradeExplanation
    status: str = "success"
    message: Optional[str] = None


class PortfolioExplanationResponse(BaseModel):
    """Response model for portfolio explanation."""
    explanation: str
    confidence: float
    supporting_data: Dict[str, Any]
    response_time_seconds: float
    timestamp: int
    status: str = "success"


class BatchExplanationResponse(BaseModel):
    """Response model for batch explanations."""
    explanations: List[TradeExplanation]
    total_requested: int
    total_generated: int
    status: str = "success"


class ServiceStatusResponse(BaseModel):
    """Response model for service status."""
    service_name: str
    status: str
    bedrock_connection: bool
    model_id: str
    region: str
    timestamp: int


@router.post("/trade", response_model=ExplanationResponse)
async def generate_trade_explanation(
    request: ExplanationRequest,
    explanation_service: ExplanationService = Depends(get_explanation_service)
):
    """
    Generate explanation for a single trade decision.
    
    Args:
        request: Trade explanation request with trade data and optional context
        explanation_service: Injected explanation service
        
    Returns:
        Generated trade explanation with metadata
    """
    try:
        # Get contextual data if requested and not provided
        sentiment_data = request.sentiment_data
        prediction_data = request.prediction_data
        market_data = request.market_data
        
        if request.include_context and not any([sentiment_data, prediction_data, market_data]):
            context = await explanation_service.get_contextual_data(request.trade.symbol)
            sentiment_data = context.get('sentiment_data')
            prediction_data = context.get('prediction_data')
            market_data = context.get('market_data')
        
        # Generate explanation
        explanation = await explanation_service.generate_trade_explanation(
            trade=request.trade,
            sentiment_data=sentiment_data,
            prediction_data=prediction_data,
            market_data=market_data
        )
        
        return ExplanationResponse(
            explanation=explanation,
            status="success",
            message=f"Generated explanation for {request.trade.symbol} trade"
        )
        
    except Exception as e:
        logger.error(f"Error generating trade explanation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate trade explanation: {str(e)}"
        )


@router.post("/portfolio", response_model=PortfolioExplanationResponse)
async def generate_portfolio_explanation(
    request: PortfolioExplanationRequest,
    explanation_service: ExplanationService = Depends(get_explanation_service)
):
    """
    Generate explanation for portfolio performance and composition.
    
    Args:
        request: Portfolio explanation request with positions and risk metrics
        explanation_service: Injected explanation service
        
    Returns:
        Generated portfolio explanation with analysis
    """
    try:
        result = await explanation_service.generate_portfolio_explanation(
            positions=request.positions,
            risk_metrics=request.risk_metrics
        )
        
        return PortfolioExplanationResponse(
            explanation=result['explanation'],
            confidence=result['confidence'],
            supporting_data=result['supporting_data'],
            response_time_seconds=result['response_time_seconds'],
            timestamp=result['timestamp'],
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error generating portfolio explanation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate portfolio explanation: {str(e)}"
        )


@router.post("/batch", response_model=BatchExplanationResponse)
async def generate_batch_explanations(
    request: BatchExplanationRequest,
    background_tasks: BackgroundTasks,
    explanation_service: ExplanationService = Depends(get_explanation_service)
):
    """
    Generate explanations for multiple trades in batch.
    
    Args:
        request: Batch explanation request with list of trades
        background_tasks: FastAPI background tasks for async processing
        explanation_service: Injected explanation service
        
    Returns:
        List of generated explanations with batch statistics
    """
    try:
        explanations = await explanation_service.batch_generate_explanations(
            trades=request.trades,
            include_context=request.include_context
        )
        
        return BatchExplanationResponse(
            explanations=explanations,
            total_requested=len(request.trades),
            total_generated=len(explanations),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error generating batch explanations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate batch explanations: {str(e)}"
        )


@router.get("/status", response_model=ServiceStatusResponse)
async def get_service_status(
    explanation_service: ExplanationService = Depends(get_explanation_service)
):
    """
    Get current status of the explanation service.
    
    Args:
        explanation_service: Injected explanation service
        
    Returns:
        Service status including Bedrock connection health
    """
    try:
        status = explanation_service.get_service_status()
        return ServiceStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Error getting service status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get service status: {str(e)}"
        )


@router.get("/test-connection")
async def test_bedrock_connection(
    explanation_service: ExplanationService = Depends(get_explanation_service)
):
    """
    Test connection to AWS Bedrock service.
    
    Args:
        explanation_service: Injected explanation service
        
    Returns:
        Connection test result
    """
    try:
        connection_ok = explanation_service.test_bedrock_connection()
        
        return {
            "bedrock_connection": connection_ok,
            "status": "success" if connection_ok else "failed",
            "message": "Bedrock connection test completed"
        }
        
    except Exception as e:
        logger.error(f"Error testing Bedrock connection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to test Bedrock connection: {str(e)}"
        )