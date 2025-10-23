"""
API endpoints for system integration and orchestration
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from ..services.integration_orchestrator import get_orchestrator
from ..models.data_models import ApiResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integration", tags=["integration"])


@router.post("/run-pipeline")
async def run_complete_pipeline(
    symbols: List[str],
    include_trading: bool = True,
    include_explanations: bool = True
) -> ApiResponse[Dict[str, Any]]:
    """
    Run the complete end-to-end pipeline for given symbols
    
    Args:
        symbols: List of stock symbols to process
        include_trading: Whether to execute trading signals
        include_explanations: Whether to generate explanations
        
    Returns:
        Pipeline execution results
    """
    try:
        if not symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required")
        
        # Validate symbols
        valid_symbols = [symbol.upper().strip() for symbol in symbols if symbol.strip()]
        if not valid_symbols:
            raise HTTPException(status_code=400, detail="No valid symbols provided")
        
        logger.info(f"Running complete pipeline for symbols: {valid_symbols}")
        
        orchestrator = await get_orchestrator()
        result = await orchestrator.run_complete_pipeline(
            symbols=valid_symbols,
            include_trading=include_trading,
            include_explanations=include_explanations
        )
        
        return ApiResponse(
            success=result['status'] == 'success',
            data=result,
            message=f"Pipeline executed for {len(valid_symbols)} symbols"
        )
        
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-pipeline-async")
async def run_complete_pipeline_async(
    background_tasks: BackgroundTasks,
    symbols: List[str],
    include_trading: bool = True,
    include_explanations: bool = True
) -> ApiResponse[Dict[str, str]]:
    """
    Run the complete pipeline asynchronously in the background
    
    Args:
        background_tasks: FastAPI background tasks
        symbols: List of stock symbols to process
        include_trading: Whether to execute trading signals
        include_explanations: Whether to generate explanations
        
    Returns:
        Task submission confirmation
    """
    try:
        if not symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required")
        
        valid_symbols = [symbol.upper().strip() for symbol in symbols if symbol.strip()]
        if not valid_symbols:
            raise HTTPException(status_code=400, detail="No valid symbols provided")
        
        # Add background task
        background_tasks.add_task(
            _run_pipeline_background,
            valid_symbols,
            include_trading,
            include_explanations
        )
        
        return ApiResponse(
            success=True,
            data={
                "status": "submitted",
                "symbols": valid_symbols,
                "timestamp": datetime.utcnow().isoformat()
            },
            message=f"Pipeline task submitted for {len(valid_symbols)} symbols"
        )
        
    except Exception as e:
        logger.error(f"Error submitting pipeline task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_system_health() -> ApiResponse[Dict[str, Any]]:
    """
    Get comprehensive system health status
    
    Returns:
        System health information
    """
    try:
        orchestrator = await get_orchestrator()
        health_status = await orchestrator.get_system_health()
        
        return ApiResponse(
            success=health_status['overall_status'] in ['healthy', 'degraded'],
            data=health_status,
            message=f"System status: {health_status['overall_status']}"
        )
        
    except Exception as e:
        logger.error(f"Error checking system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_system_metrics() -> ApiResponse[Dict[str, Any]]:
    """
    Get current system processing metrics
    
    Returns:
        System metrics
    """
    try:
        orchestrator = await get_orchestrator()
        metrics = orchestrator.get_metrics()
        
        return ApiResponse(
            success=True,
            data=metrics,
            message="System metrics retrieved"
        )
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/reset")
async def reset_system_metrics() -> ApiResponse[Dict[str, str]]:
    """
    Reset system processing metrics
    
    Returns:
        Reset confirmation
    """
    try:
        orchestrator = await get_orchestrator()
        orchestrator.reset_metrics()
        
        return ApiResponse(
            success=True,
            data={"status": "reset", "timestamp": datetime.utcnow().isoformat()},
            message="System metrics reset"
        )
        
    except Exception as e:
        logger.error(f"Error resetting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/data/ingest")
async def ingest_data_only(
    symbols: List[str],
    include_social: bool = True,
    include_news: bool = True,
    include_market: bool = True
) -> ApiResponse[Dict[str, Any]]:
    """
    Run only the data ingestion part of the pipeline
    
    Args:
        symbols: List of stock symbols
        include_social: Include social media data
        include_news: Include news data
        include_market: Include market data
        
    Returns:
        Data ingestion results
    """
    try:
        if not symbols:
            raise HTTPException(status_code=400, detail="At least one symbol is required")
        
        valid_symbols = [symbol.upper().strip() for symbol in symbols if symbol.strip()]
        if not valid_symbols:
            raise HTTPException(status_code=400, detail="No valid symbols provided")
        
        orchestrator = await get_orchestrator()
        
        # Run only data collection and streaming
        result = await orchestrator._collect_and_stream_data(valid_symbols)
        
        return ApiResponse(
            success=True,
            data=result,
            message=f"Data ingestion completed for {len(valid_symbols)} symbols"
        )
        
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_integration_status() -> ApiResponse[Dict[str, Any]]:
    """
    Get current integration system status
    
    Returns:
        Integration status information
    """
    try:
        orchestrator = await get_orchestrator()
        
        status = {
            "system_status": "operational",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "data_integration": "active",
                "kinesis_producer": "active",
                "ml_services": "active",
                "trading_services": "active"
            },
            "metrics": orchestrator.get_metrics()
        }
        
        return ApiResponse(
            success=True,
            data=status,
            message="Integration status retrieved"
        )
        
    except Exception as e:
        logger.error(f"Error getting integration status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_pipeline_background(
    symbols: List[str],
    include_trading: bool,
    include_explanations: bool
) -> None:
    """
    Background task to run the complete pipeline
    
    Args:
        symbols: List of stock symbols
        include_trading: Whether to execute trading
        include_explanations: Whether to generate explanations
    """
    try:
        logger.info(f"Starting background pipeline for symbols: {symbols}")
        
        orchestrator = await get_orchestrator()
        result = await orchestrator.run_complete_pipeline(
            symbols=symbols,
            include_trading=include_trading,
            include_explanations=include_explanations
        )
        
        logger.info(f"Background pipeline completed: {result['status']}")
        
    except Exception as e:
        logger.error(f"Error in background pipeline: {e}")