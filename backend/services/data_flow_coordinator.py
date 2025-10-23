"""
Data flow coordinator that ensures proper sequencing and timing of the complete pipeline.
This service manages the flow from data ingestion through ML processing to trading decisions.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages"""
    DATA_INGESTION = "data_ingestion"
    LAMBDA_PROCESSING = "lambda_processing"
    ML_PREDICTION = "ml_prediction"
    TRADING_SIGNAL = "trading_signal"
    TRADE_EXECUTION = "trade_execution"
    EXPLANATION_GENERATION = "explanation_generation"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineContext:
    """Context object that flows through the pipeline"""
    symbols: List[str]
    start_time: datetime
    current_stage: PipelineStage
    data: Dict[str, Any]
    errors: List[str]
    metrics: Dict[str, Any]
    config: Dict[str, Any]


class DataFlowCoordinator:
    """Coordinates the complete data flow pipeline with proper sequencing"""
    
    def __init__(self):
        self.pipeline_stages = [
            PipelineStage.DATA_INGESTION,
            PipelineStage.LAMBDA_PROCESSING,
            PipelineStage.ML_PREDICTION,
            PipelineStage.TRADING_SIGNAL,
            PipelineStage.TRADE_EXECUTION,
            PipelineStage.EXPLANATION_GENERATION,
            PipelineStage.COMPLETED
        ]
        
        # Stage handlers
        self.stage_handlers: Dict[PipelineStage, Callable] = {}
        self.stage_timeouts: Dict[PipelineStage, int] = {
            PipelineStage.DATA_INGESTION: 60,
            PipelineStage.LAMBDA_PROCESSING: 120,
            PipelineStage.ML_PREDICTION: 180,
            PipelineStage.TRADING_SIGNAL: 30,
            PipelineStage.TRADE_EXECUTION: 60,
            PipelineStage.EXPLANATION_GENERATION: 90
        }
        
        # Pipeline state tracking
        self.active_pipelines: Dict[str, PipelineContext] = {}
        
    def register_stage_handler(self, stage: PipelineStage, handler: Callable):
        """Register a handler for a specific pipeline stage"""
        self.stage_handlers[stage] = handler
        logger.info(f"Registered handler for stage: {stage.value}")
    
    async def execute_pipeline(
        self,
        symbols: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete pipeline with proper stage sequencing
        
        Args:
            symbols: List of stock symbols to process
            config: Pipeline configuration options
            
        Returns:
            Pipeline execution results
        """
        pipeline_id = f"pipeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(tuple(symbols)) % 10000}"
        
        # Initialize pipeline context
        context = PipelineContext(
            symbols=symbols,
            start_time=datetime.utcnow(),
            current_stage=PipelineStage.DATA_INGESTION,
            data={},
            errors=[],
            metrics={
                'stages_completed': 0,
                'total_stages': len(self.pipeline_stages) - 1,  # Exclude COMPLETED
                'stage_timings': {}
            },
            config=config or {}
        )
        
        self.active_pipelines[pipeline_id] = context
        
        try:
            logger.info(f"Starting pipeline {pipeline_id} for symbols: {symbols}")
            
            # Execute each stage in sequence
            for stage in self.pipeline_stages[:-1]:  # Exclude COMPLETED stage
                stage_start = datetime.utcnow()
                context.current_stage = stage
                
                logger.info(f"Pipeline {pipeline_id}: Executing stage {stage.value}")
                
                try:
                    # Execute stage with timeout
                    timeout = self.stage_timeouts.get(stage, 300)
                    await asyncio.wait_for(
                        self._execute_stage(context, stage),
                        timeout=timeout
                    )
                    
                    # Record stage timing
                    stage_duration = (datetime.utcnow() - stage_start).total_seconds()
                    context.metrics['stage_timings'][stage.value] = stage_duration
                    context.metrics['stages_completed'] += 1
                    
                    logger.info(f"Pipeline {pipeline_id}: Completed stage {stage.value} in {stage_duration:.2f}s")
                    
                except asyncio.TimeoutError:
                    error_msg = f"Stage {stage.value} timed out after {timeout}s"
                    context.errors.append(error_msg)
                    logger.error(f"Pipeline {pipeline_id}: {error_msg}")
                    context.current_stage = PipelineStage.FAILED
                    break
                    
                except Exception as e:
                    error_msg = f"Stage {stage.value} failed: {str(e)}"
                    context.errors.append(error_msg)
                    logger.error(f"Pipeline {pipeline_id}: {error_msg}")
                    context.current_stage = PipelineStage.FAILED
                    break
            
            # Mark as completed if no errors
            if context.current_stage != PipelineStage.FAILED:
                context.current_stage = PipelineStage.COMPLETED
                logger.info(f"Pipeline {pipeline_id}: Completed successfully")
            
            # Calculate total execution time
            total_duration = (datetime.utcnow() - context.start_time).total_seconds()
            context.metrics['total_duration'] = total_duration
            
            # Build result
            result = {
                'pipeline_id': pipeline_id,
                'status': 'success' if context.current_stage == PipelineStage.COMPLETED else 'failed',
                'symbols': context.symbols,
                'start_time': context.start_time.isoformat(),
                'end_time': datetime.utcnow().isoformat(),
                'duration_seconds': total_duration,
                'stages_completed': context.metrics['stages_completed'],
                'total_stages': context.metrics['total_stages'],
                'stage_timings': context.metrics['stage_timings'],
                'data': context.data,
                'errors': context.errors,
                'final_stage': context.current_stage.value
            }
            
            return result
            
        finally:
            # Clean up pipeline context
            if pipeline_id in self.active_pipelines:
                del self.active_pipelines[pipeline_id]
    
    async def _execute_stage(self, context: PipelineContext, stage: PipelineStage):
        """Execute a specific pipeline stage"""
        if stage not in self.stage_handlers:
            logger.warning(f"No handler registered for stage: {stage.value}")
            return
        
        handler = self.stage_handlers[stage]
        
        try:
            # Execute the stage handler
            stage_result = await handler(context)
            
            # Store stage result in context
            if stage_result:
                context.data[stage.value] = stage_result
                
        except Exception as e:
            logger.error(f"Error in stage {stage.value}: {e}")
            raise
    
    def get_pipeline_status(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running pipeline"""
        if pipeline_id not in self.active_pipelines:
            return None
        
        context = self.active_pipelines[pipeline_id]
        
        return {
            'pipeline_id': pipeline_id,
            'symbols': context.symbols,
            'current_stage': context.current_stage.value,
            'stages_completed': context.metrics['stages_completed'],
            'total_stages': context.metrics['total_stages'],
            'start_time': context.start_time.isoformat(),
            'elapsed_seconds': (datetime.utcnow() - context.start_time).total_seconds(),
            'errors': context.errors,
            'stage_timings': context.metrics['stage_timings']
        }
    
    def list_active_pipelines(self) -> List[Dict[str, Any]]:
        """List all currently active pipelines"""
        return [
            self.get_pipeline_status(pipeline_id)
            for pipeline_id in self.active_pipelines.keys()
        ]
    
    async def wait_for_stage_completion(
        self,
        pipeline_id: str,
        target_stage: PipelineStage,
        timeout: int = 300
    ) -> bool:
        """
        Wait for a pipeline to reach a specific stage
        
        Args:
            pipeline_id: Pipeline identifier
            target_stage: Stage to wait for
            timeout: Maximum wait time in seconds
            
        Returns:
            True if stage reached, False if timeout or error
        """
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).total_seconds() < timeout:
            if pipeline_id not in self.active_pipelines:
                return False
            
            context = self.active_pipelines[pipeline_id]
            
            # Check if target stage reached
            if context.current_stage == target_stage:
                return True
            
            # Check if pipeline failed
            if context.current_stage == PipelineStage.FAILED:
                return False
            
            # Check if pipeline completed (target stage passed)
            if context.current_stage == PipelineStage.COMPLETED:
                target_index = self.pipeline_stages.index(target_stage)
                completed_index = self.pipeline_stages.index(PipelineStage.COMPLETED)
                return target_index < completed_index
            
            await asyncio.sleep(1)  # Check every second
        
        return False  # Timeout
    
    def get_stage_metrics(self) -> Dict[str, Any]:
        """Get metrics for all pipeline stages"""
        stage_metrics = {}
        
        for stage in PipelineStage:
            if stage in [PipelineStage.COMPLETED, PipelineStage.FAILED]:
                continue
                
            stage_metrics[stage.value] = {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'average_duration': 0,
                'timeout_count': 0
            }
        
        # This would be populated from historical data in a real implementation
        return stage_metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the data flow coordinator"""
        return {
            'status': 'healthy',
            'active_pipelines': len(self.active_pipelines),
            'registered_handlers': len(self.stage_handlers),
            'supported_stages': [stage.value for stage in self.pipeline_stages],
            'timestamp': datetime.utcnow().isoformat()
        }


# Global coordinator instance
_coordinator_instance = None

def get_coordinator() -> DataFlowCoordinator:
    """Get singleton coordinator instance"""
    global _coordinator_instance
    if _coordinator_instance is None:
        _coordinator_instance = DataFlowCoordinator()
    return _coordinator_instance