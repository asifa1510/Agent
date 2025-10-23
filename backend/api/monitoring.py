"""
API endpoints for monitoring, metrics, and system health
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta

from ..services.monitoring_service import monitoring_service, LogLevel
from ..models.data_models import ApiResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/metrics")
async def get_system_metrics(
    hours: int = Query(24, ge=1, le=168, description="Hours of metrics to retrieve")
) -> ApiResponse[Dict[str, Any]]:
    """
    Get system metrics for the specified time period
    
    Args:
        hours: Number of hours of metrics to retrieve (1-168)
        
    Returns:
        System metrics data
    """
    try:
        metrics = monitoring_service.get_dashboard_metrics(hours=hours)
        
        return ApiResponse(
            success=True,
            data={
                'metrics': metrics,
                'time_period_hours': hours,
                'timestamp': datetime.utcnow().isoformat()
            },
            message=f"Retrieved metrics for last {hours} hours"
        )
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/record")
async def record_custom_metric(
    name: str,
    value: float,
    unit: str = "Count",
    dimensions: Optional[Dict[str, str]] = None
) -> ApiResponse[Dict[str, str]]:
    """
    Record a custom metric
    
    Args:
        name: Metric name
        value: Metric value
        unit: Metric unit (default: Count)
        dimensions: Optional metric dimensions
        
    Returns:
        Recording confirmation
    """
    try:
        monitoring_service.record_metric(
            name=name,
            value=value,
            unit=unit,
            dimensions=dimensions
        )
        
        return ApiResponse(
            success=True,
            data={
                "status": "recorded",
                "metric_name": name,
                "timestamp": datetime.utcnow().isoformat()
            },
            message=f"Recorded metric: {name}"
        )
        
    except Exception as e:
        logger.error(f"Error recording custom metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logs/record")
async def record_custom_log(
    level: str,
    service: str,
    component: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> ApiResponse[Dict[str, str]]:
    """
    Record a custom log entry
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        service: Service name
        component: Component name
        message: Log message
        context: Optional context data
        trace_id: Optional trace ID
        user_id: Optional user ID
        
    Returns:
        Recording confirmation
    """
    try:
        # Validate log level
        try:
            log_level = LogLevel(level.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid log level: {level}")
        
        monitoring_service.log_structured(
            level=log_level,
            service=service,
            component=component,
            message=message,
            context=context,
            trace_id=trace_id,
            user_id=user_id
        )
        
        return ApiResponse(
            success=True,
            data={
                "status": "recorded",
                "log_level": level,
                "timestamp": datetime.utcnow().isoformat()
            },
            message=f"Recorded {level} log entry"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording custom log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/flush")
async def flush_monitoring_buffers() -> ApiResponse[Dict[str, str]]:
    """
    Manually flush monitoring buffers to CloudWatch
    
    Returns:
        Flush confirmation
    """
    try:
        await monitoring_service.flush_all()
        
        return ApiResponse(
            success=True,
            data={
                "status": "flushed",
                "timestamp": datetime.utcnow().isoformat()
            },
            message="Monitoring buffers flushed to CloudWatch"
        )
        
    except Exception as e:
        logger.error(f"Error flushing monitoring buffers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def get_monitoring_health() -> ApiResponse[Dict[str, Any]]:
    """
    Get monitoring service health status
    
    Returns:
        Monitoring service health information
    """
    try:
        health_data = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'buffer_status': {
                'metric_buffer_size': len(monitoring_service.metric_buffer),
                'log_buffer_size': len(monitoring_service.log_buffer),
                'max_buffer_size': monitoring_service.buffer_size
            },
            'configuration': {
                'namespace': monitoring_service.namespace,
                'flush_interval': monitoring_service.flush_interval,
                'log_groups': list(monitoring_service.log_groups.keys())
            }
        }
        
        return ApiResponse(
            success=True,
            data=health_data,
            message="Monitoring service is healthy"
        )
        
    except Exception as e:
        logger.error(f"Error getting monitoring health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_metrics(
    hours: int = Query(24, ge=1, le=168, description="Hours of performance data to retrieve")
) -> ApiResponse[Dict[str, Any]]:
    """
    Get performance metrics for system components
    
    Args:
        hours: Number of hours of performance data to retrieve
        
    Returns:
        Performance metrics data
    """
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        # Get performance metrics from CloudWatch
        performance_data = {}
        
        # API performance
        try:
            api_latency = monitoring_service.cloudwatch.get_metric_statistics(
                Namespace=monitoring_service.namespace,
                MetricName='APILatency',
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour
                Statistics=['Average', 'Maximum']
            )
            
            performance_data['api_latency'] = {
                'average_ms': [point['Average'] for point in api_latency['Datapoints']],
                'maximum_ms': [point['Maximum'] for point in api_latency['Datapoints']],
                'timestamps': [point['Timestamp'].isoformat() for point in api_latency['Datapoints']]
            }
        except Exception as e:
            logger.warning(f"Could not retrieve API latency metrics: {e}")
            performance_data['api_latency'] = {'error': str(e)}
        
        # ML inference performance
        try:
            ml_latency = monitoring_service.cloudwatch.get_metric_statistics(
                Namespace=monitoring_service.namespace,
                MetricName='MLInferenceTime',
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average', 'Maximum']
            )
            
            performance_data['ml_inference_latency'] = {
                'average_ms': [point['Average'] for point in ml_latency['Datapoints']],
                'maximum_ms': [point['Maximum'] for point in ml_latency['Datapoints']],
                'timestamps': [point['Timestamp'].isoformat() for point in ml_latency['Datapoints']]
            }
        except Exception as e:
            logger.warning(f"Could not retrieve ML inference metrics: {e}")
            performance_data['ml_inference_latency'] = {'error': str(e)}
        
        # Trade execution performance
        try:
            trade_latency = monitoring_service.cloudwatch.get_metric_statistics(
                Namespace=monitoring_service.namespace,
                MetricName='TradeExecutionTime',
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average', 'Maximum']
            )
            
            performance_data['trade_execution_latency'] = {
                'average_ms': [point['Average'] for point in trade_latency['Datapoints']],
                'maximum_ms': [point['Maximum'] for point in trade_latency['Datapoints']],
                'timestamps': [point['Timestamp'].isoformat() for point in trade_latency['Datapoints']]
            }
        except Exception as e:
            logger.warning(f"Could not retrieve trade execution metrics: {e}")
            performance_data['trade_execution_latency'] = {'error': str(e)}
        
        return ApiResponse(
            success=True,
            data={
                'performance_metrics': performance_data,
                'time_period_hours': hours,
                'timestamp': datetime.utcnow().isoformat()
            },
            message=f"Retrieved performance metrics for last {hours} hours"
        )
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/errors")
async def get_error_metrics(
    hours: int = Query(24, ge=1, le=168, description="Hours of error data to retrieve")
) -> ApiResponse[Dict[str, Any]]:
    """
    Get error metrics and recent error logs
    
    Args:
        hours: Number of hours of error data to retrieve
        
    Returns:
        Error metrics and logs
    """
    try:
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        error_data = {}
        
        # Get error count metrics
        try:
            error_metrics = monitoring_service.cloudwatch.get_metric_statistics(
                Namespace=monitoring_service.namespace,
                MetricName='Error',
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Sum'],
                Dimensions=[
                    {'Name': 'Service', 'Value': '*'},
                    {'Name': 'Component', 'Value': '*'}
                ]
            )
            
            error_data['total_errors'] = sum(point['Sum'] for point in error_metrics['Datapoints'])
            error_data['error_timeline'] = [
                {
                    'timestamp': point['Timestamp'].isoformat(),
                    'count': point['Sum']
                }
                for point in error_metrics['Datapoints']
            ]
        except Exception as e:
            logger.warning(f"Could not retrieve error metrics: {e}")
            error_data['error_metrics'] = {'error': str(e)}
        
        return ApiResponse(
            success=True,
            data={
                'error_data': error_data,
                'time_period_hours': hours,
                'timestamp': datetime.utcnow().isoformat()
            },
            message=f"Retrieved error data for last {hours} hours"
        )
        
    except Exception as e:
        logger.error(f"Error getting error metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))