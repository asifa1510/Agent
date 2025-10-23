"""
Monitoring and logging service for CloudWatch integration and performance tracking.
"""

import boto3
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import time
from functools import wraps

from ..config import settings

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to track"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class LogLevel(Enum):
    """Log levels for structured logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class MetricData:
    """Structured metric data"""
    name: str
    value: Union[int, float]
    unit: str
    timestamp: datetime
    dimensions: Dict[str, str]
    metric_type: MetricType


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    service: str
    component: str
    message: str
    context: Dict[str, Any]
    trace_id: Optional[str] = None
    user_id: Optional[str] = None


class MonitoringService:
    """Service for monitoring, logging, and alerting"""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch', region_name=settings.aws_region)
        self.logs_client = boto3.client('logs', region_name=settings.aws_region)
        
        # Metric namespace
        self.namespace = "SentimentTradingAgent"
        
        # Log group names
        self.log_groups = {
            'application': '/aws/sentiment-trading-agent/application',
            'api': '/aws/sentiment-trading-agent/api',
            'ml': '/aws/sentiment-trading-agent/ml',
            'trading': '/aws/sentiment-trading-agent/trading',
            'data_pipeline': '/aws/sentiment-trading-agent/data-pipeline',
            'errors': '/aws/sentiment-trading-agent/errors'
        }
        
        # Metric buffer for batch sending
        self.metric_buffer: List[MetricData] = []
        self.log_buffer: List[LogEntry] = []
        self.buffer_size = 20
        self.flush_interval = 30  # seconds
        
        # Performance tracking
        self.performance_metrics = {}
        
        # Initialize log groups
        asyncio.create_task(self._ensure_log_groups_exist())
    
    async def _ensure_log_groups_exist(self):
        """Ensure all required CloudWatch log groups exist"""
        try:
            for log_group_name in self.log_groups.values():
                try:
                    self.logs_client.create_log_group(logGroupName=log_group_name)
                    logger.info(f"Created log group: {log_group_name}")
                except self.logs_client.exceptions.ResourceAlreadyExistsException:
                    pass  # Log group already exists
                except Exception as e:
                    logger.error(f"Error creating log group {log_group_name}: {e}")
        except Exception as e:
            logger.error(f"Error ensuring log groups exist: {e}")
    
    def log_structured(
        self,
        level: LogLevel,
        service: str,
        component: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Log structured data to CloudWatch"""
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            service=service,
            component=component,
            message=message,
            context=context or {},
            trace_id=trace_id,
            user_id=user_id
        )
        
        self.log_buffer.append(log_entry)
        
        # Also log to local logger
        local_logger = logging.getLogger(f"{service}.{component}")
        log_data = {
            'message': message,
            'context': context,
            'trace_id': trace_id,
            'user_id': user_id
        }
        
        if level == LogLevel.DEBUG:
            local_logger.debug(json.dumps(log_data))
        elif level == LogLevel.INFO:
            local_logger.info(json.dumps(log_data))
        elif level == LogLevel.WARNING:
            local_logger.warning(json.dumps(log_data))
        elif level == LogLevel.ERROR:
            local_logger.error(json.dumps(log_data))
        elif level == LogLevel.CRITICAL:
            local_logger.critical(json.dumps(log_data))
        
        # Flush if buffer is full
        if len(self.log_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_logs())
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        unit: str = "Count",
        dimensions: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.COUNTER
    ):
        """Record a metric for CloudWatch"""
        metric = MetricData(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.utcnow(),
            dimensions=dimensions or {},
            metric_type=metric_type
        )
        
        self.metric_buffer.append(metric)
        
        # Flush if buffer is full
        if len(self.metric_buffer) >= self.buffer_size:
            asyncio.create_task(self._flush_metrics())
    
    def record_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float,
        user_id: Optional[str] = None
    ):
        """Record API call metrics"""
        dimensions = {
            'Endpoint': endpoint,
            'Method': method,
            'StatusCode': str(status_code)
        }
        
        if user_id:
            dimensions['UserId'] = user_id
        
        # Record multiple metrics
        self.record_metric("APICall", 1, "Count", dimensions)
        self.record_metric("APILatency", duration_ms, "Milliseconds", dimensions, MetricType.HISTOGRAM)
        
        if status_code >= 400:
            self.record_metric("APIError", 1, "Count", dimensions)
        
        # Log the API call
        self.log_structured(
            LogLevel.INFO,
            "api",
            "request",
            f"{method} {endpoint} - {status_code}",
            {
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'duration_ms': duration_ms
            },
            user_id=user_id
        )
    
    def record_ml_inference(
        self,
        model_name: str,
        inference_time_ms: float,
        success: bool,
        input_size: Optional[int] = None,
        confidence: Optional[float] = None
    ):
        """Record ML model inference metrics"""
        dimensions = {
            'ModelName': model_name,
            'Success': str(success)
        }
        
        self.record_metric("MLInference", 1, "Count", dimensions)
        self.record_metric("MLInferenceTime", inference_time_ms, "Milliseconds", dimensions, MetricType.HISTOGRAM)
        
        if not success:
            self.record_metric("MLInferenceError", 1, "Count", dimensions)
        
        context = {
            'model_name': model_name,
            'inference_time_ms': inference_time_ms,
            'success': success
        }
        
        if input_size is not None:
            context['input_size'] = input_size
        if confidence is not None:
            context['confidence'] = confidence
        
        self.log_structured(
            LogLevel.INFO,
            "ml",
            "inference",
            f"ML inference for {model_name}: {'success' if success else 'failed'}",
            context
        )
    
    def record_trade_execution(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        success: bool,
        execution_time_ms: float
    ):
        """Record trade execution metrics"""
        dimensions = {
            'Symbol': symbol,
            'Action': action,
            'Success': str(success)
        }
        
        self.record_metric("TradeExecution", 1, "Count", dimensions)
        self.record_metric("TradeExecutionTime", execution_time_ms, "Milliseconds", dimensions, MetricType.HISTOGRAM)
        
        if success:
            self.record_metric("TradeValue", quantity * price, "None", dimensions, MetricType.GAUGE)
        else:
            self.record_metric("TradeExecutionError", 1, "Count", dimensions)
        
        self.log_structured(
            LogLevel.INFO,
            "trading",
            "execution",
            f"Trade execution: {action} {quantity} {symbol} @ ${price}",
            {
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'success': success,
                'execution_time_ms': execution_time_ms
            }
        )
    
    def record_data_pipeline_metrics(
        self,
        stage: str,
        records_processed: int,
        processing_time_ms: float,
        errors: int = 0
    ):
        """Record data pipeline metrics"""
        dimensions = {'Stage': stage}
        
        self.record_metric("DataPipelineRecords", records_processed, "Count", dimensions)
        self.record_metric("DataPipelineTime", processing_time_ms, "Milliseconds", dimensions, MetricType.HISTOGRAM)
        
        if errors > 0:
            self.record_metric("DataPipelineErrors", errors, "Count", dimensions)
        
        self.log_structured(
            LogLevel.INFO,
            "data_pipeline",
            stage,
            f"Processed {records_processed} records in {processing_time_ms:.2f}ms",
            {
                'stage': stage,
                'records_processed': records_processed,
                'processing_time_ms': processing_time_ms,
                'errors': errors
            }
        )
    
    def record_error(
        self,
        service: str,
        component: str,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ):
        """Record error metrics and logs"""
        dimensions = {
            'Service': service,
            'Component': component,
            'ErrorType': error_type
        }
        
        self.record_metric("Error", 1, "Count", dimensions)
        
        self.log_structured(
            LogLevel.ERROR,
            service,
            component,
            f"Error: {error_message}",
            {
                'error_type': error_type,
                'error_message': error_message,
                **(context or {})
            },
            trace_id=trace_id
        )
    
    async def _flush_metrics(self):
        """Flush metric buffer to CloudWatch"""
        if not self.metric_buffer:
            return
        
        try:
            # Group metrics by dimensions for batch sending
            metric_data = []
            
            for metric in self.metric_buffer:
                metric_datum = {
                    'MetricName': metric.name,
                    'Value': metric.value,
                    'Unit': metric.unit,
                    'Timestamp': metric.timestamp
                }
                
                if metric.dimensions:
                    metric_datum['Dimensions'] = [
                        {'Name': k, 'Value': v} for k, v in metric.dimensions.items()
                    ]
                
                metric_data.append(metric_datum)
            
            # Send metrics in batches (CloudWatch limit is 20 per request)
            for i in range(0, len(metric_data), 20):
                batch = metric_data[i:i+20]
                
                self.cloudwatch.put_metric_data(
                    Namespace=self.namespace,
                    MetricData=batch
                )
            
            logger.debug(f"Flushed {len(self.metric_buffer)} metrics to CloudWatch")
            self.metric_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing metrics to CloudWatch: {e}")
    
    async def _flush_logs(self):
        """Flush log buffer to CloudWatch Logs"""
        if not self.log_buffer:
            return
        
        try:
            # Group logs by service for different log groups
            log_groups = {}
            
            for log_entry in self.log_buffer:
                log_group = self.log_groups.get(log_entry.service, self.log_groups['application'])
                
                if log_group not in log_groups:
                    log_groups[log_group] = []
                
                log_event = {
                    'timestamp': int(log_entry.timestamp.timestamp() * 1000),
                    'message': json.dumps(asdict(log_entry))
                }
                
                log_groups[log_group].append(log_event)
            
            # Send logs to each group
            for log_group, events in log_groups.items():
                try:
                    # Sort events by timestamp
                    events.sort(key=lambda x: x['timestamp'])
                    
                    self.logs_client.put_log_events(
                        logGroupName=log_group,
                        logStreamName=f"stream-{datetime.utcnow().strftime('%Y-%m-%d')}",
                        logEvents=events
                    )
                    
                except self.logs_client.exceptions.ResourceNotFoundException:
                    # Create log stream if it doesn't exist
                    try:
                        self.logs_client.create_log_stream(
                            logGroupName=log_group,
                            logStreamName=f"stream-{datetime.utcnow().strftime('%Y-%m-%d')}"
                        )
                        
                        self.logs_client.put_log_events(
                            logGroupName=log_group,
                            logStreamName=f"stream-{datetime.utcnow().strftime('%Y-%m-%d')}",
                            logEvents=events
                        )
                    except Exception as e:
                        logger.error(f"Error creating log stream for {log_group}: {e}")
                
                except Exception as e:
                    logger.error(f"Error sending logs to {log_group}: {e}")
            
            logger.debug(f"Flushed {len(self.log_buffer)} log entries to CloudWatch Logs")
            self.log_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing logs to CloudWatch: {e}")
    
    async def flush_all(self):
        """Flush all buffered metrics and logs"""
        await asyncio.gather(
            self._flush_metrics(),
            self._flush_logs(),
            return_exceptions=True
        )
    
    def get_dashboard_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics for dashboard display"""
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        try:
            # Get key metrics from CloudWatch
            metrics = {}
            
            # API metrics
            api_calls = self.cloudwatch.get_metric_statistics(
                Namespace=self.namespace,
                MetricName='APICall',
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour
                Statistics=['Sum']
            )
            
            metrics['api_calls'] = sum(point['Sum'] for point in api_calls['Datapoints'])
            
            # ML inference metrics
            ml_inferences = self.cloudwatch.get_metric_statistics(
                Namespace=self.namespace,
                MetricName='MLInference',
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Sum']
            )
            
            metrics['ml_inferences'] = sum(point['Sum'] for point in ml_inferences['Datapoints'])
            
            # Trade execution metrics
            trades = self.cloudwatch.get_metric_statistics(
                Namespace=self.namespace,
                MetricName='TradeExecution',
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Sum']
            )
            
            metrics['trades_executed'] = sum(point['Sum'] for point in trades['Datapoints'])
            
            # Error metrics
            errors = self.cloudwatch.get_metric_statistics(
                Namespace=self.namespace,
                MetricName='Error',
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Sum']
            )
            
            metrics['total_errors'] = sum(point['Sum'] for point in errors['Datapoints'])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting dashboard metrics: {e}")
            return {}
    
    def create_performance_timer(self, name: str):
        """Create a performance timer context manager"""
        return PerformanceTimer(self, name)


class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, monitoring_service: MonitoringService, name: str):
        self.monitoring_service = monitoring_service
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            self.monitoring_service.record_metric(
                f"{self.name}Duration",
                duration_ms,
                "Milliseconds",
                metric_type=MetricType.HISTOGRAM
            )


def monitor_performance(metric_name: str):
    """Decorator for monitoring function performance"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                monitoring_service.record_metric(
                    f"{metric_name}Duration",
                    duration_ms,
                    "Milliseconds",
                    metric_type=MetricType.HISTOGRAM
                )
                
                return result
            except Exception as e:
                monitoring_service.record_error(
                    "application",
                    func.__name__,
                    type(e).__name__,
                    str(e)
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                monitoring_service.record_metric(
                    f"{metric_name}Duration",
                    duration_ms,
                    "Milliseconds",
                    metric_type=MetricType.HISTOGRAM
                )
                
                return result
            except Exception as e:
                monitoring_service.record_error(
                    "application",
                    func.__name__,
                    type(e).__name__,
                    str(e)
                )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Global monitoring service instance
monitoring_service = MonitoringService()