"""
Application startup script for initializing monitoring, logging, and system components.
"""

import asyncio
import logging
import sys
from datetime import datetime
from typing import Dict, Any

from services.monitoring_service import monitoring_service, LogLevel
from services.integration_orchestrator import get_orchestrator
from config import settings

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)


async def initialize_monitoring():
    """Initialize monitoring service and CloudWatch integration"""
    try:
        logger.info("Initializing monitoring service...")
        
        # Record startup metric
        monitoring_service.record_metric("ApplicationStartup", 1, "Count")
        
        # Log structured startup event
        monitoring_service.log_structured(
            LogLevel.INFO,
            "application",
            "startup",
            "Application starting up",
            {
                "version": settings.api_version,
                "environment": "production" if not settings.debug else "development",
                "aws_region": settings.aws_region
            }
        )
        
        # Flush initial logs and metrics
        await monitoring_service.flush_all()
        
        logger.info("Monitoring service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize monitoring service: {e}")
        raise


async def initialize_system_components():
    """Initialize and health check all system components"""
    try:
        logger.info("Initializing system components...")
        
        # Initialize orchestrator
        orchestrator = await get_orchestrator()
        
        # Perform system health check
        health_status = await orchestrator.get_system_health()
        
        if health_status['overall_status'] == 'healthy':
            logger.info("All system components initialized successfully")
            
            monitoring_service.log_structured(
                LogLevel.INFO,
                "application",
                "startup",
                "System components initialized",
                {
                    "overall_status": health_status['overall_status'],
                    "components_count": len(health_status.get('services', {}))
                }
            )
        else:
            logger.warning(f"System health check shows degraded status: {health_status['overall_status']}")
            
            monitoring_service.log_structured(
                LogLevel.WARNING,
                "application",
                "startup",
                "System components partially initialized",
                {
                    "overall_status": health_status['overall_status'],
                    "health_details": health_status
                }
            )
        
        return health_status
        
    except Exception as e:
        logger.error(f"Failed to initialize system components: {e}")
        
        monitoring_service.record_error(
            "application",
            "startup",
            "InitializationError",
            str(e)
        )
        
        raise


async def setup_periodic_tasks():
    """Setup periodic background tasks for monitoring and maintenance"""
    try:
        logger.info("Setting up periodic tasks...")
        
        # Schedule periodic metric flushing
        async def periodic_flush():
            while True:
                try:
                    await asyncio.sleep(monitoring_service.flush_interval)
                    await monitoring_service.flush_all()
                except Exception as e:
                    logger.error(f"Error in periodic flush: {e}")
        
        # Schedule periodic health checks
        async def periodic_health_check():
            while True:
                try:
                    await asyncio.sleep(300)  # Every 5 minutes
                    orchestrator = await get_orchestrator()
                    health_status = await orchestrator.get_system_health()
                    
                    monitoring_service.record_metric(
                        "SystemHealthCheck",
                        1 if health_status['overall_status'] == 'healthy' else 0,
                        "Count"
                    )
                    
                except Exception as e:
                    logger.error(f"Error in periodic health check: {e}")
                    monitoring_service.record_error(
                        "application",
                        "health_check",
                        "PeriodicHealthCheckError",
                        str(e)
                    )
        
        # Start background tasks
        asyncio.create_task(periodic_flush())
        asyncio.create_task(periodic_health_check())
        
        logger.info("Periodic tasks started successfully")
        
    except Exception as e:
        logger.error(f"Failed to setup periodic tasks: {e}")
        raise


async def startup_sequence():
    """Complete application startup sequence"""
    startup_start = datetime.utcnow()
    
    try:
        logger.info("=== Starting Sentiment Trading Agent ===")
        
        # Step 1: Initialize monitoring
        await initialize_monitoring()
        
        # Step 2: Initialize system components
        health_status = await initialize_system_components()
        
        # Step 3: Setup periodic tasks
        await setup_periodic_tasks()
        
        startup_duration = (datetime.utcnow() - startup_start).total_seconds()
        
        logger.info(f"=== Startup completed successfully in {startup_duration:.2f} seconds ===")
        
        # Record successful startup
        monitoring_service.record_metric("StartupDuration", startup_duration * 1000, "Milliseconds")
        monitoring_service.record_metric("StartupSuccess", 1, "Count")
        
        monitoring_service.log_structured(
            LogLevel.INFO,
            "application",
            "startup",
            "Application startup completed",
            {
                "startup_duration_seconds": startup_duration,
                "system_health": health_status['overall_status'],
                "timestamp": startup_start.isoformat()
            }
        )
        
        return {
            "status": "success",
            "startup_duration_seconds": startup_duration,
            "system_health": health_status,
            "timestamp": startup_start.isoformat()
        }
        
    except Exception as e:
        startup_duration = (datetime.utcnow() - startup_start).total_seconds()
        
        logger.error(f"=== Startup failed after {startup_duration:.2f} seconds: {e} ===")
        
        # Record failed startup
        monitoring_service.record_metric("StartupFailure", 1, "Count")
        monitoring_service.record_error(
            "application",
            "startup",
            "StartupError",
            str(e),
            {"startup_duration_seconds": startup_duration}
        )
        
        # Try to flush error logs
        try:
            await monitoring_service.flush_all()
        except:
            pass  # Don't fail on flush error during startup failure
        
        return {
            "status": "failed",
            "error": str(e),
            "startup_duration_seconds": startup_duration,
            "timestamp": startup_start.isoformat()
        }


async def shutdown_sequence():
    """Application shutdown sequence"""
    try:
        logger.info("=== Starting application shutdown ===")
        
        # Record shutdown event
        monitoring_service.log_structured(
            LogLevel.INFO,
            "application",
            "shutdown",
            "Application shutdown initiated",
            {"timestamp": datetime.utcnow().isoformat()}
        )
        
        # Flush all remaining logs and metrics
        await monitoring_service.flush_all()
        
        logger.info("=== Application shutdown completed ===")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def get_startup_info() -> Dict[str, Any]:
    """Get information about the startup configuration"""
    return {
        "api_title": settings.api_title,
        "api_version": settings.api_version,
        "debug_mode": settings.debug,
        "aws_region": settings.aws_region,
        "monitoring_namespace": monitoring_service.namespace,
        "log_groups": list(monitoring_service.log_groups.keys()),
        "startup_timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    # Run startup sequence for testing
    result = asyncio.run(startup_sequence())
    print(f"Startup result: {result}")