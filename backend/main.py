from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import logging
import time
from config import settings
from services.monitoring_service import monitoring_service
from startup import startup_sequence, shutdown_sequence, get_startup_info
from api.explanations import router as explanations_router
from api.sentiment import router as sentiment_router
from api.predictions import router as predictions_router
from api.trades import router as trades_router
from api.portfolio import router as portfolio_router
from api.portfolio_simulation import router as simulation_router
from api.integration import router as integration_router
from api.monitoring import router as monitoring_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    debug=settings.debug
)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request logging and monitoring middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Log to standard logger
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.4f}s"
    )
    
    # Record metrics and structured logs
    monitoring_service.record_api_call(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        duration_ms=process_time * 1000
    )
    
    return response

# Include API routers
app.include_router(sentiment_router)
app.include_router(predictions_router)
app.include_router(trades_router)
app.include_router(portfolio_router)
app.include_router(explanations_router)
app.include_router(simulation_router)
app.include_router(integration_router)
app.include_router(monitoring_router)

@app.get("/")
async def root():
    return {
        "message": "Sentiment Trading Agent API",
        "version": settings.api_version,
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/startup-info")
async def startup_info():
    return get_startup_info()

# Application lifecycle events
@app.on_event("startup")
async def on_startup():
    """Application startup event handler"""
    result = await startup_sequence()
    if result["status"] != "success":
        logger.error(f"Startup failed: {result}")
        # Don't exit here to allow for debugging

@app.on_event("shutdown")
async def on_shutdown():
    """Application shutdown event handler"""
    await shutdown_sequence()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)