"""
API endpoints for portfolio simulation and benchmark comparison.
Provides Monte Carlo simulation and benchmark analysis functionality.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

from ..services.portfolio_simulation import (
    MonteCarloSimulator, 
    SimulationParameters, 
    SimulationAnalyzer
)
from ..services.benchmark_comparison import (
    BenchmarkComparator, 
    BenchmarkType, 
    BenchmarkReportGenerator
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/simulation", tags=["Portfolio Simulation"])

# Request/Response Models
class SimulationRequest(BaseModel):
    """Request model for Monte Carlo simulation."""
    initial_capital: float = Field(default=100000.0, gt=0, description="Initial portfolio capital")
    num_iterations: int = Field(default=1000, ge=10, le=10000, description="Number of simulation iterations")
    simulation_days: int = Field(default=252, ge=30, le=1000, description="Number of trading days to simulate")
    risk_free_rate: float = Field(default=0.02, ge=0, le=0.1, description="Annual risk-free rate")
    max_position_size: float = Field(default=0.05, gt=0, le=0.5, description="Maximum position size as fraction of portfolio")
    stop_loss_threshold: float = Field(default=0.02, gt=0, le=0.2, description="Stop-loss threshold")
    confidence_threshold: float = Field(default=0.7, ge=0, le=1, description="Minimum confidence for trades")
    rebalance_frequency: int = Field(default=5, ge=1, le=30, description="Rebalance frequency in days")
    start_date: Optional[str] = Field(None, description="Start date for historical data (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date for historical data (YYYY-MM-DD)")

class BenchmarkComparisonRequest(BaseModel):
    """Request model for benchmark comparison."""
    benchmark_types: List[str] = Field(default=["SPY", "QQQ"], description="List of benchmark symbols")
    custom_benchmarks: Optional[List[str]] = Field(None, description="Custom benchmark symbols")
    start_date: Optional[str] = Field(None, description="Start date for comparison (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date for comparison (YYYY-MM-DD)")

class SimulationResponse(BaseModel):
    """Response model for simulation results."""
    simulation_id: str
    parameters: Dict[str, Any]
    results_summary: Dict[str, Any]
    benchmark_comparison: Optional[Dict[str, Any]] = None
    report: Optional[str] = None
    status: str
    created_at: str

# Global storage for simulation results (in production, use Redis or database)
simulation_cache: Dict[str, Any] = {}

@router.post("/run", response_model=SimulationResponse)
async def run_monte_carlo_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    include_benchmarks: bool = Query(default=True, description="Include benchmark comparison")
):
    """
    Run Monte Carlo portfolio simulation.
    
    Args:
        request: Simulation parameters
        background_tasks: FastAPI background tasks
        include_benchmarks: Whether to include benchmark comparison
        
    Returns:
        Simulation response with results summary
    """
    try:
        # Generate simulation ID
        simulation_id = f"sim_{int(datetime.now().timestamp())}"
        
        # Parse dates
        start_date = None
        end_date = None
        if request.start_date:
            start_date = datetime.strptime(request.start_date, '%Y-%m-%d')
        if request.end_date:
            end_date = datetime.strptime(request.end_date, '%Y-%m-%d')
        
        # Create simulation parameters
        parameters = SimulationParameters(
            initial_capital=request.initial_capital,
            num_iterations=request.num_iterations,
            simulation_days=request.simulation_days,
            risk_free_rate=request.risk_free_rate,
            max_position_size=request.max_position_size,
            stop_loss_threshold=request.stop_loss_threshold,
            confidence_threshold=request.confidence_threshold,
            rebalance_frequency=request.rebalance_frequency
        )
        
        # Store initial status
        simulation_cache[simulation_id] = {
            'status': 'running',
            'created_at': datetime.now().isoformat(),
            'parameters': parameters.__dict__,
            'results': None,
            'benchmark_comparison': None
        }
        
        # Run simulation in background
        background_tasks.add_task(
            _run_simulation_background,
            simulation_id,
            parameters,
            start_date,
            end_date,
            include_benchmarks
        )
        
        return SimulationResponse(
            simulation_id=simulation_id,
            parameters=parameters.__dict__,
            results_summary={'message': 'Simulation started'},
            status='running',
            created_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start simulation: {str(e)}")

@router.get("/status/{simulation_id}", response_model=SimulationResponse)
async def get_simulation_status(simulation_id: str):
    """
    Get status of a running or completed simulation.
    
    Args:
        simulation_id: Unique simulation identifier
        
    Returns:
        Current simulation status and results
    """
    if simulation_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim_data = simulation_cache[simulation_id]
    
    return SimulationResponse(
        simulation_id=simulation_id,
        parameters=sim_data['parameters'],
        results_summary=sim_data.get('results', {'message': 'Simulation in progress'}),
        benchmark_comparison=sim_data.get('benchmark_comparison'),
        report=sim_data.get('report'),
        status=sim_data['status'],
        created_at=sim_data['created_at']
    )

@router.post("/benchmark-comparison")
async def run_benchmark_comparison(
    request: BenchmarkComparisonRequest,
    simulation_id: Optional[str] = Query(None, description="Simulation ID to compare")
):
    """
    Run benchmark comparison analysis.
    
    Args:
        request: Benchmark comparison parameters
        simulation_id: Optional simulation ID to use existing results
        
    Returns:
        Benchmark comparison results
    """
    try:
        # Parse dates
        start_date = datetime.now() - timedelta(days=365)  # Default to 1 year
        end_date = datetime.now()
        
        if request.start_date:
            start_date = datetime.strptime(request.start_date, '%Y-%m-%d')
        if request.end_date:
            end_date = datetime.strptime(request.end_date, '%Y-%m-%d')
        
        # Convert benchmark types
        benchmark_types = []
        for bt in request.benchmark_types:
            try:
                benchmark_types.append(BenchmarkType(bt))
            except ValueError:
                logger.warning(f"Unknown benchmark type: {bt}")
        
        if not benchmark_types:
            raise HTTPException(status_code=400, detail="No valid benchmark types provided")
        
        # Initialize comparator
        comparator = BenchmarkComparator()
        
        # If simulation_id provided, use those results
        if simulation_id and simulation_id in simulation_cache:
            sim_data = simulation_cache[simulation_id]
            if sim_data['status'] == 'completed' and sim_data.get('results'):
                # Use simulation results for comparison
                # This would require extracting simulation results properly
                # For now, return a placeholder
                return {
                    "message": "Benchmark comparison with simulation results",
                    "simulation_id": simulation_id,
                    "benchmarks": [bt.value for bt in benchmark_types],
                    "status": "completed"
                }
        
        # Run standalone benchmark comparison
        # This would typically compare against historical portfolio performance
        return {
            "message": "Standalone benchmark comparison",
            "benchmarks": [bt.value for bt in benchmark_types],
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "status": "completed"
        }
        
    except Exception as e:
        logger.error(f"Error running benchmark comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to run benchmark comparison: {str(e)}")

@router.get("/benchmarks")
async def get_available_benchmarks():
    """
    Get list of available benchmark types.
    
    Returns:
        List of available benchmarks with descriptions
    """
    benchmarks = []
    
    for benchmark_type in BenchmarkType:
        if benchmark_type != BenchmarkType.CUSTOM:
            benchmarks.append({
                'symbol': benchmark_type.value,
                'name': BenchmarkDataProvider.BENCHMARK_INFO[benchmark_type]['name'],
                'description': BenchmarkDataProvider.BENCHMARK_INFO[benchmark_type]['description']
            })
    
    benchmarks.append({
        'symbol': 'CUSTOM',
        'name': 'Custom Benchmark',
        'description': 'User-defined benchmark symbol'
    })
    
    return {'benchmarks': benchmarks}

@router.delete("/results/{simulation_id}")
async def delete_simulation_results(simulation_id: str):
    """
    Delete simulation results from cache.
    
    Args:
        simulation_id: Simulation ID to delete
        
    Returns:
        Deletion confirmation
    """
    if simulation_id not in simulation_cache:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    del simulation_cache[simulation_id]
    
    return {"message": f"Simulation {simulation_id} deleted successfully"}

@router.get("/results")
async def list_simulation_results():
    """
    List all cached simulation results.
    
    Returns:
        List of simulation summaries
    """
    results = []
    
    for sim_id, sim_data in simulation_cache.items():
        results.append({
            'simulation_id': sim_id,
            'status': sim_data['status'],
            'created_at': sim_data['created_at'],
            'parameters': {
                'num_iterations': sim_data['parameters'].get('num_iterations'),
                'simulation_days': sim_data['parameters'].get('simulation_days'),
                'initial_capital': sim_data['parameters'].get('initial_capital')
            }
        })
    
    return {'simulations': results}

# Background task functions
async def _run_simulation_background(
    simulation_id: str,
    parameters: SimulationParameters,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    include_benchmarks: bool
):
    """
    Run simulation in background task.
    
    Args:
        simulation_id: Unique simulation identifier
        parameters: Simulation parameters
        start_date: Start date for historical data
        end_date: End date for historical data
        include_benchmarks: Whether to include benchmark comparison
    """
    try:
        # Initialize simulator
        simulator = MonteCarloSimulator()
        
        # Run simulation
        logger.info(f"Starting simulation {simulation_id}")
        results = await simulator.run_simulation(parameters, start_date, end_date)
        
        if not results:
            simulation_cache[simulation_id]['status'] = 'failed'
            simulation_cache[simulation_id]['results'] = {'error': 'No simulation results generated'}
            return
        
        # Analyze results
        analysis = SimulationAnalyzer.analyze_results(results)
        
        # Update cache with results
        simulation_cache[simulation_id]['results'] = analysis
        simulation_cache[simulation_id]['status'] = 'completed'
        
        # Run benchmark comparison if requested
        if include_benchmarks and results:
            try:
                comparator = BenchmarkComparator()
                benchmark_types = [BenchmarkType.SPY, BenchmarkType.QQQ]
                
                comparison_results = await comparator.compare_simulation_results_to_benchmarks(
                    results, benchmark_types, start_date or datetime.now() - timedelta(days=365), 
                    end_date or datetime.now()
                )
                
                # Generate report
                report = BenchmarkReportGenerator.generate_comparison_report(
                    comparison_results, analysis
                )
                
                simulation_cache[simulation_id]['benchmark_comparison'] = comparison_results
                simulation_cache[simulation_id]['report'] = report
                
            except Exception as e:
                logger.error(f"Error in benchmark comparison for {simulation_id}: {e}")
                simulation_cache[simulation_id]['benchmark_comparison'] = {'error': str(e)}
        
        logger.info(f"Simulation {simulation_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in background simulation {simulation_id}: {e}")
        simulation_cache[simulation_id]['status'] = 'failed'
        simulation_cache[simulation_id]['results'] = {'error': str(e)}

# Import the BenchmarkDataProvider for the endpoint
from ..services.benchmark_comparison import BenchmarkDataProvider