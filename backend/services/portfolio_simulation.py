"""
Portfolio simulation service for Monte Carlo analysis and backtesting.
Implements simulation framework with performance metrics calculation.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import partial

from ..models.data_models import Trade, PortfolioPosition, PricePrediction, SentimentScore
from ..repositories.trades_repository import TradesRepository
from ..repositories.portfolio_repository import PortfolioRepository
from ..repositories.predictions_repository import PredictionsRepository
from ..repositories.sentiment_repository import SentimentRepository

logger = logging.getLogger(__name__)

@dataclass
class SimulationParameters:
    """Parameters for Monte Carlo simulation."""
    initial_capital: float = 100000.0
    num_iterations: int = 1000
    simulation_days: int = 252  # Trading days in a year
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    max_position_size: float = 0.05  # 5% max allocation per position
    stop_loss_threshold: float = 0.02  # 2% stop loss
    confidence_threshold: float = 0.7  # Minimum confidence for trades
    rebalance_frequency: int = 5  # Rebalance every 5 days

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional Value at Risk
    final_portfolio_value: float
    num_trades: int

@dataclass
class SimulationResult:
    """Result of a single Monte Carlo simulation."""
    iteration: int
    performance_metrics: PerformanceMetrics
    portfolio_values: List[float]
    trades_executed: List[Dict[str, Any]]
    daily_returns: List[float]

class MonteCarloSimulator:
    """Monte Carlo simulation engine for portfolio performance analysis."""
    
    def __init__(self):
        self.trades_repo = TradesRepository()
        self.portfolio_repo = PortfolioRepository()
        self.predictions_repo = PredictionsRepository()
        self.sentiment_repo = SentimentRepository()
    
    async def run_simulation(
        self, 
        parameters: SimulationParameters,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[SimulationResult]:
        """
        Run Monte Carlo simulation with specified parameters.
        
        Args:
            parameters: Simulation parameters
            start_date: Start date for historical data (optional)
            end_date: End date for historical data (optional)
            
        Returns:
            List of simulation results
        """
        logger.info(f"Starting Monte Carlo simulation with {parameters.num_iterations} iterations")
        
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=parameters.simulation_days)
        
        # Get historical data for simulation
        historical_data = await self._prepare_historical_data(start_date, end_date)
        
        if not historical_data:
            logger.error("No historical data available for simulation")
            return []
        
        # Run simulations in parallel
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Create partial function with fixed parameters
            simulate_func = partial(
                self._run_single_simulation,
                parameters=parameters,
                historical_data=historical_data,
                start_date=start_date,
                end_date=end_date
            )
            
            # Submit all simulation tasks
            future_to_iteration = {
                executor.submit(simulate_func, i): i 
                for i in range(parameters.num_iterations)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_iteration):
                iteration = future_to_iteration[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        if len(results) % 100 == 0:
                            logger.info(f"Completed {len(results)} simulations")
                except Exception as e:
                    logger.error(f"Simulation {iteration} failed: {e}")
        
        logger.info(f"Completed {len(results)} successful simulations")
        return results
    
    def _run_single_simulation(
        self,
        iteration: int,
        parameters: SimulationParameters,
        historical_data: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[SimulationResult]:
        """
        Run a single Monte Carlo simulation iteration.
        
        Args:
            iteration: Simulation iteration number
            parameters: Simulation parameters
            historical_data: Historical market data
            start_date: Simulation start date
            end_date: Simulation end date
            
        Returns:
            Simulation result or None if failed
        """
        try:
            # Initialize simulation state
            portfolio_value = parameters.initial_capital
            positions = {}  # symbol -> quantity
            portfolio_values = [portfolio_value]
            trades_executed = []
            daily_returns = []
            
            # Generate random price paths using historical volatility
            price_paths = self._generate_price_paths(
                historical_data, 
                parameters.simulation_days
            )
            
            # Simulate trading over the specified period
            for day in range(parameters.simulation_days):
                # Calculate current portfolio value
                current_value = self._calculate_portfolio_value(
                    positions, price_paths, day
                )
                
                # Calculate daily return
                if portfolio_values:
                    daily_return = (current_value - portfolio_values[-1]) / portfolio_values[-1]
                    daily_returns.append(daily_return)
                
                portfolio_values.append(current_value)
                
                # Generate trading signals and execute trades
                if day % parameters.rebalance_frequency == 0:
                    new_trades = self._generate_trades(
                        historical_data, price_paths, day, 
                        current_value, positions, parameters
                    )
                    trades_executed.extend(new_trades)
                    
                    # Update positions based on trades
                    for trade in new_trades:
                        symbol = trade['symbol']
                        quantity = trade['quantity'] if trade['action'] == 'buy' else -trade['quantity']
                        positions[symbol] = positions.get(symbol, 0) + quantity
                
                # Apply stop-loss rules
                stop_loss_trades = self._apply_stop_loss(
                    positions, price_paths, day, parameters
                )
                trades_executed.extend(stop_loss_trades)
                
                # Update positions after stop-loss
                for trade in stop_loss_trades:
                    symbol = trade['symbol']
                    positions[symbol] = 0  # Stop-loss closes position
            
            # Calculate performance metrics
            final_value = portfolio_values[-1]
            performance_metrics = self._calculate_performance_metrics(
                portfolio_values, daily_returns, trades_executed, parameters
            )
            
            return SimulationResult(
                iteration=iteration,
                performance_metrics=performance_metrics,
                portfolio_values=portfolio_values,
                trades_executed=trades_executed,
                daily_returns=daily_returns
            )
            
        except Exception as e:
            logger.error(f"Error in simulation iteration {iteration}: {e}")
            return None
    
    async def _prepare_historical_data(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Prepare historical data for simulation.
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            Dictionary containing historical data
        """
        try:
            # Get historical trades
            days_back = (end_date - start_date).days
            historical_trades = await self.trades_repo.get_recent_trades(days_back)
            
            # Get historical predictions
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            historical_predictions = await self.predictions_repo.get_predictions_by_timerange(
                start_timestamp, end_timestamp
            )
            
            # Get historical sentiment data
            historical_sentiment = await self.sentiment_repo.get_sentiment_by_timerange(
                start_timestamp, end_timestamp
            )
            
            # Extract unique symbols
            symbols = set()
            for trade in historical_trades:
                symbols.add(trade.symbol)
            for pred in historical_predictions:
                symbols.add(pred.symbol)
            for sent in historical_sentiment:
                symbols.add(sent.symbol)
            
            # Calculate historical volatilities and returns
            symbol_stats = {}
            for symbol in symbols:
                symbol_trades = [t for t in historical_trades if t.symbol == symbol]
                if len(symbol_trades) >= 10:  # Minimum trades for statistics
                    prices = [t.price for t in symbol_trades]
                    returns = np.diff(np.log(prices))
                    symbol_stats[symbol] = {
                        'volatility': np.std(returns) * np.sqrt(252),  # Annualized
                        'mean_return': np.mean(returns) * 252,  # Annualized
                        'last_price': prices[-1]
                    }
            
            return {
                'trades': historical_trades,
                'predictions': historical_predictions,
                'sentiment': historical_sentiment,
                'symbols': list(symbols),
                'symbol_stats': symbol_stats,
                'start_date': start_date,
                'end_date': end_date
            }
            
        except Exception as e:
            logger.error(f"Error preparing historical data: {e}")
            return {}
    
    def _generate_price_paths(
        self, 
        historical_data: Dict[str, Any], 
        num_days: int
    ) -> Dict[str, List[float]]:
        """
        Generate random price paths using geometric Brownian motion.
        
        Args:
            historical_data: Historical market data
            num_days: Number of days to simulate
            
        Returns:
            Dictionary mapping symbols to price paths
        """
        price_paths = {}
        
        for symbol, stats in historical_data['symbol_stats'].items():
            # Parameters for geometric Brownian motion
            S0 = stats['last_price']  # Initial price
            mu = stats['mean_return'] / 252  # Daily drift
            sigma = stats['volatility'] / np.sqrt(252)  # Daily volatility
            
            # Generate random price path
            dt = 1.0  # Daily time step
            prices = [S0]
            
            for _ in range(num_days):
                # Geometric Brownian motion: dS = S * (mu * dt + sigma * dW)
                dW = np.random.normal(0, np.sqrt(dt))
                S_new = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
                prices.append(max(S_new, 0.01))  # Prevent negative prices
            
            price_paths[symbol] = prices
        
        return price_paths
    
    def _calculate_portfolio_value(
        self, 
        positions: Dict[str, int], 
        price_paths: Dict[str, List[float]], 
        day: int
    ) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            positions: Current positions (symbol -> quantity)
            price_paths: Price paths for all symbols
            day: Current day index
            
        Returns:
            Current portfolio value
        """
        total_value = 0.0
        
        for symbol, quantity in positions.items():
            if symbol in price_paths and day < len(price_paths[symbol]):
                current_price = price_paths[symbol][day]
                total_value += quantity * current_price
        
        return total_value
    
    def _generate_trades(
        self,
        historical_data: Dict[str, Any],
        price_paths: Dict[str, List[float]],
        day: int,
        portfolio_value: float,
        positions: Dict[str, int],
        parameters: SimulationParameters
    ) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on simulated market conditions.
        
        Args:
            historical_data: Historical market data
            price_paths: Simulated price paths
            day: Current day
            portfolio_value: Current portfolio value
            positions: Current positions
            parameters: Simulation parameters
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        
        # Simple momentum-based trading strategy for simulation
        for symbol in historical_data['symbols'][:10]:  # Limit to top 10 symbols
            if symbol not in price_paths or day < 5:
                continue
            
            # Calculate short-term momentum
            recent_prices = price_paths[symbol][max(0, day-5):day+1]
            if len(recent_prices) < 2:
                continue
            
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # Generate trading signal with some randomness
            signal_strength = abs(momentum) + np.random.normal(0, 0.1)
            signal_strength = max(0, min(1, signal_strength))  # Clamp to [0, 1]
            
            # Only trade if signal is strong enough
            if signal_strength < parameters.confidence_threshold:
                continue
            
            # Determine trade action
            action = 'buy' if momentum > 0 else 'sell'
            
            # Calculate position size (limited by max allocation)
            max_position_value = portfolio_value * parameters.max_position_size
            current_price = price_paths[symbol][day]
            max_quantity = int(max_position_value / current_price)
            
            # Adjust for existing positions
            current_position = positions.get(symbol, 0)
            if action == 'buy' and current_position >= max_quantity:
                continue  # Already at max position
            if action == 'sell' and current_position <= 0:
                continue  # No position to sell
            
            # Calculate trade quantity
            if action == 'buy':
                quantity = min(max_quantity - current_position, max_quantity // 2)
            else:
                quantity = min(abs(current_position), abs(current_position) // 2)
            
            if quantity > 0:
                trades.append({
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': current_price,
                    'signal_strength': signal_strength,
                    'day': day
                })
        
        return trades
    
    def _apply_stop_loss(
        self,
        positions: Dict[str, int],
        price_paths: Dict[str, List[float]],
        day: int,
        parameters: SimulationParameters
    ) -> List[Dict[str, Any]]:
        """
        Apply stop-loss rules to current positions.
        
        Args:
            positions: Current positions
            price_paths: Price paths
            day: Current day
            parameters: Simulation parameters
            
        Returns:
            List of stop-loss trades
        """
        stop_loss_trades = []
        
        for symbol, quantity in positions.items():
            if quantity == 0 or symbol not in price_paths:
                continue
            
            if day >= len(price_paths[symbol]):
                continue
            
            current_price = price_paths[symbol][day]
            
            # Simple stop-loss: if price drops more than threshold, close position
            # This is a simplified implementation - in reality, we'd track entry prices
            if day >= 5:  # Need some history
                entry_price = price_paths[symbol][max(0, day-5)]  # Approximate entry
                price_change = (current_price - entry_price) / entry_price
                
                # Apply stop-loss for long positions
                if quantity > 0 and price_change < -parameters.stop_loss_threshold:
                    stop_loss_trades.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': quantity,
                        'price': current_price,
                        'signal_strength': 1.0,  # Stop-loss is always executed
                        'day': day,
                        'reason': 'stop_loss'
                    })
        
        return stop_loss_trades
    
    def _calculate_performance_metrics(
        self,
        portfolio_values: List[float],
        daily_returns: List[float],
        trades: List[Dict[str, Any]],
        parameters: SimulationParameters
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_values: Daily portfolio values
            daily_returns: Daily returns
            trades: Executed trades
            parameters: Simulation parameters
            
        Returns:
            Performance metrics
        """
        if not portfolio_values or len(portfolio_values) < 2:
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, calmar_ratio=0.0,
                win_rate=0.0, profit_factor=0.0, var_95=0.0, cvar_95=0.0,
                final_portfolio_value=parameters.initial_capital, num_trades=0
            )
        
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        
        # Total and annualized returns
        total_return = (final_value - initial_value) / initial_value
        days = len(portfolio_values) - 1
        annualized_return = (final_value / initial_value) ** (252 / days) - 1
        
        # Volatility (annualized)
        if daily_returns:
            volatility = np.std(daily_returns) * np.sqrt(252)
        else:
            volatility = 0.0
        
        # Sharpe ratio
        excess_return = annualized_return - parameters.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
        
        # Maximum drawdown
        peak = portfolio_values[0]
        max_drawdown = 0.0
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        
        # Trade-based metrics
        winning_trades = 0
        losing_trades = 0
        total_profit = 0.0
        total_loss = 0.0
        
        # Simplified P&L calculation (would need more sophisticated tracking in reality)
        for trade in trades:
            # Simulate random P&L for demonstration
            pnl = np.random.normal(0, trade['price'] * 0.02)  # 2% volatility
            if pnl > 0:
                winning_trades += 1
                total_profit += pnl
            else:
                losing_trades += 1
                total_loss += abs(pnl)
        
        total_trades = winning_trades + losing_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        # Value at Risk (95% confidence)
        if daily_returns:
            var_95 = np.percentile(daily_returns, 5) * initial_value
            cvar_95 = np.mean([r for r in daily_returns if r <= np.percentile(daily_returns, 5)]) * initial_value
        else:
            var_95 = 0.0
            cvar_95 = 0.0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            var_95=var_95,
            cvar_95=cvar_95,
            final_portfolio_value=final_value,
            num_trades=len(trades)
        )

class SimulationAnalyzer:
    """Analyzer for Monte Carlo simulation results."""
    
    @staticmethod
    def analyze_results(results: List[SimulationResult]) -> Dict[str, Any]:
        """
        Analyze Monte Carlo simulation results.
        
        Args:
            results: List of simulation results
            
        Returns:
            Dictionary with aggregated analysis
        """
        if not results:
            return {}
        
        # Extract metrics from all simulations
        returns = [r.performance_metrics.total_return for r in results]
        sharpe_ratios = [r.performance_metrics.sharpe_ratio for r in results]
        max_drawdowns = [r.performance_metrics.max_drawdown for r in results]
        final_values = [r.performance_metrics.final_portfolio_value for r in results]
        
        # Calculate statistics
        analysis = {
            'num_simulations': len(results),
            'returns': {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'min': np.min(returns),
                'max': np.max(returns),
                'percentiles': {
                    '5th': np.percentile(returns, 5),
                    '25th': np.percentile(returns, 25),
                    '50th': np.percentile(returns, 50),
                    '75th': np.percentile(returns, 75),
                    '95th': np.percentile(returns, 95)
                }
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'min': np.min(sharpe_ratios),
                'max': np.max(sharpe_ratios)
            },
            'max_drawdown': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'worst': np.max(max_drawdowns)
            },
            'final_portfolio_value': {
                'mean': np.mean(final_values),
                'std': np.std(final_values),
                'min': np.min(final_values),
                'max': np.max(final_values)
            },
            'probability_of_profit': sum(1 for r in returns if r > 0) / len(returns),
            'probability_of_loss': sum(1 for r in returns if r < 0) / len(returns)
        }
        
        return analysis
    
    @staticmethod
    def generate_summary_report(analysis: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary report.
        
        Args:
            analysis: Analysis results
            
        Returns:
            Formatted summary report
        """
        if not analysis:
            return "No simulation results to analyze."
        
        report = f"""
Monte Carlo Simulation Analysis Report
=====================================

Simulation Overview:
- Number of simulations: {analysis['num_simulations']:,}
- Probability of profit: {analysis['probability_of_profit']:.1%}
- Probability of loss: {analysis['probability_of_loss']:.1%}

Return Statistics:
- Mean return: {analysis['returns']['mean']:.2%}
- Standard deviation: {analysis['returns']['std']:.2%}
- Best case (95th percentile): {analysis['returns']['percentiles']['95th']:.2%}
- Worst case (5th percentile): {analysis['returns']['percentiles']['5th']:.2%}
- Median return: {analysis['returns']['percentiles']['50th']:.2%}

Risk Metrics:
- Average Sharpe ratio: {analysis['sharpe_ratio']['mean']:.2f}
- Average max drawdown: {analysis['max_drawdown']['mean']:.2%}
- Worst drawdown: {analysis['max_drawdown']['worst']:.2%}

Portfolio Value:
- Mean final value: ${analysis['final_portfolio_value']['mean']:,.2f}
- Range: ${analysis['final_portfolio_value']['min']:,.2f} - ${analysis['final_portfolio_value']['max']:,.2f}
"""
        
        return report.strip()