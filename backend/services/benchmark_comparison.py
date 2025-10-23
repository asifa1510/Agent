"""
Benchmark comparison system for portfolio performance analysis.
Implements benchmark index data integration and statistical significance testing.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
from scipy import stats
import yfinance as yf

from .portfolio_simulation import SimulationResult, PerformanceMetrics
from ..models.data_models import Trade, PortfolioPosition
from ..repositories.trades_repository import TradesRepository
from ..repositories.portfolio_repository import PortfolioRepository

logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """Supported benchmark types."""
    SPY = "SPY"  # S&P 500 ETF
    QQQ = "QQQ"  # NASDAQ-100 ETF
    IWM = "IWM"  # Russell 2000 ETF
    VTI = "VTI"  # Total Stock Market ETF
    CUSTOM = "CUSTOM"  # Custom benchmark

@dataclass
class BenchmarkData:
    """Benchmark performance data."""
    symbol: str
    name: str
    start_date: datetime
    end_date: datetime
    prices: List[float]
    returns: List[float]
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    timestamps: List[int]

@dataclass
class ComparisonResult:
    """Result of portfolio vs benchmark comparison."""
    portfolio_metrics: PerformanceMetrics
    benchmark_data: BenchmarkData
    alpha: float  # Excess return over benchmark
    beta: float   # Portfolio sensitivity to benchmark
    correlation: float
    tracking_error: float
    information_ratio: float
    up_capture: float    # Upside capture ratio
    down_capture: float  # Downside capture ratio
    statistical_significance: Dict[str, Any]

class BenchmarkDataProvider:
    """Provider for benchmark market data."""
    
    BENCHMARK_INFO = {
        BenchmarkType.SPY: {
            'name': 'SPDR S&P 500 ETF',
            'description': 'Tracks the S&P 500 Index'
        },
        BenchmarkType.QQQ: {
            'name': 'Invesco QQQ ETF',
            'description': 'Tracks the NASDAQ-100 Index'
        },
        BenchmarkType.IWM: {
            'name': 'iShares Russell 2000 ETF',
            'description': 'Tracks the Russell 2000 Index'
        },
        BenchmarkType.VTI: {
            'name': 'Vanguard Total Stock Market ETF',
            'description': 'Tracks the entire U.S. stock market'
        }
    }
    
    @staticmethod
    async def fetch_benchmark_data(
        benchmark_type: BenchmarkType,
        start_date: datetime,
        end_date: datetime,
        custom_symbol: Optional[str] = None
    ) -> Optional[BenchmarkData]:
        """
        Fetch benchmark data from Yahoo Finance.
        
        Args:
            benchmark_type: Type of benchmark
            start_date: Start date for data
            end_date: End date for data
            custom_symbol: Custom symbol if benchmark_type is CUSTOM
            
        Returns:
            BenchmarkData object or None if failed
        """
        try:
            # Determine symbol
            if benchmark_type == BenchmarkType.CUSTOM:
                if not custom_symbol:
                    raise ValueError("Custom symbol required for CUSTOM benchmark type")
                symbol = custom_symbol
                name = f"Custom Benchmark ({custom_symbol})"
            else:
                symbol = benchmark_type.value
                name = BenchmarkDataProvider.BENCHMARK_INFO[benchmark_type]['name']
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d'
            )
            
            if hist.empty:
                logger.error(f"No data found for benchmark {symbol}")
                return None
            
            # Extract prices and calculate returns
            prices = hist['Close'].tolist()
            timestamps = [int(ts.timestamp()) for ts in hist.index]
            
            if len(prices) < 2:
                logger.error(f"Insufficient data for benchmark {symbol}")
                return None
            
            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            
            # Calculate performance metrics
            total_return = (prices[-1] - prices[0]) / prices[0]
            days = len(prices) - 1
            annualized_return = (prices[-1] / prices[0]) ** (252 / days) - 1
            volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            excess_return = annualized_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
            
            # Maximum drawdown
            peak = prices[0]
            max_drawdown = 0.0
            for price in prices[1:]:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            return BenchmarkData(
                symbol=symbol,
                name=name,
                start_date=start_date,
                end_date=end_date,
                prices=prices,
                returns=returns,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                timestamps=timestamps
            )
            
        except Exception as e:
            logger.error(f"Error fetching benchmark data for {symbol}: {e}")
            return None

class BenchmarkComparator:
    """Compares portfolio performance against benchmarks."""
    
    def __init__(self):
        self.trades_repo = TradesRepository()
        self.portfolio_repo = PortfolioRepository()
        self.data_provider = BenchmarkDataProvider()
    
    async def compare_portfolio_to_benchmark(
        self,
        portfolio_metrics: PerformanceMetrics,
        portfolio_returns: List[float],
        benchmark_type: BenchmarkType,
        start_date: datetime,
        end_date: datetime,
        custom_symbol: Optional[str] = None
    ) -> Optional[ComparisonResult]:
        """
        Compare portfolio performance to a benchmark.
        
        Args:
            portfolio_metrics: Portfolio performance metrics
            portfolio_returns: Daily portfolio returns
            benchmark_type: Type of benchmark to compare against
            start_date: Start date for comparison
            end_date: End date for comparison
            custom_symbol: Custom benchmark symbol if needed
            
        Returns:
            ComparisonResult or None if failed
        """
        # Fetch benchmark data
        benchmark_data = await self.data_provider.fetch_benchmark_data(
            benchmark_type, start_date, end_date, custom_symbol
        )
        
        if not benchmark_data:
            logger.error("Failed to fetch benchmark data")
            return None
        
        # Align portfolio and benchmark returns
        aligned_portfolio_returns, aligned_benchmark_returns = self._align_returns(
            portfolio_returns, benchmark_data.returns
        )
        
        if len(aligned_portfolio_returns) < 10:
            logger.error("Insufficient aligned data for comparison")
            return None
        
        # Calculate comparison metrics
        alpha, beta = self._calculate_alpha_beta(
            aligned_portfolio_returns, aligned_benchmark_returns
        )
        
        correlation = np.corrcoef(aligned_portfolio_returns, aligned_benchmark_returns)[0, 1]
        tracking_error = np.std(np.array(aligned_portfolio_returns) - np.array(aligned_benchmark_returns)) * np.sqrt(252)
        
        # Information ratio
        excess_returns = np.array(aligned_portfolio_returns) - np.array(aligned_benchmark_returns)
        information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0.0
        
        # Capture ratios
        up_capture, down_capture = self._calculate_capture_ratios(
            aligned_portfolio_returns, aligned_benchmark_returns
        )
        
        # Statistical significance testing
        statistical_significance = self._test_statistical_significance(
            aligned_portfolio_returns, aligned_benchmark_returns
        )
        
        return ComparisonResult(
            portfolio_metrics=portfolio_metrics,
            benchmark_data=benchmark_data,
            alpha=alpha,
            beta=beta,
            correlation=correlation,
            tracking_error=tracking_error,
            information_ratio=information_ratio,
            up_capture=up_capture,
            down_capture=down_capture,
            statistical_significance=statistical_significance
        )
    
    async def compare_simulation_results_to_benchmarks(
        self,
        simulation_results: List[SimulationResult],
        benchmark_types: List[BenchmarkType],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Compare Monte Carlo simulation results to multiple benchmarks.
        
        Args:
            simulation_results: List of simulation results
            benchmark_types: List of benchmarks to compare against
            start_date: Start date for comparison
            end_date: End date for comparison
            
        Returns:
            Dictionary with comparison results for each benchmark
        """
        if not simulation_results:
            return {}
        
        # Extract portfolio statistics from simulation results
        portfolio_returns = []
        portfolio_metrics_list = []
        
        for result in simulation_results:
            portfolio_returns.extend(result.daily_returns)
            portfolio_metrics_list.append(result.performance_metrics)
        
        # Calculate aggregate portfolio metrics
        total_returns = [m.total_return for m in portfolio_metrics_list]
        sharpe_ratios = [m.sharpe_ratio for m in portfolio_metrics_list]
        max_drawdowns = [m.max_drawdown for m in portfolio_metrics_list]
        
        aggregate_metrics = PerformanceMetrics(
            total_return=np.mean(total_returns),
            annualized_return=np.mean([m.annualized_return for m in portfolio_metrics_list]),
            volatility=np.mean([m.volatility for m in portfolio_metrics_list]),
            sharpe_ratio=np.mean(sharpe_ratios),
            max_drawdown=np.mean(max_drawdowns),
            calmar_ratio=np.mean([m.calmar_ratio for m in portfolio_metrics_list]),
            win_rate=np.mean([m.win_rate for m in portfolio_metrics_list]),
            profit_factor=np.mean([m.profit_factor for m in portfolio_metrics_list]),
            var_95=np.mean([m.var_95 for m in portfolio_metrics_list]),
            cvar_95=np.mean([m.cvar_95 for m in portfolio_metrics_list]),
            final_portfolio_value=np.mean([m.final_portfolio_value for m in portfolio_metrics_list]),
            num_trades=int(np.mean([m.num_trades for m in portfolio_metrics_list]))
        )
        
        # Compare against each benchmark
        comparison_results = {}
        
        for benchmark_type in benchmark_types:
            try:
                comparison = await self.compare_portfolio_to_benchmark(
                    aggregate_metrics,
                    portfolio_returns,
                    benchmark_type,
                    start_date,
                    end_date
                )
                
                if comparison:
                    comparison_results[benchmark_type.value] = {
                        'benchmark_name': comparison.benchmark_data.name,
                        'alpha': comparison.alpha,
                        'beta': comparison.beta,
                        'correlation': comparison.correlation,
                        'tracking_error': comparison.tracking_error,
                        'information_ratio': comparison.information_ratio,
                        'up_capture': comparison.up_capture,
                        'down_capture': comparison.down_capture,
                        'portfolio_return': comparison.portfolio_metrics.total_return,
                        'benchmark_return': comparison.benchmark_data.total_return,
                        'portfolio_sharpe': comparison.portfolio_metrics.sharpe_ratio,
                        'benchmark_sharpe': comparison.benchmark_data.sharpe_ratio,
                        'portfolio_volatility': comparison.portfolio_metrics.volatility,
                        'benchmark_volatility': comparison.benchmark_data.volatility,
                        'statistical_significance': comparison.statistical_significance
                    }
                    
            except Exception as e:
                logger.error(f"Error comparing to benchmark {benchmark_type.value}: {e}")
                continue
        
        return comparison_results
    
    def _align_returns(
        self, 
        portfolio_returns: List[float], 
        benchmark_returns: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Align portfolio and benchmark returns by length.
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            
        Returns:
            Tuple of aligned returns
        """
        min_length = min(len(portfolio_returns), len(benchmark_returns))
        
        # Take the most recent returns if lengths differ
        aligned_portfolio = portfolio_returns[-min_length:] if len(portfolio_returns) > min_length else portfolio_returns
        aligned_benchmark = benchmark_returns[-min_length:] if len(benchmark_returns) > min_length else benchmark_returns
        
        return aligned_portfolio, aligned_benchmark
    
    def _calculate_alpha_beta(
        self, 
        portfolio_returns: List[float], 
        benchmark_returns: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate alpha and beta using linear regression.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Tuple of (alpha, beta)
        """
        try:
            # Convert to numpy arrays
            portfolio_array = np.array(portfolio_returns)
            benchmark_array = np.array(benchmark_returns)
            
            # Linear regression: portfolio_return = alpha + beta * benchmark_return
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                benchmark_array, portfolio_array
            )
            
            # Annualize alpha
            alpha = intercept * 252
            beta = slope
            
            return alpha, beta
            
        except Exception as e:
            logger.error(f"Error calculating alpha/beta: {e}")
            return 0.0, 1.0
    
    def _calculate_capture_ratios(
        self, 
        portfolio_returns: List[float], 
        benchmark_returns: List[float]
    ) -> Tuple[float, float]:
        """
        Calculate upside and downside capture ratios.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Tuple of (up_capture, down_capture)
        """
        try:
            portfolio_array = np.array(portfolio_returns)
            benchmark_array = np.array(benchmark_returns)
            
            # Separate up and down periods
            up_periods = benchmark_array > 0
            down_periods = benchmark_array < 0
            
            if np.sum(up_periods) > 0:
                up_portfolio = np.mean(portfolio_array[up_periods])
                up_benchmark = np.mean(benchmark_array[up_periods])
                up_capture = up_portfolio / up_benchmark if up_benchmark != 0 else 1.0
            else:
                up_capture = 1.0
            
            if np.sum(down_periods) > 0:
                down_portfolio = np.mean(portfolio_array[down_periods])
                down_benchmark = np.mean(benchmark_array[down_periods])
                down_capture = down_portfolio / down_benchmark if down_benchmark != 0 else 1.0
            else:
                down_capture = 1.0
            
            return up_capture, down_capture
            
        except Exception as e:
            logger.error(f"Error calculating capture ratios: {e}")
            return 1.0, 1.0
    
    def _test_statistical_significance(
        self, 
        portfolio_returns: List[float], 
        benchmark_returns: List[float]
    ) -> Dict[str, Any]:
        """
        Test statistical significance of performance differences.
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Dictionary with statistical test results
        """
        try:
            portfolio_array = np.array(portfolio_returns)
            benchmark_array = np.array(benchmark_returns)
            excess_returns = portfolio_array - benchmark_array
            
            # T-test for mean excess return
            t_stat, t_p_value = stats.ttest_1samp(excess_returns, 0)
            
            # Kolmogorov-Smirnov test for distribution differences
            ks_stat, ks_p_value = stats.ks_2samp(portfolio_returns, benchmark_returns)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(
                portfolio_returns, benchmark_returns, alternative='two-sided'
            )
            
            return {
                'mean_excess_return': float(np.mean(excess_returns)),
                'excess_return_std': float(np.std(excess_returns)),
                't_test': {
                    'statistic': float(t_stat),
                    'p_value': float(t_p_value),
                    'significant_at_5pct': t_p_value < 0.05,
                    'significant_at_1pct': t_p_value < 0.01
                },
                'ks_test': {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_p_value),
                    'significant_at_5pct': ks_p_value < 0.05
                },
                'mann_whitney_test': {
                    'statistic': float(u_stat),
                    'p_value': float(u_p_value),
                    'significant_at_5pct': u_p_value < 0.05
                }
            }
            
        except Exception as e:
            logger.error(f"Error in statistical significance testing: {e}")
            return {}

class BenchmarkReportGenerator:
    """Generates comprehensive benchmark comparison reports."""
    
    @staticmethod
    def generate_comparison_report(
        comparison_results: Dict[str, Any],
        simulation_summary: Dict[str, Any]
    ) -> str:
        """
        Generate a comprehensive benchmark comparison report.
        
        Args:
            comparison_results: Results from benchmark comparison
            simulation_summary: Summary of simulation results
            
        Returns:
            Formatted report string
        """
        if not comparison_results:
            return "No benchmark comparison data available."
        
        report_lines = [
            "Portfolio vs Benchmark Analysis Report",
            "=" * 50,
            "",
            "SIMULATION OVERVIEW:",
            f"Number of simulations: {simulation_summary.get('num_simulations', 'N/A')}",
            f"Mean portfolio return: {simulation_summary.get('returns', {}).get('mean', 0):.2%}",
            f"Portfolio volatility: {simulation_summary.get('returns', {}).get('std', 0):.2%}",
            "",
            "BENCHMARK COMPARISONS:",
            "-" * 30
        ]
        
        for benchmark_symbol, results in comparison_results.items():
            report_lines.extend([
                f"",
                f"{results['benchmark_name']} ({benchmark_symbol}):",
                f"  Portfolio Return: {results['portfolio_return']:.2%}",
                f"  Benchmark Return: {results['benchmark_return']:.2%}",
                f"  Alpha (excess return): {results['alpha']:.2%}",
                f"  Beta (market sensitivity): {results['beta']:.3f}",
                f"  Correlation: {results['correlation']:.3f}",
                f"  Tracking Error: {results['tracking_error']:.2%}",
                f"  Information Ratio: {results['information_ratio']:.3f}",
                f"  Upside Capture: {results['up_capture']:.1%}",
                f"  Downside Capture: {results['down_capture']:.1%}",
                f"  Portfolio Sharpe: {results['portfolio_sharpe']:.3f}",
                f"  Benchmark Sharpe: {results['benchmark_sharpe']:.3f}"
            ])
            
            # Add statistical significance
            if 'statistical_significance' in results:
                sig = results['statistical_significance']
                if 't_test' in sig:
                    t_test = sig['t_test']
                    significance = "Yes" if t_test.get('significant_at_5pct', False) else "No"
                    report_lines.append(f"  Statistically Significant (5%): {significance}")
        
        report_lines.extend([
            "",
            "INTERPRETATION GUIDE:",
            "-" * 20,
            "• Alpha > 0: Portfolio outperformed benchmark",
            "• Beta > 1: Portfolio more volatile than benchmark",
            "• Information Ratio > 0.5: Good risk-adjusted excess return",
            "• Upside Capture > 100%: Captured more upside than benchmark",
            "• Downside Capture < 100%: Better downside protection"
        ])
        
        return "\n".join(report_lines)
    
    @staticmethod
    def generate_statistical_summary(comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate statistical summary of benchmark comparisons.
        
        Args:
            comparison_results: Results from benchmark comparison
            
        Returns:
            Statistical summary dictionary
        """
        if not comparison_results:
            return {}
        
        # Collect metrics across all benchmarks
        alphas = [r['alpha'] for r in comparison_results.values()]
        betas = [r['beta'] for r in comparison_results.values()]
        correlations = [r['correlation'] for r in comparison_results.values()]
        information_ratios = [r['information_ratio'] for r in comparison_results.values()]
        
        return {
            'alpha_stats': {
                'mean': np.mean(alphas),
                'std': np.std(alphas),
                'min': np.min(alphas),
                'max': np.max(alphas),
                'positive_count': sum(1 for a in alphas if a > 0)
            },
            'beta_stats': {
                'mean': np.mean(betas),
                'std': np.std(betas),
                'min': np.min(betas),
                'max': np.max(betas)
            },
            'correlation_stats': {
                'mean': np.mean(correlations),
                'std': np.std(correlations),
                'min': np.min(correlations),
                'max': np.max(correlations)
            },
            'information_ratio_stats': {
                'mean': np.mean(information_ratios),
                'std': np.std(information_ratios),
                'positive_count': sum(1 for ir in information_ratios if ir > 0)
            },
            'benchmarks_outperformed': sum(1 for a in alphas if a > 0),
            'total_benchmarks': len(alphas)
        }