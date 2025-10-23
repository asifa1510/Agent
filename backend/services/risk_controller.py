"""
Risk Controller Service

Implements risk management functionality including:
- Position size validation
- Portfolio allocation limits (5% per position)
- Volatility-based trading halts
- Stop-loss management

Requirements: 4.1, 4.3, 4.5
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

from ..models.data_models import Trade, PortfolioPosition, RiskMetrics
from ..repositories.portfolio_repository import PortfolioRepository
from ..repositories.trades_repository import TradesRepository
from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    """Risk management limits configuration"""
    max_position_allocation: float = 0.05  # 5% max per position
    max_portfolio_risk: float = 0.20  # 20% max portfolio risk
    volatility_halt_threshold: float = 0.30  # 30% volatility threshold
    stop_loss_threshold: float = 0.02  # 2% stop loss threshold
    max_daily_trades: int = 50  # Maximum trades per day
    min_signal_strength: float = 0.6  # Minimum signal strength for trades

@dataclass
class RiskAssessment:
    """Risk assessment result"""
    approved: bool
    risk_score: float
    warnings: List[str]
    violations: List[str]
    recommended_position_size: Optional[int] = None

class RiskController:
    """
    Risk management controller for trading operations.
    
    Validates trades against risk limits and portfolio constraints.
    """
    
    def __init__(self):
        self.portfolio_repo = PortfolioRepository()
        self.trades_repo = TradesRepository()
        self.risk_limits = RiskLimits()
        self._volatility_cache = {}
        self._halt_status = {}
    
    async def validate_trade(
        self, 
        symbol: str, 
        action: str, 
        quantity: int, 
        price: float,
        signal_strength: float
    ) -> RiskAssessment:
        """
        Validate a proposed trade against risk management rules.
        
        Args:
            symbol: Stock symbol
            action: Trade action ('buy' or 'sell')
            quantity: Number of shares
            price: Price per share
            signal_strength: Trading signal strength (0-1)
            
        Returns:
            RiskAssessment with approval status and details
        """
        warnings = []
        violations = []
        
        try:
            # Check if trading is halted for this symbol
            if await self._is_trading_halted(symbol):
                violations.append(f"Trading halted for {symbol} due to high volatility")
            
            # Check signal strength
            if signal_strength < self.risk_limits.min_signal_strength:
                violations.append(f"Signal strength {signal_strength:.2f} below minimum {self.risk_limits.min_signal_strength}")
            
            # Check daily trade limits
            daily_trades = await self._get_daily_trade_count()
            if daily_trades >= self.risk_limits.max_daily_trades:
                violations.append(f"Daily trade limit reached: {daily_trades}/{self.risk_limits.max_daily_trades}")
            
            # For buy orders, check position size and allocation limits
            if action == 'buy':
                position_check = await self._validate_position_size(symbol, quantity, price)
                warnings.extend(position_check['warnings'])
                violations.extend(position_check['violations'])
                
                allocation_check = await self._validate_allocation_limit(symbol, quantity, price)
                warnings.extend(allocation_check['warnings'])
                violations.extend(allocation_check['violations'])
            
            # For sell orders, validate we have sufficient position
            elif action == 'sell':
                position_check = await self._validate_sell_position(symbol, quantity)
                violations.extend(position_check['violations'])
            
            # Calculate overall risk score
            risk_score = await self._calculate_risk_score(symbol, action, quantity, price)
            
            # Determine if trade is approved
            approved = len(violations) == 0
            
            # Calculate recommended position size if trade was rejected due to size
            recommended_size = None
            if not approved and action == 'buy':
                recommended_size = await self._calculate_max_position_size(symbol, price)
            
            return RiskAssessment(
                approved=approved,
                risk_score=risk_score,
                warnings=warnings,
                violations=violations,
                recommended_position_size=recommended_size
            )
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return RiskAssessment(
                approved=False,
                risk_score=1.0,
                warnings=[],
                violations=[f"Risk validation error: {str(e)}"]
            )
    
    async def _validate_position_size(
        self, 
        symbol: str, 
        quantity: int, 
        price: float
    ) -> Dict[str, List[str]]:
        """Validate position size against limits."""
        warnings = []
        violations = []
        
        try:
            # Get current position
            current_position = await self.portfolio_repo.get_position(symbol)
            current_quantity = current_position.quantity if current_position else 0
            
            # Calculate new position size
            new_quantity = current_quantity + quantity
            position_value = new_quantity * price
            
            # Get total portfolio value
            metrics = await self.portfolio_repo.calculate_portfolio_metrics()
            total_portfolio_value = metrics.total_value
            
            if total_portfolio_value > 0:
                allocation_percent = position_value / total_portfolio_value
                
                if allocation_percent > self.risk_limits.max_position_allocation:
                    violations.append(
                        f"Position allocation {allocation_percent:.1%} exceeds limit "
                        f"{self.risk_limits.max_position_allocation:.1%}"
                    )
                elif allocation_percent > self.risk_limits.max_position_allocation * 0.8:
                    warnings.append(
                        f"Position allocation {allocation_percent:.1%} approaching limit "
                        f"{self.risk_limits.max_position_allocation:.1%}"
                    )
            
        except Exception as e:
            logger.error(f"Error validating position size: {e}")
            violations.append("Unable to validate position size")
        
        return {'warnings': warnings, 'violations': violations}
    
    async def _validate_allocation_limit(
        self, 
        symbol: str, 
        quantity: int, 
        price: float
    ) -> Dict[str, List[str]]:
        """Validate allocation limit enforcement."""
        warnings = []
        violations = []
        
        try:
            # Get all active positions
            positions = await self.portfolio_repo.get_active_positions()
            
            # Calculate total portfolio value including new trade
            total_value = sum(pos.current_price * abs(pos.quantity) for pos in positions)
            trade_value = quantity * price
            new_total_value = total_value + trade_value
            
            if new_total_value > 0:
                new_allocation = trade_value / new_total_value
                
                if new_allocation > self.risk_limits.max_position_allocation:
                    violations.append(
                        f"New position would represent {new_allocation:.1%} of portfolio, "
                        f"exceeding {self.risk_limits.max_position_allocation:.1%} limit"
                    )
            
        except Exception as e:
            logger.error(f"Error validating allocation limit: {e}")
            violations.append("Unable to validate allocation limit")
        
        return {'warnings': warnings, 'violations': violations}
    
    async def _validate_sell_position(self, symbol: str, quantity: int) -> Dict[str, List[str]]:
        """Validate we have sufficient position to sell."""
        violations = []
        
        try:
            current_position = await self.portfolio_repo.get_position(symbol)
            current_quantity = current_position.quantity if current_position else 0
            
            if current_quantity < quantity:
                violations.append(
                    f"Insufficient position to sell {quantity} shares of {symbol}. "
                    f"Current position: {current_quantity}"
                )
        except Exception as e:
            logger.error(f"Error validating sell position: {e}")
            violations.append("Unable to validate sell position")
        
        return {'violations': violations}
    
    async def _is_trading_halted(self, symbol: str) -> bool:
        """Check if trading is halted for a symbol due to volatility."""
        try:
            # Check halt status cache
            if symbol in self._halt_status:
                halt_info = self._halt_status[symbol]
                # Check if halt is still active (1 hour halt period)
                if datetime.now().timestamp() - halt_info['timestamp'] < 3600:
                    return halt_info['halted']
            
            # Calculate current volatility
            volatility = await self._calculate_volatility(symbol)
            
            # Check if volatility exceeds threshold
            halted = volatility > self.risk_limits.volatility_halt_threshold
            
            # Update halt status cache
            self._halt_status[symbol] = {
                'halted': halted,
                'timestamp': datetime.now().timestamp(),
                'volatility': volatility
            }
            
            if halted:
                logger.warning(f"Trading halted for {symbol} due to high volatility: {volatility:.2%}")
            
            return halted
            
        except Exception as e:
            logger.error(f"Error checking trading halt status: {e}")
            return False
    
    async def _calculate_volatility(self, symbol: str) -> float:
        """Calculate recent volatility for a symbol."""
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H')}"
            if cache_key in self._volatility_cache:
                return self._volatility_cache[cache_key]
            
            # Get recent trades for the symbol
            trades = await self.trades_repo.get_trades_by_symbol(symbol, limit=50)
            
            if len(trades) < 5:
                return 0.0  # Not enough data
            
            # Calculate price volatility from recent trades
            prices = [trade.price for trade in trades[-20:]]  # Last 20 trades
            
            if len(prices) < 2:
                return 0.0
            
            # Calculate standard deviation of price changes
            price_changes = []
            for i in range(1, len(prices)):
                change = (prices[i] - prices[i-1]) / prices[i-1]
                price_changes.append(change)
            
            if not price_changes:
                return 0.0
            
            # Calculate volatility (standard deviation)
            mean_change = sum(price_changes) / len(price_changes)
            variance = sum((x - mean_change) ** 2 for x in price_changes) / len(price_changes)
            volatility = variance ** 0.5
            
            # Cache the result
            self._volatility_cache[cache_key] = volatility
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return 0.0
    
    async def _get_daily_trade_count(self) -> int:
        """Get number of trades executed today."""
        try:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            today_timestamp = int(today_start.timestamp())
            
            trades = await self.trades_repo.get_recent_trades(days_back=1)
            daily_trades = [t for t in trades if t.timestamp >= today_timestamp]
            
            return len(daily_trades)
            
        except Exception as e:
            logger.error(f"Error getting daily trade count: {e}")
            return 0
    
    async def _calculate_risk_score(
        self, 
        symbol: str, 
        action: str, 
        quantity: int, 
        price: float
    ) -> float:
        """Calculate overall risk score for the trade (0-1, higher is riskier)."""
        try:
            risk_factors = []
            
            # Volatility risk
            volatility = await self._calculate_volatility(symbol)
            volatility_risk = min(volatility / self.risk_limits.volatility_halt_threshold, 1.0)
            risk_factors.append(volatility_risk * 0.3)
            
            # Position concentration risk
            metrics = await self.portfolio_repo.calculate_portfolio_metrics()
            if metrics.total_value > 0:
                trade_value = quantity * price
                concentration_risk = min(trade_value / metrics.total_value / self.risk_limits.max_position_allocation, 1.0)
                risk_factors.append(concentration_risk * 0.4)
            
            # Portfolio size risk (smaller portfolios are riskier)
            portfolio_size_risk = max(0, 1.0 - metrics.total_value / 100000)  # Assume $100k as baseline
            risk_factors.append(portfolio_size_risk * 0.2)
            
            # Daily trading frequency risk
            daily_trades = await self._get_daily_trade_count()
            frequency_risk = min(daily_trades / self.risk_limits.max_daily_trades, 1.0)
            risk_factors.append(frequency_risk * 0.1)
            
            # Calculate weighted average
            total_risk = sum(risk_factors)
            return min(total_risk, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5  # Default moderate risk
    
    async def _calculate_max_position_size(self, symbol: str, price: float) -> int:
        """Calculate maximum allowed position size for a symbol."""
        try:
            metrics = await self.portfolio_repo.calculate_portfolio_metrics()
            
            if metrics.total_value <= 0:
                return 0
            
            max_position_value = metrics.total_value * self.risk_limits.max_position_allocation
            max_quantity = int(max_position_value / price)
            
            return max(max_quantity, 0)
            
        except Exception as e:
            logger.error(f"Error calculating max position size: {e}")
            return 0
    
    async def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status and metrics."""
        try:
            metrics = await self.portfolio_repo.calculate_portfolio_metrics()
            daily_trades = await self._get_daily_trade_count()
            
            # Get halt status for active positions
            positions = await self.portfolio_repo.get_active_positions()
            halted_symbols = []
            
            for position in positions:
                if await self._is_trading_halted(position.symbol):
                    halted_symbols.append(position.symbol)
            
            return {
                'portfolio_metrics': metrics.dict(),
                'daily_trades': daily_trades,
                'max_daily_trades': self.risk_limits.max_daily_trades,
                'halted_symbols': halted_symbols,
                'risk_limits': {
                    'max_position_allocation': self.risk_limits.max_position_allocation,
                    'volatility_halt_threshold': self.risk_limits.volatility_halt_threshold,
                    'stop_loss_threshold': self.risk_limits.stop_loss_threshold,
                    'min_signal_strength': self.risk_limits.min_signal_strength
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting risk status: {e}")
            return {'error': str(e)}
    
    async def update_risk_limits(self, new_limits: Dict[str, float]) -> bool:
        """Update risk management limits."""
        try:
            for key, value in new_limits.items():
                if hasattr(self.risk_limits, key):
                    setattr(self.risk_limits, key, value)
                    logger.info(f"Updated risk limit {key} to {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating risk limits: {e}")
            return False