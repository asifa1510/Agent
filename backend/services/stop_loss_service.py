"""
Stop-Loss Management Service

Implements automatic stop-loss order placement, monitoring, and execution.
Handles 2% threshold stop-loss orders and position exit logic.

Requirements: 4.4
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import uuid

from ..models.data_models import Trade, PortfolioPosition
from ..repositories.trades_repository import TradesRepository
from ..repositories.portfolio_repository import PortfolioRepository
from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class StopLossOrder:
    """Stop-loss order configuration"""
    id: str
    symbol: str
    position_quantity: int
    entry_price: float
    stop_price: float
    stop_percentage: float
    created_timestamp: int
    status: str  # 'active', 'triggered', 'cancelled', 'expired'
    trigger_timestamp: Optional[int] = None
    execution_price: Optional[float] = None

@dataclass
class StopLossConfig:
    """Stop-loss configuration parameters"""
    default_stop_percentage: float = 0.02  # 2% stop loss
    trailing_stop_enabled: bool = False
    trailing_stop_percentage: float = 0.02
    max_stop_orders_per_symbol: int = 1
    order_expiry_hours: int = 24  # Hours before stop order expires

class StopLossService:
    """
    Service for managing stop-loss orders and automatic position exits.
    
    Handles automatic stop-loss order placement with 2% threshold,
    monitors positions for stop-loss triggers, and executes position exits.
    """
    
    def __init__(self):
        self.trades_repo = TradesRepository()
        self.portfolio_repo = PortfolioRepository()
        self.config = StopLossConfig()
        
        # In-memory storage for active stop-loss orders
        # In production, this would be stored in a database
        self.active_orders: Dict[str, StopLossOrder] = {}
        self.symbol_orders: Dict[str, List[str]] = {}  # symbol -> list of order IDs
    
    async def create_stop_loss_order(
        self, 
        symbol: str,
        position_quantity: int,
        entry_price: float,
        stop_percentage: Optional[float] = None
    ) -> Optional[StopLossOrder]:
        """
        Create a stop-loss order for a position.
        
        Args:
            symbol: Stock symbol
            position_quantity: Number of shares in position
            entry_price: Entry price for the position
            stop_percentage: Stop-loss percentage (default: 2%)
            
        Returns:
            Created StopLossOrder or None if failed
        """
        try:
            if stop_percentage is None:
                stop_percentage = self.config.default_stop_percentage
            
            # Check if we already have max orders for this symbol
            existing_orders = self.symbol_orders.get(symbol, [])
            active_orders = [
                order_id for order_id in existing_orders
                if self.active_orders.get(order_id, {}).status == 'active'
            ]
            
            if len(active_orders) >= self.config.max_stop_orders_per_symbol:
                logger.warning(f"Maximum stop orders reached for {symbol}")
                return None
            
            # Calculate stop price
            stop_price = entry_price * (1 - stop_percentage)
            
            # Create stop-loss order
            order = StopLossOrder(
                id=str(uuid.uuid4()),
                symbol=symbol,
                position_quantity=position_quantity,
                entry_price=entry_price,
                stop_price=stop_price,
                stop_percentage=stop_percentage,
                created_timestamp=int(datetime.now().timestamp()),
                status='active'
            )
            
            # Store order
            self.active_orders[order.id] = order
            
            if symbol not in self.symbol_orders:
                self.symbol_orders[symbol] = []
            self.symbol_orders[symbol].append(order.id)
            
            logger.info(f"Created stop-loss order for {symbol}: {position_quantity} shares at ${stop_price:.2f} "
                       f"({stop_percentage:.1%} below ${entry_price:.2f})")
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating stop-loss order for {symbol}: {e}")
            return None
    
    async def auto_create_stop_loss_for_position(self, symbol: str) -> Optional[StopLossOrder]:
        """
        Automatically create stop-loss order for an existing position.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Created StopLossOrder or None if failed/not needed
        """
        try:
            # Get current position
            position = await self.portfolio_repo.get_position(symbol)
            
            if not position or position.quantity <= 0:
                logger.info(f"No long position found for {symbol}, skipping stop-loss creation")
                return None
            
            # Check if stop-loss already exists
            existing_orders = self.symbol_orders.get(symbol, [])
            active_orders = [
                self.active_orders[order_id] for order_id in existing_orders
                if order_id in self.active_orders and self.active_orders[order_id].status == 'active'
            ]
            
            if active_orders:
                logger.info(f"Stop-loss order already exists for {symbol}")
                return active_orders[0]
            
            # Create stop-loss order
            return await self.create_stop_loss_order(
                symbol=symbol,
                position_quantity=position.quantity,
                entry_price=position.avg_price
            )
            
        except Exception as e:
            logger.error(f"Error auto-creating stop-loss for {symbol}: {e}")
            return None
    
    async def check_stop_loss_triggers(self, market_prices: Dict[str, float]) -> List[StopLossOrder]:
        """
        Check all active stop-loss orders for trigger conditions.
        
        Args:
            market_prices: Dictionary mapping symbols to current market prices
            
        Returns:
            List of triggered stop-loss orders
        """
        triggered_orders = []
        
        try:
            current_timestamp = int(datetime.now().timestamp())
            
            for order_id, order in self.active_orders.items():
                if order.status != 'active':
                    continue
                
                # Check if order has expired
                if self._is_order_expired(order, current_timestamp):
                    order.status = 'expired'
                    logger.info(f"Stop-loss order {order_id} expired for {order.symbol}")
                    continue
                
                # Check if we have current price for this symbol
                current_price = market_prices.get(order.symbol)
                if current_price is None:
                    continue
                
                # Check if stop price is triggered
                if current_price <= order.stop_price:
                    order.status = 'triggered'
                    order.trigger_timestamp = current_timestamp
                    order.execution_price = current_price
                    
                    triggered_orders.append(order)
                    
                    logger.warning(f"Stop-loss triggered for {order.symbol}: "
                                 f"Price ${current_price:.2f} <= Stop ${order.stop_price:.2f}")
            
            return triggered_orders
            
        except Exception as e:
            logger.error(f"Error checking stop-loss triggers: {e}")
            return []
    
    async def execute_stop_loss_order(self, order: StopLossOrder) -> Optional[Trade]:
        """
        Execute a triggered stop-loss order.
        
        Args:
            order: Triggered stop-loss order
            
        Returns:
            Executed Trade or None if failed
        """
        try:
            if order.status != 'triggered':
                logger.warning(f"Attempted to execute non-triggered stop-loss order {order.id}")
                return None
            
            # Verify we still have the position
            position = await self.portfolio_repo.get_position(order.symbol)
            if not position or position.quantity < order.position_quantity:
                logger.warning(f"Insufficient position for stop-loss execution: {order.symbol}")
                order.status = 'cancelled'
                return None
            
            # Create sell trade
            trade = Trade(
                id=str(uuid.uuid4()),
                symbol=order.symbol,
                timestamp=order.trigger_timestamp or int(datetime.now().timestamp()),
                action='sell',
                quantity=order.position_quantity,
                price=order.execution_price or order.stop_price,
                signal_strength=1.0,  # Stop-loss is always executed
                explanation_id=None
            )
            
            # Execute the trade
            success = await self.trades_repo.create_item(trade)
            
            if success:
                # Update portfolio position
                await self._update_position_after_stop_loss(trade, position)
                
                # Mark order as executed
                order.status = 'executed'
                
                logger.info(f"Executed stop-loss order: Sold {trade.quantity} shares of {trade.symbol} "
                           f"at ${trade.price:.2f}")
                
                return trade
            else:
                logger.error(f"Failed to store stop-loss trade for {order.symbol}")
                return None
            
        except Exception as e:
            logger.error(f"Error executing stop-loss order {order.id}: {e}")
            return None
    
    async def _update_position_after_stop_loss(self, trade: Trade, position: PortfolioPosition) -> None:
        """Update portfolio position after stop-loss execution."""
        try:
            new_quantity = position.quantity - trade.quantity
            
            # Update position
            await self.portfolio_repo.update_position(
                symbol=trade.symbol,
                quantity=new_quantity,
                avg_price=position.avg_price,  # Keep same average price
                current_price=trade.price
            )
            
            # Update allocation percentages
            await self.portfolio_repo.update_allocation_percentages()
            
            logger.info(f"Updated position for {trade.symbol}: {position.quantity} -> {new_quantity} shares")
            
        except Exception as e:
            logger.error(f"Error updating position after stop-loss: {e}")
    
    def _is_order_expired(self, order: StopLossOrder, current_timestamp: int) -> bool:
        """Check if a stop-loss order has expired."""
        expiry_seconds = self.config.order_expiry_hours * 3600
        return (current_timestamp - order.created_timestamp) > expiry_seconds
    
    async def cancel_stop_loss_order(self, order_id: str) -> bool:
        """
        Cancel an active stop-loss order.
        
        Args:
            order_id: Stop-loss order ID
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Stop-loss order {order_id} not found")
                return False
            
            order = self.active_orders[order_id]
            
            if order.status != 'active':
                logger.warning(f"Cannot cancel stop-loss order {order_id} with status {order.status}")
                return False
            
            order.status = 'cancelled'
            
            logger.info(f"Cancelled stop-loss order {order_id} for {order.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling stop-loss order {order_id}: {e}")
            return False
    
    async def cancel_all_orders_for_symbol(self, symbol: str) -> int:
        """
        Cancel all active stop-loss orders for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Number of orders cancelled
        """
        try:
            cancelled_count = 0
            order_ids = self.symbol_orders.get(symbol, [])
            
            for order_id in order_ids:
                if order_id in self.active_orders:
                    order = self.active_orders[order_id]
                    if order.status == 'active':
                        order.status = 'cancelled'
                        cancelled_count += 1
            
            logger.info(f"Cancelled {cancelled_count} stop-loss orders for {symbol}")
            return cancelled_count
            
        except Exception as e:
            logger.error(f"Error cancelling orders for {symbol}: {e}")
            return 0
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[StopLossOrder]:
        """
        Get active stop-loss orders.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of active stop-loss orders
        """
        try:
            active_orders = []
            
            for order in self.active_orders.values():
                if order.status == 'active':
                    if symbol is None or order.symbol == symbol:
                        active_orders.append(order)
            
            return active_orders
            
        except Exception as e:
            logger.error(f"Error getting active orders: {e}")
            return []
    
    def get_order_by_id(self, order_id: str) -> Optional[StopLossOrder]:
        """Get stop-loss order by ID."""
        return self.active_orders.get(order_id)
    
    async def monitor_and_execute_stop_losses(self, market_prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Monitor all positions and execute triggered stop-losses.
        
        Args:
            market_prices: Current market prices by symbol
            
        Returns:
            Dictionary with monitoring results
        """
        try:
            # Check for triggered orders
            triggered_orders = await self.check_stop_loss_triggers(market_prices)
            
            executed_trades = []
            failed_executions = []
            
            # Execute triggered orders
            for order in triggered_orders:
                trade = await self.execute_stop_loss_order(order)
                if trade:
                    executed_trades.append(trade)
                else:
                    failed_executions.append(order)
            
            # Clean up expired orders
            expired_count = self._cleanup_expired_orders()
            
            result = {
                'timestamp': int(datetime.now().timestamp()),
                'triggered_orders': len(triggered_orders),
                'executed_trades': len(executed_trades),
                'failed_executions': len(failed_executions),
                'expired_orders_cleaned': expired_count,
                'active_orders_count': len(self.get_active_orders()),
                'executed_trade_ids': [trade.id for trade in executed_trades]
            }
            
            if executed_trades:
                logger.info(f"Stop-loss monitoring: Executed {len(executed_trades)} trades")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in stop-loss monitoring: {e}")
            return {'error': str(e)}
    
    def _cleanup_expired_orders(self) -> int:
        """Clean up expired stop-loss orders."""
        try:
            current_timestamp = int(datetime.now().timestamp())
            expired_count = 0
            
            for order in self.active_orders.values():
                if order.status == 'active' and self._is_order_expired(order, current_timestamp):
                    order.status = 'expired'
                    expired_count += 1
            
            return expired_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired orders: {e}")
            return 0
    
    async def create_stop_losses_for_all_positions(self) -> Dict[str, Any]:
        """
        Create stop-loss orders for all current long positions that don't have them.
        
        Returns:
            Dictionary with creation results
        """
        try:
            positions = await self.portfolio_repo.get_active_positions()
            long_positions = [pos for pos in positions if pos.quantity > 0]
            
            created_orders = []
            skipped_positions = []
            
            for position in long_positions:
                # Check if stop-loss already exists
                existing_orders = self.get_active_orders(position.symbol)
                
                if existing_orders:
                    skipped_positions.append(position.symbol)
                    continue
                
                # Create stop-loss order
                order = await self.create_stop_loss_order(
                    symbol=position.symbol,
                    position_quantity=position.quantity,
                    entry_price=position.avg_price
                )
                
                if order:
                    created_orders.append(order)
            
            result = {
                'total_positions': len(long_positions),
                'created_orders': len(created_orders),
                'skipped_positions': len(skipped_positions),
                'created_order_ids': [order.id for order in created_orders],
                'skipped_symbols': skipped_positions
            }
            
            logger.info(f"Created {len(created_orders)} stop-loss orders for {len(long_positions)} positions")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating stop-losses for all positions: {e}")
            return {'error': str(e)}
    
    def get_stop_loss_summary(self) -> Dict[str, Any]:
        """Get summary of all stop-loss orders and their status."""
        try:
            status_counts = {}
            orders_by_symbol = {}
            
            for order in self.active_orders.values():
                # Count by status
                status_counts[order.status] = status_counts.get(order.status, 0) + 1
                
                # Group by symbol
                if order.symbol not in orders_by_symbol:
                    orders_by_symbol[order.symbol] = []
                orders_by_symbol[order.symbol].append({
                    'id': order.id,
                    'status': order.status,
                    'stop_price': order.stop_price,
                    'stop_percentage': order.stop_percentage,
                    'created': order.created_timestamp
                })
            
            return {
                'total_orders': len(self.active_orders),
                'status_counts': status_counts,
                'orders_by_symbol': orders_by_symbol,
                'config': {
                    'default_stop_percentage': self.config.default_stop_percentage,
                    'max_orders_per_symbol': self.config.max_stop_orders_per_symbol,
                    'order_expiry_hours': self.config.order_expiry_hours
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting stop-loss summary: {e}")
            return {'error': str(e)}
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update stop-loss configuration."""
        try:
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    logger.info(f"Updated stop-loss config {key} to {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating stop-loss config: {e}")
            return False