"""
Repository for trade data operations.
Handles DynamoDB operations for trading transactions.
"""

import logging
from typing import List, Optional, Type, Dict, Any
from datetime import datetime, timedelta

from .base_repository import BaseRepository
from ..models.data_models import Trade
from ..config import settings

logger = logging.getLogger(__name__)

class TradesRepository(BaseRepository):
    """Repository for trade data."""
    
    def __init__(self):
        super().__init__(settings.trades_table)
    
    def get_model_class(self) -> Type[Trade]:
        """Return the Trade model class."""
        return Trade
    
    async def get_by_id(self, trade_id: str) -> Optional[Trade]:
        """
        Get a trade by its ID.
        
        Args:
            trade_id: Unique trade identifier
            
        Returns:
            Trade instance if found, None otherwise
        """
        return await self.get_item({'id': trade_id})
    
    async def get_trades_by_symbol(
        self, 
        symbol: str, 
        limit: int = 100
    ) -> List[Trade]:
        """
        Get trades for a specific symbol.
        
        Args:
            symbol: Stock symbol
            limit: Maximum number of records to return
            
        Returns:
            List of trades for the symbol
        """
        filter_expression = "symbol = :symbol"
        expression_values = {':symbol': symbol}
        
        return await self.scan_items(
            filter_expression=filter_expression,
            expression_values=expression_values,
            limit=limit
        )
    
    async def get_trades_by_action(
        self, 
        action: str, 
        limit: int = 100
    ) -> List[Trade]:
        """
        Get trades by action type (buy/sell).
        
        Args:
            action: Trade action ('buy' or 'sell')
            limit: Maximum number of records to return
            
        Returns:
            List of trades with the specified action
        """
        filter_expression = "action = :action"
        expression_values = {':action': action}
        
        return await self.scan_items(
            filter_expression=filter_expression,
            expression_values=expression_values,
            limit=limit
        )
    
    async def get_recent_trades(
        self, 
        days_back: int = 30, 
        limit: int = 1000
    ) -> List[Trade]:
        """
        Get recent trades within the specified time period.
        
        Args:
            days_back: Number of days to look back
            limit: Maximum number of records to return
            
        Returns:
            List of recent trades
        """
        cutoff_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        filter_expression = "#ts >= :cutoff_ts"
        expression_values = {':cutoff_ts': cutoff_timestamp}
        expression_names = {'#ts': 'timestamp'}
        
        try:
            scan_params = {
                'FilterExpression': filter_expression,
                'ExpressionAttributeValues': expression_values,
                'ExpressionAttributeNames': expression_names,
                'Limit': limit
            }
            
            response = self.table.scan(**scan_params)
            trades = [Trade(**item) for item in response.get('Items', [])]
            
            # Sort by timestamp descending (most recent first)
            return sorted(trades, key=lambda x: x.timestamp, reverse=True)
        except Exception as e:
            logger.error(f"Error scanning recent trades: {e}")
            return []
    
    async def get_trade_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Calculate trade summary statistics.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Dictionary with trade summary statistics
        """
        trades = await self.get_recent_trades(days_back)
        
        if not trades:
            return {
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'total_volume': 0,
                'avg_signal_strength': 0.0,
                'unique_symbols': 0,
                'date_range': (0, 0)
            }
        
        buy_trades = [t for t in trades if t.action == 'buy']
        sell_trades = [t for t in trades if t.action == 'sell']
        total_volume = sum(t.quantity for t in trades)
        avg_signal_strength = sum(t.signal_strength for t in trades) / len(trades)
        unique_symbols = len(set(t.symbol for t in trades))
        
        oldest_timestamp = min(t.timestamp for t in trades)
        newest_timestamp = max(t.timestamp for t in trades)
        
        return {
            'total_trades': len(trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'total_volume': total_volume,
            'avg_signal_strength': avg_signal_strength,
            'unique_symbols': unique_symbols,
            'date_range': (oldest_timestamp, newest_timestamp)
        }
    
    async def get_position_trades(self, symbol: str) -> List[Trade]:
        """
        Get all trades for a symbol to calculate current position.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of all trades for the symbol, sorted by timestamp
        """
        trades = await self.get_trades_by_symbol(symbol)
        return sorted(trades, key=lambda x: x.timestamp)
    
    async def calculate_current_position(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate current position for a symbol based on trade history.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with current position information
        """
        trades = await self.get_position_trades(symbol)
        
        if not trades:
            return {
                'symbol': symbol,
                'quantity': 0,
                'avg_price': 0.0,
                'total_cost': 0.0,
                'trade_count': 0
            }
        
        total_quantity = 0
        total_cost = 0.0
        
        for trade in trades:
            if trade.action == 'buy':
                total_quantity += trade.quantity
                total_cost += trade.quantity * trade.price
            elif trade.action == 'sell':
                total_quantity -= trade.quantity
                total_cost -= trade.quantity * trade.price
        
        avg_price = total_cost / total_quantity if total_quantity != 0 else 0.0
        
        return {
            'symbol': symbol,
            'quantity': total_quantity,
            'avg_price': avg_price,
            'total_cost': total_cost,
            'trade_count': len(trades)
        }