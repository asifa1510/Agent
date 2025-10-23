"""
Repository for portfolio data operations.
Handles DynamoDB operations for portfolio positions and risk metrics.
"""

import logging
from typing import List, Optional, Type, Dict, Any
from datetime import datetime

from .base_repository import BaseRepository
from ..models.data_models import PortfolioPosition, RiskMetrics
from ..config import settings

logger = logging.getLogger(__name__)

class PortfolioRepository(BaseRepository):
    """Repository for portfolio data."""
    
    def __init__(self):
        super().__init__(settings.portfolio_table)
    
    def get_model_class(self) -> Type[PortfolioPosition]:
        """Return the PortfolioPosition model class."""
        return PortfolioPosition
    
    async def get_position(self, symbol: str) -> Optional[PortfolioPosition]:
        """
        Get portfolio position for a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Portfolio position if found, None otherwise
        """
        return await self.get_item({'symbol': symbol})
    
    async def get_all_positions(self) -> List[PortfolioPosition]:
        """
        Get all current portfolio positions.
        
        Returns:
            List of all portfolio positions
        """
        return await self.scan_items()
    
    async def get_active_positions(self) -> List[PortfolioPosition]:
        """
        Get only active positions (quantity != 0).
        
        Returns:
            List of active portfolio positions
        """
        filter_expression = "quantity <> :zero"
        expression_values = {':zero': 0}
        
        return await self.scan_items(
            filter_expression=filter_expression,
            expression_values=expression_values
        )
    
    async def update_position(
        self, 
        symbol: str, 
        quantity: int, 
        avg_price: float,
        current_price: float
    ) -> bool:
        """
        Update or create a portfolio position.
        
        Args:
            symbol: Stock symbol
            quantity: Current quantity held
            avg_price: Average purchase price
            current_price: Current market price
            
        Returns:
            True if successful, False otherwise
        """
        unrealized_pnl = (current_price - avg_price) * quantity
        
        # Calculate allocation percentage (placeholder - needs total portfolio value)
        allocation_percent = 0.0  # TODO: Calculate based on total portfolio value
        
        position = PortfolioPosition(
            symbol=symbol,
            quantity=quantity,
            avg_price=avg_price,
            current_price=current_price,
            unrealized_pnl=unrealized_pnl,
            allocation_percent=allocation_percent,
            last_updated=int(datetime.now().timestamp())
        )
        
        return await self.create_item(position)
    
    async def calculate_portfolio_metrics(self) -> RiskMetrics:
        """
        Calculate portfolio-wide risk metrics.
        
        Returns:
            Risk metrics for the entire portfolio
        """
        positions = await self.get_active_positions()
        
        if not positions:
            return RiskMetrics(
                total_value=0.0,
                total_pnl=0.0,
                max_drawdown=0.0,
                sharpe_ratio=None,
                volatility=0.0,
                var_95=0.0,
                positions_count=0,
                timestamp=int(datetime.now().timestamp())
            )
        
        total_value = sum(pos.current_price * abs(pos.quantity) for pos in positions)
        total_pnl = sum(pos.unrealized_pnl for pos in positions)
        
        # TODO: Implement proper risk calculations
        # These are placeholder values
        max_drawdown = 0.0
        sharpe_ratio = None
        volatility = 0.0
        var_95 = 0.0
        
        return RiskMetrics(
            total_value=total_value,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility,
            var_95=var_95,
            positions_count=len(positions),
            timestamp=int(datetime.now().timestamp())
        )
    
    async def get_portfolio_performance(self) -> Dict[str, Any]:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        positions = await self.get_active_positions()
        
        if not positions:
            return {
                'total_return': 0.0,
                'total_return_percent': 0.0,
                'best_performer': None,
                'worst_performer': None,
                'positions_count': 0
            }
        
        total_cost = sum(pos.avg_price * abs(pos.quantity) for pos in positions)
        total_value = sum(pos.current_price * abs(pos.quantity) for pos in positions)
        total_return = total_value - total_cost
        total_return_percent = (total_return / total_cost * 100) if total_cost > 0 else 0.0
        
        # Find best and worst performers
        best_performer = None
        worst_performer = None
        best_return = float('-inf')
        worst_return = float('inf')
        
        for pos in positions:
            if pos.quantity != 0:
                return_percent = ((pos.current_price - pos.avg_price) / pos.avg_price * 100)
                if return_percent > best_return:
                    best_return = return_percent
                    best_performer = pos.symbol
                if return_percent < worst_return:
                    worst_return = return_percent
                    worst_performer = pos.symbol
        
        return {
            'total_return': total_return,
            'total_return_percent': total_return_percent,
            'best_performer': best_performer,
            'worst_performer': worst_performer,
            'positions_count': len(positions)
        }
    
    async def update_allocation_percentages(self) -> bool:
        """
        Update allocation percentages for all positions based on current values.
        
        Returns:
            True if successful, False otherwise
        """
        positions = await self.get_active_positions()
        
        if not positions:
            return True
        
        total_value = sum(pos.current_price * abs(pos.quantity) for pos in positions)
        
        if total_value == 0:
            return True
        
        try:
            for position in positions:
                position_value = position.current_price * abs(position.quantity)
                allocation_percent = (position_value / total_value) * 100
                
                await self.update_item(
                    key={'symbol': position.symbol},
                    updates={
                        'allocation_percent': allocation_percent,
                        'last_updated': int(datetime.now().timestamp())
                    }
                )
            
            return True
        except Exception as e:
            logger.error(f"Error updating allocation percentages: {e}")
            return False