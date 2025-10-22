"""
Explanation service for generating AI-powered trade and portfolio explanations.
Integrates with AWS Bedrock and manages explanation data persistence.
"""

import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models.data_models import Trade, TradeExplanation, PortfolioPosition, RiskMetrics
from ..config import settings

# Import Bedrock client from ml-models
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ml-models'))
from utils.bedrock_client import BedrockExplanationEngine

logger = logging.getLogger(__name__)


class ExplanationService:
    """
    Service for generating and managing trade and portfolio explanations.
    Handles context injection, explanation generation, and data persistence.
    """
    
    def __init__(self):
        """Initialize explanation service with Bedrock client."""
        try:
            self.bedrock_engine = BedrockExplanationEngine(
                region=settings.aws_region,
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                max_tokens=1000,
                temperature=0.3
            )
            logger.info("Explanation service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize explanation service: {e}")
            raise
    
    async def generate_trade_explanation(self,
                                       trade: Trade,
                                       sentiment_data: Optional[Dict[str, Any]] = None,
                                       prediction_data: Optional[Dict[str, Any]] = None,
                                       market_data: Optional[Dict[str, Any]] = None) -> TradeExplanation:
        """
        Generate explanation for a trade decision with context injection.
        
        Args:
            trade: Trade object containing trade details
            sentiment_data: Optional sentiment analysis results
            prediction_data: Optional price prediction results
            market_data: Optional market data and indicators
            
        Returns:
            TradeExplanation object with generated explanation
        """
        try:
            # Convert trade to dictionary for prompt generation
            trade_data = {
                'symbol': trade.symbol,
                'action': trade.action,
                'quantity': trade.quantity,
                'price': trade.price,
                'signal_strength': trade.signal_strength,
                'timestamp': trade.timestamp
            }
            
            # Generate explanation using Bedrock
            result = self.bedrock_engine.generate_trade_explanation(
                trade_data=trade_data,
                sentiment_data=sentiment_data,
                prediction_data=prediction_data,
                market_data=market_data
            )
            
            # Create TradeExplanation object
            explanation = TradeExplanation(
                id=str(uuid.uuid4()),
                trade_id=trade.id,
                explanation=result['explanation'],
                confidence=result['confidence'],
                supporting_data=result['supporting_data'],
                timestamp=result['timestamp']
            )
            
            logger.info(f"Generated explanation for trade {trade.id} with confidence {result['confidence']:.2f}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating trade explanation: {e}")
            raise
    
    async def generate_portfolio_explanation(self,
                                           positions: List[PortfolioPosition],
                                           risk_metrics: Optional[RiskMetrics] = None) -> Dict[str, Any]:
        """
        Generate explanation for portfolio performance and composition.
        
        Args:
            positions: List of current portfolio positions
            risk_metrics: Optional risk metrics for the portfolio
            
        Returns:
            Dictionary containing portfolio explanation and metadata
        """
        try:
            # Prepare portfolio data
            total_value = sum(pos.quantity * pos.current_price for pos in positions)
            total_pnl = sum(pos.unrealized_pnl for pos in positions)
            
            portfolio_data = {
                'total_value': total_value,
                'total_pnl': total_pnl,
                'positions_count': len(positions),
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'allocation_percent': pos.allocation_percent,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'current_price': pos.current_price
                    }
                    for pos in sorted(positions, key=lambda x: abs(x.unrealized_pnl), reverse=True)
                ]
            }
            
            # Prepare risk metrics data if available
            risk_data = None
            if risk_metrics:
                risk_data = {
                    'sharpe_ratio': risk_metrics.sharpe_ratio,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'volatility': risk_metrics.volatility,
                    'var_95': risk_metrics.var_95
                }
            
            # Generate explanation using Bedrock
            result = self.bedrock_engine.generate_portfolio_explanation(
                portfolio_data=portfolio_data,
                risk_metrics=risk_data
            )
            
            logger.info(f"Generated portfolio explanation with confidence {result['confidence']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating portfolio explanation: {e}")
            raise
    
    async def get_contextual_data(self, symbol: str) -> Dict[str, Any]:
        """
        Gather contextual data for explanation generation.
        
        Args:
            symbol: Stock symbol to gather context for
            
        Returns:
            Dictionary containing sentiment, prediction, and market data
        """
        context = {}
        
        try:
            # TODO: Integrate with actual data services
            # This would typically fetch from:
            # - Sentiment service for latest sentiment scores
            # - Prediction service for latest price predictions
            # - Market data service for current market information
            
            # Placeholder implementation
            context = {
                'sentiment_data': {
                    'score': 0.0,
                    'confidence': 0.0,
                    'volume': 0,
                    'source': 'twitter'
                },
                'prediction_data': {
                    'model': 'lstm',
                    'predicted_price': 0.0,
                    'confidence_lower': 0.0,
                    'confidence_upper': 0.0,
                    'horizon': '1d'
                },
                'market_data': {
                    'current_price': 0.0,
                    'volume': 0,
                    'price_change_percent': 0.0,
                    'volatility': 0.0
                }
            }
            
            logger.info(f"Gathered contextual data for {symbol}")
            
        except Exception as e:
            logger.warning(f"Error gathering contextual data for {symbol}: {e}")
            # Return empty context on error
            context = {}
        
        return context
    
    async def batch_generate_explanations(self,
                                        trades: List[Trade],
                                        include_context: bool = True) -> List[TradeExplanation]:
        """
        Generate explanations for multiple trades in batch.
        
        Args:
            trades: List of trades to generate explanations for
            include_context: Whether to include contextual data
            
        Returns:
            List of TradeExplanation objects
        """
        explanations = []
        
        for trade in trades:
            try:
                context_data = {}
                if include_context:
                    context_data = await self.get_contextual_data(trade.symbol)
                
                explanation = await self.generate_trade_explanation(
                    trade=trade,
                    sentiment_data=context_data.get('sentiment_data'),
                    prediction_data=context_data.get('prediction_data'),
                    market_data=context_data.get('market_data')
                )
                
                explanations.append(explanation)
                
            except Exception as e:
                logger.error(f"Error generating explanation for trade {trade.id}: {e}")
                # Continue with other trades even if one fails
                continue
        
        logger.info(f"Generated {len(explanations)} explanations from {len(trades)} trades")
        return explanations
    
    def test_bedrock_connection(self) -> bool:
        """
        Test connection to AWS Bedrock service.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            return self.bedrock_engine.test_connection()
        except Exception as e:
            logger.error(f"Bedrock connection test failed: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get current status of the explanation service.
        
        Returns:
            Dictionary containing service status information
        """
        try:
            bedrock_status = self.test_bedrock_connection()
            
            return {
                'service_name': 'ExplanationService',
                'status': 'healthy' if bedrock_status else 'degraded',
                'bedrock_connection': bedrock_status,
                'model_id': self.bedrock_engine.model_id,
                'region': self.bedrock_engine.region,
                'timestamp': int(datetime.now().timestamp())
            }
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {
                'service_name': 'ExplanationService',
                'status': 'error',
                'error': str(e),
                'timestamp': int(datetime.now().timestamp())
            }