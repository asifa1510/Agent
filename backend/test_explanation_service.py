"""
Test script for the AWS Bedrock explanation service.
Tests the core functionality of trade and portfolio explanation generation.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any

from models.data_models import Trade, PortfolioPosition, RiskMetrics
from services.explanation_service import ExplanationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_trade() -> Trade:
    """Create a sample trade for testing."""
    return Trade(
        id="test-trade-001",
        symbol="AAPL",
        timestamp=int(datetime.now().timestamp()),
        action="buy",
        quantity=100,
        price=150.25,
        signal_strength=0.85,
        explanation_id=None
    )


def create_sample_portfolio() -> tuple[list[PortfolioPosition], RiskMetrics]:
    """Create sample portfolio data for testing."""
    positions = [
        PortfolioPosition(
            symbol="AAPL",
            quantity=100,
            avg_price=145.00,
            current_price=150.25,
            unrealized_pnl=525.00,
            allocation_percent=25.0,
            last_updated=int(datetime.now().timestamp())
        ),
        PortfolioPosition(
            symbol="GOOGL",
            quantity=50,
            avg_price=2800.00,
            current_price=2850.00,
            unrealized_pnl=2500.00,
            allocation_percent=35.0,
            last_updated=int(datetime.now().timestamp())
        ),
        PortfolioPosition(
            symbol="MSFT",
            quantity=75,
            avg_price=380.00,
            current_price=375.00,
            unrealized_pnl=-375.00,
            allocation_percent=20.0,
            last_updated=int(datetime.now().timestamp())
        )
    ]
    
    risk_metrics = RiskMetrics(
        total_value=400000.00,
        total_pnl=2650.00,
        max_drawdown=5.2,
        sharpe_ratio=1.45,
        volatility=18.5,
        var_95=12000.00,
        positions_count=3,
        timestamp=int(datetime.now().timestamp())
    )
    
    return positions, risk_metrics


def create_sample_context_data() -> Dict[str, Any]:
    """Create sample context data for testing."""
    return {
        'sentiment_data': {
            'score': 0.65,
            'confidence': 0.82,
            'volume': 1250,
            'source': 'twitter'
        },
        'prediction_data': {
            'model': 'lstm',
            'predicted_price': 155.50,
            'confidence_lower': 148.00,
            'confidence_upper': 163.00,
            'horizon': '3d'
        },
        'market_data': {
            'current_price': 150.25,
            'volume': 45000000,
            'price_change_percent': 2.1,
            'volatility': 22.5
        }
    }


async def test_trade_explanation():
    """Test trade explanation generation."""
    logger.info("Testing trade explanation generation...")
    
    try:
        service = ExplanationService()
        
        # Test connection first
        if not service.test_bedrock_connection():
            logger.warning("Bedrock connection test failed - using mock mode")
            return
        
        # Create sample data
        trade = create_sample_trade()
        context = create_sample_context_data()
        
        # Generate explanation
        explanation = await service.generate_trade_explanation(
            trade=trade,
            sentiment_data=context['sentiment_data'],
            prediction_data=context['prediction_data'],
            market_data=context['market_data']
        )
        
        logger.info("Trade explanation generated successfully:")
        logger.info(f"ID: {explanation.id}")
        logger.info(f"Confidence: {explanation.confidence:.2f}")
        logger.info(f"Explanation: {explanation.explanation[:200]}...")
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error testing trade explanation: {e}")
        return None


async def test_portfolio_explanation():
    """Test portfolio explanation generation."""
    logger.info("Testing portfolio explanation generation...")
    
    try:
        service = ExplanationService()
        
        # Test connection first
        if not service.test_bedrock_connection():
            logger.warning("Bedrock connection test failed - using mock mode")
            return
        
        # Create sample data
        positions, risk_metrics = create_sample_portfolio()
        
        # Generate explanation
        result = await service.generate_portfolio_explanation(
            positions=positions,
            risk_metrics=risk_metrics
        )
        
        logger.info("Portfolio explanation generated successfully:")
        logger.info(f"Confidence: {result['confidence']:.2f}")
        logger.info(f"Response time: {result['response_time_seconds']:.2f}s")
        logger.info(f"Explanation: {result['explanation'][:200]}...")
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing portfolio explanation: {e}")
        return None


async def test_batch_explanations():
    """Test batch explanation generation."""
    logger.info("Testing batch explanation generation...")
    
    try:
        service = ExplanationService()
        
        # Test connection first
        if not service.test_bedrock_connection():
            logger.warning("Bedrock connection test failed - using mock mode")
            return
        
        # Create multiple sample trades
        trades = []
        symbols = ["AAPL", "GOOGL", "MSFT"]
        actions = ["buy", "sell", "buy"]
        
        for i, (symbol, action) in enumerate(zip(symbols, actions)):
            trade = Trade(
                id=f"test-trade-{i+1:03d}",
                symbol=symbol,
                timestamp=int(datetime.now().timestamp()) + i * 60,
                action=action,
                quantity=100 + i * 50,
                price=150.0 + i * 10.0,
                signal_strength=0.7 + i * 0.1,
                explanation_id=None
            )
            trades.append(trade)
        
        # Generate batch explanations
        explanations = await service.batch_generate_explanations(
            trades=trades,
            include_context=True
        )
        
        logger.info(f"Batch explanations generated: {len(explanations)}/{len(trades)}")
        for exp in explanations:
            logger.info(f"- {exp.trade_id}: confidence {exp.confidence:.2f}")
        
        return explanations
        
    except Exception as e:
        logger.error(f"Error testing batch explanations: {e}")
        return None


async def test_service_status():
    """Test service status functionality."""
    logger.info("Testing service status...")
    
    try:
        service = ExplanationService()
        status = service.get_service_status()
        
        logger.info("Service status:")
        logger.info(json.dumps(status, indent=2))
        
        return status
        
    except Exception as e:
        logger.error(f"Error testing service status: {e}")
        return None


async def main():
    """Run all tests."""
    logger.info("Starting AWS Bedrock explanation service tests...")
    
    # Test service status
    await test_service_status()
    
    # Test individual components
    await test_trade_explanation()
    await test_portfolio_explanation()
    await test_batch_explanations()
    
    logger.info("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())