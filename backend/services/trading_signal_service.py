"""
Trading Signal Generation Service

Combines sentiment, news, and prediction signals to generate trading signals.
Implements signal strength calculation and trade execution logic with validation.

Requirements: 4.2, 4.4
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
import uuid

from ..models.data_models import Trade
from ..repositories.trades_repository import TradesRepository
from ..repositories.portfolio_repository import PortfolioRepository
from .sentiment_service import SentimentAggregationService
from .prediction_service import PredictionService
from .risk_controller import RiskController
from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class SignalWeights:
    """Signal weighting configuration"""
    sentiment_weight: float = 0.3
    news_weight: float = 0.2
    prediction_weight: float = 0.4
    technical_weight: float = 0.1

@dataclass
class TradingSignal:
    """Trading signal result"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    signal_strength: float  # 0-1
    confidence: float  # 0-1
    recommended_quantity: int
    price_target: float
    reasoning: Dict[str, Any]
    timestamp: int
    components: Dict[str, float]  # Individual signal components

@dataclass
class SignalComponents:
    """Individual signal components"""
    sentiment_score: float
    sentiment_confidence: float
    news_score: float
    news_confidence: float
    prediction_score: float
    prediction_confidence: float
    technical_score: float
    technical_confidence: float

class TradingSignalService:
    """
    Service for generating trading signals by combining multiple data sources.
    
    Combines sentiment analysis, news scoring, price predictions, and technical
    indicators to generate actionable trading signals with confidence scores.
    """
    
    def __init__(self):
        self.sentiment_service = SentimentAggregationService()
        self.prediction_service = PredictionService()
        self.risk_controller = RiskController()
        self.trades_repo = TradesRepository()
        self.portfolio_repo = PortfolioRepository()
        
        self.signal_weights = SignalWeights()
        self.min_signal_strength = 0.6  # Minimum signal strength for trade execution
        self.min_confidence = 0.5  # Minimum confidence for trade execution
    
    async def generate_trading_signal(
        self, 
        symbol: str,
        current_price: float,
        market_data: Optional[Dict[str, Any]] = None
    ) -> TradingSignal:
        """
        Generate comprehensive trading signal for a symbol.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            market_data: Additional market data for analysis
            
        Returns:
            TradingSignal with action recommendation and details
        """
        try:
            # Gather all signal components concurrently
            sentiment_task = self._get_sentiment_signal(symbol)
            news_task = self._get_news_signal(symbol)
            prediction_task = self._get_prediction_signal(symbol, current_price)
            technical_task = self._get_technical_signal(symbol, current_price, market_data)
            
            sentiment_data, news_data, prediction_data, technical_data = await asyncio.gather(
                sentiment_task, news_task, prediction_task, technical_task,
                return_exceptions=True
            )
            
            # Handle any exceptions in signal gathering
            if isinstance(sentiment_data, Exception):
                logger.error(f"Sentiment signal error: {sentiment_data}")
                sentiment_data = {'score': 0.0, 'confidence': 0.0}
            
            if isinstance(news_data, Exception):
                logger.error(f"News signal error: {news_data}")
                news_data = {'score': 0.0, 'confidence': 0.0}
            
            if isinstance(prediction_data, Exception):
                logger.error(f"Prediction signal error: {prediction_data}")
                prediction_data = {'score': 0.0, 'confidence': 0.0, 'price_target': current_price}
            
            if isinstance(technical_data, Exception):
                logger.error(f"Technical signal error: {technical_data}")
                technical_data = {'score': 0.0, 'confidence': 0.0}
            
            # Create signal components
            components = SignalComponents(
                sentiment_score=sentiment_data['score'],
                sentiment_confidence=sentiment_data['confidence'],
                news_score=news_data['score'],
                news_confidence=news_data['confidence'],
                prediction_score=prediction_data['score'],
                prediction_confidence=prediction_data['confidence'],
                technical_score=technical_data['score'],
                technical_confidence=technical_data['confidence']
            )
            
            # Calculate composite signal
            signal_result = self._calculate_composite_signal(
                symbol, current_price, components, prediction_data.get('price_target', current_price)
            )
            
            logger.info(f"Generated trading signal for {symbol}: {signal_result.action} "
                       f"(strength: {signal_result.signal_strength:.2f})")
            
            return signal_result
            
        except Exception as e:
            logger.error(f"Error generating trading signal for {symbol}: {e}")
            return self._create_neutral_signal(symbol, current_price, str(e))
    
    async def _get_sentiment_signal(self, symbol: str) -> Dict[str, float]:
        """Get sentiment-based trading signal."""
        try:
            # Get real-time sentiment (5 minutes)
            real_time = await self.sentiment_service.get_real_time_sentiment(symbol, minutes_back=5)
            
            # Get trend analysis
            trend = await self.sentiment_service.analyze_sentiment_trend(symbol, hours_back=24)
            
            # Calculate sentiment signal score
            sentiment_score = real_time['avg_score']  # -1 to 1
            sentiment_confidence = real_time['avg_confidence']
            
            # Adjust score based on trend
            if trend['trend'] == 'bullish':
                sentiment_score += 0.1 * trend['trend_strength']
            elif trend['trend'] == 'bearish':
                sentiment_score -= 0.1 * trend['trend_strength']
            
            # Normalize to 0-1 range (0.5 = neutral)
            normalized_score = (sentiment_score + 1) / 2
            
            # Adjust confidence based on volume
            volume_factor = min(real_time['total_volume'] / 100, 1.0)  # Scale by volume
            adjusted_confidence = sentiment_confidence * volume_factor
            
            return {
                'score': normalized_score,
                'confidence': adjusted_confidence,
                'raw_sentiment': sentiment_score,
                'trend': trend['trend'],
                'volume': real_time['total_volume']
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment signal for {symbol}: {e}")
            return {'score': 0.5, 'confidence': 0.0}
    
    async def _get_news_signal(self, symbol: str) -> Dict[str, float]:
        """Get news-based trading signal."""
        try:
            # This would integrate with news processing service
            # For now, return neutral signal as news service is not implemented
            # TODO: Implement news signal generation when news service is available
            
            return {
                'score': 0.5,  # Neutral
                'confidence': 0.0,
                'news_count': 0,
                'avg_relevance': 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting news signal for {symbol}: {e}")
            return {'score': 0.5, 'confidence': 0.0}
    
    async def _get_prediction_signal(self, symbol: str, current_price: float) -> Dict[str, float]:
        """Get prediction-based trading signal."""
        try:
            # Get predictions for multiple horizons
            input_data = {'symbol': symbol, 'current_price': current_price}
            predictions = await self.prediction_service.get_multiple_predictions(
                symbol, horizons=['1d', '3d', '7d'], input_data=input_data
            )
            
            # Calculate weighted prediction signal
            total_weight = 0
            weighted_score = 0
            weighted_confidence = 0
            price_targets = []
            
            horizon_weights = {'1d': 0.5, '3d': 0.3, '7d': 0.2}
            
            for horizon, prediction in predictions.items():
                if prediction and horizon in horizon_weights:
                    weight = horizon_weights[horizon]
                    
                    # Calculate signal based on predicted price change
                    price_change = (prediction.consensus_price - current_price) / current_price
                    
                    # Convert price change to signal score (0-1)
                    # Positive change -> higher score, negative change -> lower score
                    signal_score = 0.5 + (price_change * 2)  # Scale and center around 0.5
                    signal_score = max(0.0, min(1.0, signal_score))  # Clamp to 0-1
                    
                    weighted_score += signal_score * weight * prediction.confidence_score
                    weighted_confidence += prediction.confidence_score * weight
                    total_weight += weight
                    
                    price_targets.append(prediction.consensus_price)
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
                final_confidence = weighted_confidence / total_weight
                avg_price_target = sum(price_targets) / len(price_targets) if price_targets else current_price
            else:
                final_score = 0.5  # Neutral
                final_confidence = 0.0
                avg_price_target = current_price
            
            return {
                'score': final_score,
                'confidence': final_confidence,
                'price_target': avg_price_target,
                'predictions_count': len([p for p in predictions.values() if p is not None])
            }
            
        except Exception as e:
            logger.error(f"Error getting prediction signal for {symbol}: {e}")
            return {'score': 0.5, 'confidence': 0.0, 'price_target': current_price}
    
    async def _get_technical_signal(
        self, 
        symbol: str, 
        current_price: float,
        market_data: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Get technical analysis signal."""
        try:
            # Basic technical analysis using recent trade data
            recent_trades = await self.trades_repo.get_trades_by_symbol(symbol, limit=20)
            
            if len(recent_trades) < 5:
                return {'score': 0.5, 'confidence': 0.0}
            
            # Calculate simple moving average
            recent_prices = [trade.price for trade in recent_trades[-10:]]
            sma = sum(recent_prices) / len(recent_prices)
            
            # Calculate price momentum
            if len(recent_prices) >= 2:
                momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            else:
                momentum = 0.0
            
            # Generate technical signal
            # Above SMA and positive momentum = bullish
            # Below SMA and negative momentum = bearish
            price_vs_sma = (current_price - sma) / sma
            
            # Combine price position and momentum
            technical_score = 0.5 + (price_vs_sma * 0.3) + (momentum * 0.2)
            technical_score = max(0.0, min(1.0, technical_score))
            
            # Confidence based on data availability
            confidence = min(len(recent_trades) / 20, 1.0)
            
            return {
                'score': technical_score,
                'confidence': confidence,
                'sma': sma,
                'momentum': momentum,
                'trades_analyzed': len(recent_trades)
            }
            
        except Exception as e:
            logger.error(f"Error getting technical signal for {symbol}: {e}")
            return {'score': 0.5, 'confidence': 0.0}
    
    def _calculate_composite_signal(
        self, 
        symbol: str, 
        current_price: float,
        components: SignalComponents,
        price_target: float
    ) -> TradingSignal:
        """Calculate composite trading signal from all components."""
        try:
            # Calculate weighted signal strength
            signal_strength = (
                components.sentiment_score * self.signal_weights.sentiment_weight +
                components.news_score * self.signal_weights.news_weight +
                components.prediction_score * self.signal_weights.prediction_weight +
                components.technical_score * self.signal_weights.technical_weight
            )
            
            # Calculate weighted confidence
            confidence = (
                components.sentiment_confidence * self.signal_weights.sentiment_weight +
                components.news_confidence * self.signal_weights.news_weight +
                components.prediction_confidence * self.signal_weights.prediction_weight +
                components.technical_confidence * self.signal_weights.technical_weight
            )
            
            # Determine action based on signal strength
            if signal_strength > 0.6:
                action = 'buy'
            elif signal_strength < 0.4:
                action = 'sell'
            else:
                action = 'hold'
            
            # Calculate recommended quantity (placeholder logic)
            recommended_quantity = self._calculate_position_size(
                symbol, current_price, signal_strength, confidence
            )
            
            # Create reasoning dictionary
            reasoning = {
                'sentiment': {
                    'score': components.sentiment_score,
                    'confidence': components.sentiment_confidence,
                    'weight': self.signal_weights.sentiment_weight
                },
                'news': {
                    'score': components.news_score,
                    'confidence': components.news_confidence,
                    'weight': self.signal_weights.news_weight
                },
                'prediction': {
                    'score': components.prediction_score,
                    'confidence': components.prediction_confidence,
                    'weight': self.signal_weights.prediction_weight
                },
                'technical': {
                    'score': components.technical_score,
                    'confidence': components.technical_confidence,
                    'weight': self.signal_weights.technical_weight
                },
                'composite': {
                    'signal_strength': signal_strength,
                    'confidence': confidence,
                    'action': action
                }
            }
            
            # Component scores for analysis
            component_dict = {
                'sentiment': components.sentiment_score,
                'news': components.news_score,
                'prediction': components.prediction_score,
                'technical': components.technical_score
            }
            
            return TradingSignal(
                symbol=symbol,
                action=action,
                signal_strength=signal_strength,
                confidence=confidence,
                recommended_quantity=recommended_quantity,
                price_target=price_target,
                reasoning=reasoning,
                timestamp=int(datetime.now().timestamp()),
                components=component_dict
            )
            
        except Exception as e:
            logger.error(f"Error calculating composite signal: {e}")
            return self._create_neutral_signal(symbol, current_price, str(e))
    
    def _calculate_position_size(
        self, 
        symbol: str, 
        price: float, 
        signal_strength: float, 
        confidence: float
    ) -> int:
        """Calculate recommended position size based on signal strength and risk."""
        try:
            # Base position size (placeholder logic)
            base_size = 100  # Base number of shares
            
            # Adjust based on signal strength and confidence
            strength_multiplier = signal_strength if signal_strength > 0.5 else (1 - signal_strength)
            confidence_multiplier = confidence
            
            # Calculate adjusted size
            adjusted_size = int(base_size * strength_multiplier * confidence_multiplier)
            
            # Ensure minimum viable position
            return max(adjusted_size, 1)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1
    
    def _create_neutral_signal(self, symbol: str, current_price: float, error: str) -> TradingSignal:
        """Create neutral signal for error cases."""
        return TradingSignal(
            symbol=symbol,
            action='hold',
            signal_strength=0.5,
            confidence=0.0,
            recommended_quantity=0,
            price_target=current_price,
            reasoning={'error': error},
            timestamp=int(datetime.now().timestamp()),
            components={'sentiment': 0.5, 'news': 0.5, 'prediction': 0.5, 'technical': 0.5}
        )
    
    async def execute_trade_signal(
        self, 
        signal: TradingSignal,
        current_price: float,
        force_execute: bool = False
    ) -> Dict[str, Any]:
        """
        Execute a trading signal with risk validation.
        
        Args:
            signal: Trading signal to execute
            current_price: Current market price
            force_execute: Skip risk validation if True
            
        Returns:
            Dictionary with execution result
        """
        try:
            # Skip execution for hold signals
            if signal.action == 'hold':
                return {
                    'executed': False,
                    'reason': 'Hold signal - no action taken',
                    'signal': signal
                }
            
            # Check minimum signal strength and confidence
            if not force_execute:
                if signal.signal_strength < self.min_signal_strength:
                    return {
                        'executed': False,
                        'reason': f'Signal strength {signal.signal_strength:.2f} below minimum {self.min_signal_strength}',
                        'signal': signal
                    }
                
                if signal.confidence < self.min_confidence:
                    return {
                        'executed': False,
                        'reason': f'Confidence {signal.confidence:.2f} below minimum {self.min_confidence}',
                        'signal': signal
                    }
            
            # Validate trade with risk controller
            if not force_execute:
                risk_assessment = await self.risk_controller.validate_trade(
                    symbol=signal.symbol,
                    action=signal.action,
                    quantity=signal.recommended_quantity,
                    price=current_price,
                    signal_strength=signal.signal_strength
                )
                
                if not risk_assessment.approved:
                    return {
                        'executed': False,
                        'reason': 'Risk validation failed',
                        'risk_violations': risk_assessment.violations,
                        'risk_warnings': risk_assessment.warnings,
                        'recommended_size': risk_assessment.recommended_position_size,
                        'signal': signal
                    }
            
            # Create and execute trade
            trade = Trade(
                id=str(uuid.uuid4()),
                symbol=signal.symbol,
                timestamp=int(datetime.now().timestamp()),
                action=signal.action,
                quantity=signal.recommended_quantity,
                price=current_price,
                signal_strength=signal.signal_strength,
                explanation_id=None  # Will be set by explanation service
            )
            
            # Store trade in database
            success = await self.trades_repo.create_item(trade)
            
            if success:
                # Update portfolio position
                await self._update_portfolio_position(trade)
                
                logger.info(f"Executed trade: {trade.action} {trade.quantity} shares of {trade.symbol} at ${trade.price}")
                
                return {
                    'executed': True,
                    'trade': trade,
                    'signal': signal,
                    'execution_price': current_price
                }
            else:
                return {
                    'executed': False,
                    'reason': 'Failed to store trade in database',
                    'signal': signal
                }
            
        except Exception as e:
            logger.error(f"Error executing trade signal: {e}")
            return {
                'executed': False,
                'reason': f'Execution error: {str(e)}',
                'signal': signal
            }
    
    async def _update_portfolio_position(self, trade: Trade) -> None:
        """Update portfolio position after trade execution."""
        try:
            # Get current position
            current_position = await self.portfolio_repo.get_position(trade.symbol)
            
            if current_position:
                # Update existing position
                if trade.action == 'buy':
                    new_quantity = current_position.quantity + trade.quantity
                    # Calculate new average price
                    total_cost = (current_position.quantity * current_position.avg_price) + (trade.quantity * trade.price)
                    new_avg_price = total_cost / new_quantity if new_quantity != 0 else trade.price
                else:  # sell
                    new_quantity = current_position.quantity - trade.quantity
                    new_avg_price = current_position.avg_price  # Keep same average price
                
                await self.portfolio_repo.update_position(
                    symbol=trade.symbol,
                    quantity=new_quantity,
                    avg_price=new_avg_price,
                    current_price=trade.price
                )
            else:
                # Create new position
                if trade.action == 'buy':
                    await self.portfolio_repo.update_position(
                        symbol=trade.symbol,
                        quantity=trade.quantity,
                        avg_price=trade.price,
                        current_price=trade.price
                    )
            
            # Update allocation percentages
            await self.portfolio_repo.update_allocation_percentages()
            
        except Exception as e:
            logger.error(f"Error updating portfolio position: {e}")
    
    async def get_signal_summary(self, symbols: List[str]) -> Dict[str, TradingSignal]:
        """
        Get trading signals for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to their trading signals
        """
        try:
            # Get current prices (placeholder - would integrate with market data service)
            current_prices = {symbol: 100.0 for symbol in symbols}  # Placeholder prices
            
            # Generate signals for all symbols concurrently
            tasks = [
                self.generate_trading_signal(symbol, current_prices[symbol])
                for symbol in symbols
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Map results to symbols
            signals = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.error(f"Error generating signal for {symbol}: {result}")
                    signals[symbol] = self._create_neutral_signal(symbol, current_prices[symbol], str(result))
                else:
                    signals[symbol] = result
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting signal summary: {e}")
            return {}
    
    def update_signal_weights(self, new_weights: Dict[str, float]) -> bool:
        """Update signal weighting configuration."""
        try:
            for key, value in new_weights.items():
                if hasattr(self.signal_weights, key):
                    setattr(self.signal_weights, key, value)
                    logger.info(f"Updated signal weight {key} to {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating signal weights: {e}")
            return False
    
    def get_signal_config(self) -> Dict[str, Any]:
        """Get current signal generation configuration."""
        return {
            'signal_weights': {
                'sentiment_weight': self.signal_weights.sentiment_weight,
                'news_weight': self.signal_weights.news_weight,
                'prediction_weight': self.signal_weights.prediction_weight,
                'technical_weight': self.signal_weights.technical_weight
            },
            'execution_thresholds': {
                'min_signal_strength': self.min_signal_strength,
                'min_confidence': self.min_confidence
            }
        }