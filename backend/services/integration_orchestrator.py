"""
Integration orchestrator that connects all system components and manages end-to-end data flow.
This service coordinates data pipeline, ML processing, and backend services.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json

from .data_integration import DataIntegrationService
from .sentiment_service import SentimentService
from .prediction_service import PredictionService
from .trading_signal_service import TradingSignalService
from .explanation_service import ExplanationService
from .data_flow_coordinator import get_coordinator, PipelineStage, PipelineContext
from ..aws.kinesis_producer import KinesisDataProducer
from ..repositories.sentiment_repository import SentimentRepository
from ..repositories.predictions_repository import PredictionsRepository
from ..repositories.trades_repository import TradesRepository
from ..config import settings

logger = logging.getLogger(__name__)


class IntegrationOrchestrator:
    """Orchestrates the complete data flow from ingestion to trading decisions"""
    
    def __init__(self):
        self.data_integration = DataIntegrationService()
        self.kinesis_producer = KinesisDataProducer(region_name=settings.aws_region)
        
        # Initialize services
        self.sentiment_service = SentimentService()
        self.prediction_service = PredictionService()
        self.trading_signal_service = TradingSignalService()
        self.explanation_service = ExplanationService()
        
        # Initialize repositories
        self.sentiment_repo = SentimentRepository()
        self.predictions_repo = PredictionsRepository()
        self.trades_repo = TradesRepository()
        
        # Get data flow coordinator and register handlers
        self.coordinator = get_coordinator()
        self._register_pipeline_handlers()
        
        # Track processing metrics
        self.metrics = {
            'data_ingested': 0,
            'sentiment_processed': 0,
            'predictions_generated': 0,
            'trades_executed': 0,
            'errors': 0,
            'last_run': None
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.data_integration.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.data_integration.__aexit__(exc_type, exc_val, exc_tb)
    
    def _register_pipeline_handlers(self):
        """Register pipeline stage handlers with the coordinator"""
        self.coordinator.register_stage_handler(PipelineStage.DATA_INGESTION, self._handle_data_ingestion)
        self.coordinator.register_stage_handler(PipelineStage.LAMBDA_PROCESSING, self._handle_lambda_processing)
        self.coordinator.register_stage_handler(PipelineStage.ML_PREDICTION, self._handle_ml_prediction)
        self.coordinator.register_stage_handler(PipelineStage.TRADING_SIGNAL, self._handle_trading_signal)
        self.coordinator.register_stage_handler(PipelineStage.TRADE_EXECUTION, self._handle_trade_execution)
        self.coordinator.register_stage_handler(PipelineStage.EXPLANATION_GENERATION, self._handle_explanation_generation)
    
    async def run_complete_pipeline(
        self, 
        symbols: List[str],
        include_trading: bool = True,
        include_explanations: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete end-to-end pipeline for given symbols using the data flow coordinator
        
        Args:
            symbols: List of stock symbols to process
            include_trading: Whether to execute trading signals
            include_explanations: Whether to generate explanations
            
        Returns:
            Dict with pipeline execution results
        """
        logger.info(f"Starting coordinated pipeline for symbols: {symbols}")
        
        try:
            # Configure pipeline
            config = {
                'include_trading': include_trading,
                'include_explanations': include_explanations
            }
            
            # Execute pipeline through coordinator
            result = await self.coordinator.execute_pipeline(symbols, config)
            
            # Update local metrics
            if result['status'] == 'success':
                self.metrics['last_run'] = result['start_time']
                
                # Extract metrics from pipeline data
                if 'data_ingestion' in result['data']:
                    ingestion_data = result['data']['data_ingestion']
                    streaming_results = ingestion_data.get('streaming_results', {})
                    self.metrics['data_ingested'] += (
                        streaming_results.get('social_media_sent', 0) +
                        streaming_results.get('news_sent', 0) +
                        streaming_results.get('market_data_sent', 0)
                    )
                
                if 'ml_prediction' in result['data']:
                    predictions = result['data']['ml_prediction']
                    self.metrics['predictions_generated'] += sum(
                        len(pred_data) for pred_data in predictions.values() if isinstance(pred_data, dict)
                    )
                
                if 'trade_execution' in result['data']:
                    trades = result['data']['trade_execution']
                    self.metrics['trades_executed'] += sum(
                        1 for trade_data in trades.values() 
                        if isinstance(trade_data, dict) and trade_data.get('status') == 'executed'
                    )
            else:
                self.metrics['errors'] += 1
            
            return result
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Coordinated pipeline execution failed: {e}")
            
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'symbols_processed': symbols,
                'execution_time_seconds': 0,
                'metrics': self.metrics.copy()
            }
    
    async def _collect_and_stream_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Collect data and stream to Kinesis"""
        try:
            # Collect all data
            data = await self.data_integration.collect_all_data_for_symbols(
                symbols=symbols,
                max_tweets_per_symbol=20,
                max_news_per_symbol=10
            )
            
            # Stream data to Kinesis
            streaming_results = {
                'social_media_sent': 0,
                'news_sent': 0,
                'market_data_sent': 0,
                'errors': []
            }
            
            for symbol, symbol_data in data.items():
                # Stream social media data
                for tweet in symbol_data.get('social_media', []):
                    kinesis_data = {
                        'symbol': symbol,
                        'content': tweet.get('text', ''),
                        'source': 'twitter',
                        'timestamp': tweet.get('created_at', datetime.utcnow().isoformat()),
                        'metadata': {
                            'user_id': tweet.get('user_id'),
                            'retweet_count': tweet.get('retweet_count', 0),
                            'like_count': tweet.get('like_count', 0)
                        }
                    }
                    
                    if self.kinesis_producer.send_social_media_data(kinesis_data):
                        streaming_results['social_media_sent'] += 1
                    else:
                        streaming_results['errors'].append(f"Failed to send social media data for {symbol}")
                
                # Stream news data
                for article in symbol_data.get('news', []):
                    kinesis_data = {
                        'symbol': symbol,
                        'title': article.get('title', ''),
                        'content': article.get('description', ''),
                        'source': article.get('source', ''),
                        'timestamp': article.get('published_at', datetime.utcnow().isoformat()),
                        'url': article.get('url'),
                        'metadata': {
                            'author': article.get('author'),
                            'category': article.get('category')
                        }
                    }
                    
                    if self.kinesis_producer.send_financial_news_data(kinesis_data):
                        streaming_results['news_sent'] += 1
                    else:
                        streaming_results['errors'].append(f"Failed to send news data for {symbol}")
                
                # Stream market data
                market_data = symbol_data.get('market_data', {})
                if market_data:
                    kinesis_data = {
                        'symbol': symbol,
                        'price': market_data.get('current_price', 0),
                        'volume': market_data.get('volume', 0),
                        'timestamp': datetime.utcnow().isoformat(),
                        'metadata': {
                            'open': market_data.get('open'),
                            'high': market_data.get('high'),
                            'low': market_data.get('low'),
                            'previous_close': market_data.get('previous_close')
                        }
                    }
                    
                    if self.kinesis_producer.send_market_data(kinesis_data):
                        streaming_results['market_data_sent'] += 1
                    else:
                        streaming_results['errors'].append(f"Failed to send market data for {symbol}")
            
            self.metrics['data_ingested'] += streaming_results['social_media_sent'] + streaming_results['news_sent'] + streaming_results['market_data_sent']
            
            return {
                'raw_data_summary': {
                    symbol: {
                        'social_media_count': len(data[symbol].get('social_media', [])),
                        'news_count': len(data[symbol].get('news', [])),
                        'has_market_data': bool(data[symbol].get('market_data'))
                    }
                    for symbol in symbols
                },
                'streaming_results': streaming_results
            }
            
        except Exception as e:
            logger.error(f"Error in data collection and streaming: {e}")
            raise
    
    async def _generate_predictions(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate ML predictions for symbols"""
        try:
            predictions_results = {}
            
            for symbol in symbols:
                try:
                    # Generate predictions for different horizons
                    horizons = ['1d', '3d', '7d']
                    symbol_predictions = {}
                    
                    for horizon in horizons:
                        prediction = await self.prediction_service.generate_prediction(
                            symbol=symbol,
                            horizon=horizon
                        )
                        
                        if prediction:
                            symbol_predictions[horizon] = prediction
                            self.metrics['predictions_generated'] += 1
                    
                    predictions_results[symbol] = symbol_predictions
                    
                except Exception as e:
                    logger.error(f"Error generating predictions for {symbol}: {e}")
                    predictions_results[symbol] = {'error': str(e)}
            
            return predictions_results
            
        except Exception as e:
            logger.error(f"Error in prediction generation: {e}")
            raise
    
    async def _process_trading_signals(self, symbols: List[str]) -> Dict[str, Any]:
        """Process trading signals and execute trades"""
        try:
            trading_results = {}
            
            for symbol in symbols:
                try:
                    # Generate trading signal
                    signal = await self.trading_signal_service.generate_signal(symbol)
                    
                    if signal and signal.get('action') in ['buy', 'sell']:
                        # Execute trade (this would integrate with actual broker API)
                        trade_result = await self._execute_trade(symbol, signal)
                        trading_results[symbol] = trade_result
                        
                        if trade_result.get('status') == 'executed':
                            self.metrics['trades_executed'] += 1
                    else:
                        trading_results[symbol] = {'action': 'hold', 'reason': 'No strong signal'}
                        
                except Exception as e:
                    logger.error(f"Error processing trading signal for {symbol}: {e}")
                    trading_results[symbol] = {'error': str(e)}
            
            return trading_results
            
        except Exception as e:
            logger.error(f"Error in trading signal processing: {e}")
            raise
    
    async def _execute_trade(self, symbol: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on signal (simulation for now)"""
        try:
            # This is a simulation - in production, this would integrate with broker API
            trade_data = {
                'symbol': symbol,
                'action': signal['action'],
                'quantity': signal.get('quantity', 100),
                'price': signal.get('price', 0),
                'signal_strength': signal.get('strength', 0),
                'timestamp': datetime.utcnow().isoformat(),
                'status': 'executed'  # Simulated execution
            }
            
            # Store trade in repository
            trade_id = await self.trades_repo.create_trade(trade_data)
            trade_data['id'] = trade_id
            
            logger.info(f"Executed {signal['action']} trade for {symbol}: {signal.get('quantity', 100)} shares")
            
            return trade_data
            
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _generate_explanations(self, trading_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanations for executed trades"""
        try:
            explanations = {}
            
            for symbol, trade_result in trading_results.items():
                if trade_result.get('status') == 'executed':
                    try:
                        explanation = await self.explanation_service.generate_trade_explanation(
                            trade_result,
                            include_context=True
                        )
                        
                        if explanation:
                            explanations[symbol] = explanation
                        
                    except Exception as e:
                        logger.error(f"Error generating explanation for {symbol}: {e}")
                        explanations[symbol] = {'error': str(e)}
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error in explanation generation: {e}")
            raise
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            # Check data integration health
            data_health = await self.data_integration.health_check()
            
            # Check service health
            services_health = {
                'sentiment_service': await self._check_service_health(self.sentiment_service),
                'prediction_service': await self._check_service_health(self.prediction_service),
                'trading_signal_service': await self._check_service_health(self.trading_signal_service),
                'explanation_service': await self._check_service_health(self.explanation_service)
            }
            
            # Check repository health
            repositories_health = {
                'sentiment_repository': await self._check_repository_health(self.sentiment_repo),
                'predictions_repository': await self._check_repository_health(self.predictions_repo),
                'trades_repository': await self._check_repository_health(self.trades_repo)
            }
            
            # Determine overall health
            all_healthy = (
                data_health.get('overall_status') == 'healthy' and
                all(status.get('status') == 'healthy' for status in services_health.values()) and
                all(status.get('status') == 'healthy' for status in repositories_health.values())
            )
            
            return {
                'overall_status': 'healthy' if all_healthy else 'degraded',
                'timestamp': datetime.utcnow().isoformat(),
                'data_integration': data_health,
                'services': services_health,
                'repositories': repositories_health,
                'metrics': self.metrics.copy()
            }
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {
                'overall_status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def _check_service_health(self, service) -> Dict[str, Any]:
        """Check health of a service"""
        try:
            # Try to call a basic method if available
            if hasattr(service, 'health_check'):
                return await service.health_check()
            else:
                return {'status': 'healthy', 'message': 'Service available'}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    async def _check_repository_health(self, repository) -> Dict[str, Any]:
        """Check health of a repository"""
        try:
            # Try to perform a basic operation
            if hasattr(repository, 'health_check'):
                return await repository.health_check()
            else:
                return {'status': 'healthy', 'message': 'Repository available'}
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current processing metrics"""
        return {
            **self.metrics.copy(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def reset_metrics(self) -> None:
        """Reset processing metrics"""
        self.metrics = {
            'data_ingested': 0,
            'sentiment_processed': 0,
            'predictions_generated': 0,
            'trades_executed': 0,
            'errors': 0,
            'last_run': None
        }
        logger.info("Metrics reset")
    
    # Pipeline stage handlers
    async def _handle_data_ingestion(self, context: PipelineContext) -> Dict[str, Any]:
        """Handle data ingestion stage"""
        logger.info(f"Executing data ingestion for symbols: {context.symbols}")
        return await self._collect_and_stream_data(context.symbols)
    
    async def _handle_lambda_processing(self, context: PipelineContext) -> Dict[str, Any]:
        """Handle Lambda processing stage"""
        logger.info("Waiting for Lambda processing to complete")
        # Simulate Lambda processing time
        await asyncio.sleep(5)
        return {'status': 'completed', 'processing_time': 5}
    
    async def _handle_ml_prediction(self, context: PipelineContext) -> Dict[str, Any]:
        """Handle ML prediction stage"""
        logger.info(f"Generating ML predictions for symbols: {context.symbols}")
        return await self._generate_predictions(context.symbols)
    
    async def _handle_trading_signal(self, context: PipelineContext) -> Dict[str, Any]:
        """Handle trading signal generation stage"""
        if not context.config.get('include_trading', True):
            logger.info("Trading disabled, skipping signal generation")
            return {'status': 'skipped', 'reason': 'trading_disabled'}
        
        logger.info(f"Processing trading signals for symbols: {context.symbols}")
        return await self._process_trading_signals(context.symbols)
    
    async def _handle_trade_execution(self, context: PipelineContext) -> Dict[str, Any]:
        """Handle trade execution stage"""
        if not context.config.get('include_trading', True):
            logger.info("Trading disabled, skipping trade execution")
            return {'status': 'skipped', 'reason': 'trading_disabled'}
        
        # Get trading signals from previous stage
        trading_signals = context.data.get('trading_signal', {})
        if not trading_signals:
            logger.info("No trading signals available for execution")
            return {'status': 'skipped', 'reason': 'no_signals'}
        
        logger.info("Executing trades based on signals")
        # This stage is already handled in _process_trading_signals
        return trading_signals
    
    async def _handle_explanation_generation(self, context: PipelineContext) -> Dict[str, Any]:
        """Handle explanation generation stage"""
        if not context.config.get('include_explanations', True):
            logger.info("Explanations disabled, skipping generation")
            return {'status': 'skipped', 'reason': 'explanations_disabled'}
        
        # Get trading results from previous stage
        trading_results = context.data.get('trade_execution', {})
        if not trading_results:
            logger.info("No trading results available for explanation")
            return {'status': 'skipped', 'reason': 'no_trades'}
        
        logger.info("Generating trade explanations")
        return await self._generate_explanations(trading_results)


# Singleton instance for global use
_orchestrator_instance = None

async def get_orchestrator() -> IntegrationOrchestrator:
    """Get singleton orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = IntegrationOrchestrator()
        await _orchestrator_instance.__aenter__()
    return _orchestrator_instance