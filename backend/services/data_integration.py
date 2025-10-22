"""
Data integration service that coordinates all external API clients
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from .twitter_client import TwitterClient
from .news_client import NewsClient
from .yahoo_finance_client import YahooFinanceClient
from .base_api_client import APIError

logger = logging.getLogger(__name__)


class DataIntegrationService:
    """Service that coordinates data collection from all external APIs"""
    
    def __init__(self):
        self.twitter_client = TwitterClient()
        self.news_client = NewsClient()
        self.yahoo_client = YahooFinanceClient()
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.twitter_client.__aenter__()
        await self.news_client.__aenter__()
        await self.yahoo_client.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.twitter_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.news_client.__aexit__(exc_type, exc_val, exc_tb)
        await self.yahoo_client.__aexit__(exc_type, exc_val, exc_tb)
        
    async def collect_all_data_for_symbols(
        self, 
        symbols: List[str],
        include_social: bool = True,
        include_news: bool = True,
        include_market: bool = True,
        max_tweets_per_symbol: int = 50,
        max_news_per_symbol: int = 20
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect all available data for given symbols
        
        Args:
            symbols: List of stock symbols
            include_social: Whether to collect social media data
            include_news: Whether to collect news data
            include_market: Whether to collect market data
            max_tweets_per_symbol: Max tweets per symbol
            max_news_per_symbol: Max news articles per symbol
            
        Returns:
            Dictionary with comprehensive data for each symbol
        """
        results = {}
        
        # Initialize results structure
        for symbol in symbols:
            results[symbol] = {
                'symbol': symbol,
                'social_media': [],
                'news': [],
                'market_data': {},
                'timestamp': datetime.utcnow().isoformat(),
                'errors': []
            }
            
        # Collect data concurrently
        tasks = []
        
        if include_social:
            tasks.append(self._collect_social_data(symbols, max_tweets_per_symbol, results))
            
        if include_news:
            tasks.append(self._collect_news_data(symbols, max_news_per_symbol, results))
            
        if include_market:
            tasks.append(self._collect_market_data(symbols, results))
            
        # Execute all tasks concurrently
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log summary
        total_tweets = sum(len(data['social_media']) for data in results.values())
        total_news = sum(len(data['news']) for data in results.values())
        total_market = sum(1 for data in results.values() if data['market_data'])
        
        logger.info(f"Data collection complete: {total_tweets} tweets, {total_news} news articles, {total_market} market data points")
        
        return results
        
    async def _collect_social_data(self, symbols: List[str], max_per_symbol: int, results: Dict[str, Dict[str, Any]]):
        """Collect social media data for symbols"""
        try:
            social_data = await self.twitter_client.get_tweets_by_symbols(symbols, max_per_symbol)
            
            for symbol, tweets in social_data.items():
                if symbol in results:
                    results[symbol]['social_media'] = tweets
                    
        except Exception as e:
            logger.error(f"Error collecting social media data: {e}")
            for symbol in symbols:
                if symbol in results:
                    results[symbol]['errors'].append(f"Social media error: {str(e)}")
                    
    async def _collect_news_data(self, symbols: List[str], max_per_symbol: int, results: Dict[str, Dict[str, Any]]):
        """Collect news data for symbols"""
        try:
            news_data = await self.news_client.get_news_by_symbols(symbols, max_per_symbol)
            
            for symbol, articles in news_data.items():
                if symbol in results:
                    results[symbol]['news'] = articles
                    
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            for symbol in symbols:
                if symbol in results:
                    results[symbol]['errors'].append(f"News error: {str(e)}")
                    
    async def _collect_market_data(self, symbols: List[str], results: Dict[str, Dict[str, Any]]):
        """Collect market data for symbols"""
        try:
            market_data = await self.yahoo_client.get_multiple_prices(symbols)
            
            for symbol, price_data in market_data.items():
                if symbol in results and price_data:
                    results[symbol]['market_data'] = price_data
                    
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            for symbol in symbols:
                if symbol in results:
                    results[symbol]['errors'].append(f"Market data error: {str(e)}")
                    
    async def get_market_overview(self) -> Dict[str, Any]:
        """Get general market overview data"""
        try:
            # Get general market news
            market_news = await self.news_client.get_market_news(max_articles=30)
            
            # Get trending topics from Twitter
            trending_topics = await self.twitter_client.get_trending_topics()
            
            # Get data for major indices (if available)
            major_symbols = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
            index_data = await self.yahoo_client.get_multiple_prices(major_symbols)
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'market_news': market_news,
                'trending_topics': trending_topics,
                'major_indices': index_data,
                'summary': {
                    'news_count': len(market_news),
                    'trending_count': len(trending_topics),
                    'indices_available': len([d for d in index_data.values() if d])
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'market_news': [],
                'trending_topics': [],
                'major_indices': {},
                'summary': {'news_count': 0, 'trending_count': 0, 'indices_available': 0}
            }
            
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of all API clients"""
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'services': {}
        }
        
        # Test Twitter API
        try:
            await self.twitter_client.search_tweets("test", max_results=10)
            health_status['services']['twitter'] = {'status': 'healthy', 'error': None}
        except Exception as e:
            health_status['services']['twitter'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['overall_status'] = 'degraded'
            
        # Test News API
        try:
            await self.news_client.search_news("test", page_size=10)
            health_status['services']['news'] = {'status': 'healthy', 'error': None}
        except Exception as e:
            health_status['services']['news'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['overall_status'] = 'degraded'
            
        # Test Yahoo Finance API
        try:
            await self.yahoo_client.get_real_time_price("AAPL")
            health_status['services']['yahoo_finance'] = {'status': 'healthy', 'error': None}
        except Exception as e:
            health_status['services']['yahoo_finance'] = {'status': 'unhealthy', 'error': str(e)}
            health_status['overall_status'] = 'degraded'
            
        # Determine overall status
        unhealthy_count = sum(1 for service in health_status['services'].values() if service['status'] == 'unhealthy')
        if unhealthy_count == len(health_status['services']):
            health_status['overall_status'] = 'unhealthy'
        elif unhealthy_count > 0:
            health_status['overall_status'] = 'degraded'
            
        return health_status