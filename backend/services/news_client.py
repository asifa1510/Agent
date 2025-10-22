"""
News API client for financial news data collection
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import re
import time

import feedparser
import requests
from ..config import settings
from .base_api_client import BaseAPIClient, APIError

logger = logging.getLogger(__name__)


class NewsClient(BaseAPIClient):
    """News API client for financial news with RSS feed support"""   
 
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.news_api_key
        
        # NewsAPI rate limits: 1000 requests per day for free tier
        super().__init__(
            base_url="https://newsapi.org/v2",
            api_key=self.api_key,
            rate_limit_per_minute=30  # Conservative rate limiting
        )
        
        # RSS feed sources for financial news
        self.rss_feeds = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'https://www.marketwatch.com/rss/topstories',
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.reuters.com/business/finance/rss'
        ]
        
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get NewsAPI authentication headers"""
        if self.api_key:
            return {'X-API-Key': self.api_key}
        return {}
        
    async def search_news(
        self, 
        query: str, 
        language: str = 'en',
        sort_by: str = 'publishedAt',
        page_size: int = 100,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles using NewsAPI
        
        Args:
            query: Search query (e.g., "Apple stock" or "AAPL")
            language: Language code (default: 'en')
            sort_by: Sort order ('relevancy', 'popularity', 'publishedAt')
            page_size: Number of articles to return (max 100)
            from_date: Start date for search
            to_date: End date for search
            
        Returns:
            List of news article dictionaries
        """
        if not self.api_key:
            logger.warning("No NewsAPI key provided, falling back to RSS feeds")
            return await self._search_rss_feeds(query)
            
        try:
            params = {
                'q': query,
                'language': language,
                'sortBy': sort_by,
                'pageSize': min(page_size, 100)
            }
            
            if from_date:
                params['from'] = from_date.isoformat()
            if to_date:
                params['to'] = to_date.isoformat()
                
            response = await self.get('everything', params=params)
            
            articles = []
            if 'articles' in response:
                for article in response['articles']:
                    processed_article = self._process_newsapi_article(article, query)
                    if processed_article:
                        articles.append(processed_article)
                        
            logger.info(f"Retrieved {len(articles)} articles for query: {query}")
            return articles
            
        except Exception as e:
            logger.error(f"NewsAPI search failed: {e}")
            # Fallback to RSS feeds
            return await self._search_rss_feeds(query)
            
    async def _search_rss_feeds(self, query: str) -> List[Dict[str, Any]]:
        """Fallback method to search RSS feeds when NewsAPI is unavailable"""
        articles = []
        query_terms = query.lower().split()
        
        for feed_url in self.rss_feeds:
            try:
                # Parse RSS feed
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:  # Limit per feed
                    # Simple relevance check
                    title_lower = entry.title.lower()
                    description_lower = getattr(entry, 'description', '').lower()
                    
                    relevance_score = 0
                    for term in query_terms:
                        if term in title_lower:
                            relevance_score += 2
                        if term in description_lower:
                            relevance_score += 1
                            
                    if relevance_score > 0:
                        article = self._process_rss_article(entry, feed_url, relevance_score)
                        articles.append(article)
                        
            except Exception as e:
                logger.error(f"Error parsing RSS feed {feed_url}: {e}")
                continue
                
        # Sort by relevance score
        articles.sort(key=lambda x: x['relevance_score'], reverse=True)
        logger.info(f"Retrieved {len(articles)} articles from RSS feeds for query: {query}")
        return articles[:50]  # Return top 50 most relevant
        
    def _process_newsapi_article(self, article: Dict[str, Any], query: str) -> Optional[Dict[str, Any]]:
        """Process NewsAPI article into standardized format"""
        try:
            # Calculate relevance score
            relevance_score = self._calculate_relevance(
                article.get('title', ''), 
                article.get('description', ''), 
                query
            )
            
            # Filter out low relevance articles
            if relevance_score < 0.3:
                return None
                
            return {
                'id': f"newsapi_{hash(article.get('url', ''))}",
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', ''),
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'author': article.get('author', ''),
                'published_at': article.get('publishedAt', ''),
                'url_to_image': article.get('urlToImage', ''),
                'relevance_score': relevance_score,
                'source_type': 'newsapi'
            }
            
        except Exception as e:
            logger.error(f"Error processing NewsAPI article: {e}")
            return None
            
    def _process_rss_article(self, entry: Any, feed_url: str, relevance_score: int) -> Dict[str, Any]:
        """Process RSS feed entry into standardized format"""
        return {
            'id': f"rss_{hash(entry.link)}",
            'title': entry.title,
            'description': getattr(entry, 'description', ''),
            'content': getattr(entry, 'content', [{}])[0].get('value', '') if hasattr(entry, 'content') else '',
            'url': entry.link,
            'source': feed_url.split('/')[2],  # Extract domain
            'author': getattr(entry, 'author', ''),
            'published_at': getattr(entry, 'published', ''),
            'url_to_image': '',
            'relevance_score': min(relevance_score / 10.0, 1.0),  # Normalize to 0-1
            'source_type': 'rss'
        }
        
    def _calculate_relevance(self, title: str, description: str, query: str) -> float:
        """Calculate relevance score for an article"""
        if not title and not description:
            return 0.0
            
        text = f"{title} {description}".lower()
        query_terms = query.lower().split()
        
        score = 0.0
        total_terms = len(query_terms)
        
        for term in query_terms:
            # Exact matches in title get higher weight
            if term in title.lower():
                score += 0.4
            # Matches in description get lower weight
            elif term in description.lower():
                score += 0.2
                
        # Normalize by number of query terms
        return min(score / total_terms if total_terms > 0 else 0.0, 1.0)
        
    async def get_news_by_symbols(self, symbols: List[str], max_articles_per_symbol: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get news articles for multiple stock symbols
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
            max_articles_per_symbol: Max articles per symbol
            
        Returns:
            Dictionary mapping symbols to article lists
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Create search queries for the symbol
                queries = [
                    f"{symbol} stock",
                    f"{symbol} earnings",
                    f"{symbol} financial"
                ]
                
                all_articles = []
                for query in queries:
                    articles = await self.search_news(
                        query=query,
                        page_size=max_articles_per_symbol // len(queries)
                    )
                    all_articles.extend(articles)
                    
                # Remove duplicates and sort by relevance
                unique_articles = {}
                for article in all_articles:
                    url = article.get('url', '')
                    if url and url not in unique_articles:
                        unique_articles[url] = article
                        
                sorted_articles = sorted(
                    unique_articles.values(),
                    key=lambda x: x['relevance_score'],
                    reverse=True
                )
                
                results[symbol] = sorted_articles[:max_articles_per_symbol]
                logger.info(f"Retrieved {len(results[symbol])} articles for {symbol}")
                
            except Exception as e:
                logger.error(f"Error getting news for {symbol}: {e}")
                results[symbol] = []
                
        return results
        
    async def get_market_news(self, category: str = 'business', max_articles: int = 50) -> List[Dict[str, Any]]:
        """
        Get general market news
        
        Args:
            category: News category ('business', 'technology', etc.)
            max_articles: Maximum number of articles to return
            
        Returns:
            List of market news articles
        """
        try:
            if self.api_key:
                # Use NewsAPI for general market news
                params = {
                    'category': category,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': min(max_articles, 100)
                }
                
                response = await self.get('top-headlines', params=params)
                
                articles = []
                if 'articles' in response:
                    for article in response['articles']:
                        processed_article = self._process_newsapi_article(article, 'market news')
                        if processed_article:
                            articles.append(processed_article)
                            
                return articles
            else:
                # Fallback to RSS feeds
                return await self._search_rss_feeds('market news')
                
        except Exception as e:
            logger.error(f"Error getting market news: {e}")
            return []