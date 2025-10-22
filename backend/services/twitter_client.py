"""
Twitter API client for social media sentiment data collection
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import tweepy
from ..config import settings
from .base_api_client import BaseAPIClient, APIError

logger = logging.getLogger(__name__)


class TwitterClient(BaseAPIClient):
    """Twitter API client with rate limiting and error handling"""
    
    def __init__(self, bearer_token: Optional[str] = None):
        self.bearer_token = bearer_token or settings.twitter_bearer_token
        if not self.bearer_token:
            raise ValueError("Twitter bearer token is required")
            
        # Twitter API v2 rate limits: 300 requests per 15 minutes = 20 per minute
        super().__init__(
            base_url="https://api.twitter.com/2",
            api_key=self.bearer_token,
            rate_limit_per_minute=20
        )
        
        # Initialize tweepy client
        self.tweepy_client = tweepy.Client(bearer_token=self.bearer_token)
        
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get Twitter API authentication headers"""
        return {
            'Authorization': f'Bearer {self.bearer_token}'
        }
        
    async def search_tweets(
        self, 
        query: str, 
        max_results: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for tweets matching the query
        
        Args:
            query: Search query (e.g., "$AAPL OR Apple stock")
            max_results: Maximum number of tweets to return (10-100)
            start_time: Start time for search (optional)
            end_time: End time for search (optional)
            
        Returns:
            List of tweet data dictionaries
        """
        try:
            # Default to last hour if no time range specified
            if not start_time:
                start_time = datetime.utcnow() - timedelta(hours=1)
                
            # Prepare search parameters
            params = {
                'query': query,
                'max_results': min(max_results, 100),  # API limit
                'tweet.fields': 'created_at,author_id,public_metrics,context_annotations,lang',
                'user.fields': 'verified,public_metrics',
                'expansions': 'author_id'
            }
            
            if start_time:
                params['start_time'] = start_time.isoformat()
            if end_time:
                params['end_time'] = end_time.isoformat()
                
            # Make API request
            response = await self.get('tweets/search/recent', params=params)
            
            # Process response
            tweets = []
            if 'data' in response:
                users_lookup = {}
                if 'includes' in response and 'users' in response['includes']:
                    users_lookup = {user['id']: user for user in response['includes']['users']}
                
                for tweet in response['data']:
                    processed_tweet = self._process_tweet(tweet, users_lookup)
                    tweets.append(processed_tweet)
                    
            logger.info(f"Retrieved {len(tweets)} tweets for query: {query}")
            return tweets
            
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            raise APIError(f"Twitter search failed: {e}")
            
    def _process_tweet(self, tweet: Dict[str, Any], users_lookup: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw tweet data into standardized format"""
        author_id = tweet.get('author_id')
        user_data = users_lookup.get(author_id, {})
        
        return {
            'id': tweet['id'],
            'text': tweet['text'],
            'created_at': tweet['created_at'],
            'author_id': author_id,
            'author_verified': user_data.get('verified', False),
            'author_followers': user_data.get('public_metrics', {}).get('followers_count', 0),
            'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
            'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
            'reply_count': tweet.get('public_metrics', {}).get('reply_count', 0),
            'quote_count': tweet.get('public_metrics', {}).get('quote_count', 0),
            'lang': tweet.get('lang', 'en'),
            'context_annotations': tweet.get('context_annotations', []),
            'source': 'twitter'
        }
        
    async def get_tweets_by_symbols(self, symbols: List[str], max_results_per_symbol: int = 50) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get tweets for multiple stock symbols
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'GOOGL'])
            max_results_per_symbol: Max tweets per symbol
            
        Returns:
            Dictionary mapping symbols to tweet lists
        """
        results = {}
        
        for symbol in symbols:
            try:
                # Create search query for the symbol
                query = f"${symbol} OR {symbol} stock OR {symbol} shares -is:retweet lang:en"
                
                tweets = await self.search_tweets(
                    query=query,
                    max_results=max_results_per_symbol
                )
                
                results[symbol] = tweets
                logger.info(f"Retrieved {len(tweets)} tweets for {symbol}")
                
            except Exception as e:
                logger.error(f"Error getting tweets for {symbol}: {e}")
                results[symbol] = []
                
        return results
        
    async def get_trending_topics(self, woeid: int = 1) -> List[Dict[str, Any]]:
        """
        Get trending topics (requires Twitter API v1.1)
        
        Args:
            woeid: Where On Earth ID (1 = worldwide)
            
        Returns:
            List of trending topics
        """
        try:
            # Use tweepy for v1.1 API access
            trends = self.tweepy_client.get_place_trends(woeid)
            
            trending_topics = []
            if trends and len(trends) > 0:
                for trend in trends[0]['trends']:
                    trending_topics.append({
                        'name': trend['name'],
                        'url': trend['url'],
                        'tweet_volume': trend.get('tweet_volume'),
                        'query': trend.get('query')
                    })
                    
            logger.info(f"Retrieved {len(trending_topics)} trending topics")
            return trending_topics
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []