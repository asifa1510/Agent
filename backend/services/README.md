# External API Clients

This module provides robust API clients for collecting data from external sources with built-in retry logic, circuit breaker patterns, and rate limiting.

## Overview

The external API integration consists of:

- **BaseAPIClient**: Foundation class with retry logic and circuit breaker
- **TwitterClient**: Social media sentiment data from Twitter/X API
- **NewsClient**: Financial news from NewsAPI and RSS feeds
- **YahooFinanceClient**: Market data from Yahoo Finance
- **DataIntegrationService**: Coordinated data collection from all sources

## Features

✅ **Retry Logic**: Exponential backoff with configurable attempts  
✅ **Circuit Breaker**: Automatic failure detection and recovery  
✅ **Rate Limiting**: Respects API rate limits to avoid throttling  
✅ **Error Handling**: Comprehensive error handling and logging  
✅ **Async Support**: Full async/await support for concurrent operations  
✅ **Fallback Mechanisms**: RSS feeds when NewsAPI is unavailable  

## Quick Start

### Individual Clients

```python
import asyncio
from services import TwitterClient, NewsClient, YahooFinanceClient

async def example():
    # Twitter client
    async with TwitterClient() as twitter:
        tweets = await twitter.search_tweets("$AAPL", max_results=10)
        
    # News client
    async with NewsClient() as news:
        articles = await news.search_news("Apple stock", page_size=10)
        
    # Yahoo Finance client
    async with YahooFinanceClient() as yahoo:
        price = await yahoo.get_real_time_price("AAPL")
        
asyncio.run(example())
```

### Integrated Data Collection

```python
from services.data_integration import DataIntegrationService

async def collect_data():
    async with DataIntegrationService() as service:
        # Collect all data for symbols
        data = await service.collect_all_data_for_symbols(
            ['AAPL', 'GOOGL'],
            max_tweets_per_symbol=50,
            max_news_per_symbol=20
        )
        
        # Get market overview
        overview = await service.get_market_overview()
        
        # Health check
        health = await service.health_check()
```

## Configuration

### Environment Variables

```bash
# Required for Twitter API
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Optional for NewsAPI (falls back to RSS feeds)
NEWS_API_KEY=your_news_api_key

# Rate limits (requests per minute)
TWITTER_RATE_LIMIT=20
NEWS_API_RATE_LIMIT=30
YAHOO_FINANCE_RATE_LIMIT=60
```

### API Keys Setup

1. **Twitter API**: Get bearer token from [Twitter Developer Portal](https://developer.twitter.com/)
2. **NewsAPI**: Get free API key from [NewsAPI.org](https://newsapi.org/) (optional)
3. **Yahoo Finance**: No API key required

## API Client Details

### TwitterClient

Collects social media sentiment data from Twitter/X.

**Key Methods:**
- `search_tweets(query, max_results)`: Search tweets by query
- `get_tweets_by_symbols(symbols)`: Get tweets for stock symbols
- `get_trending_topics()`: Get trending topics

**Rate Limits:** 20 requests/minute (Twitter API v2 limits)

### NewsClient

Collects financial news from NewsAPI and RSS feeds.

**Key Methods:**
- `search_news(query, page_size)`: Search news articles
- `get_news_by_symbols(symbols)`: Get news for stock symbols
- `get_market_news(category)`: Get general market news

**Fallback:** Automatically uses RSS feeds when NewsAPI is unavailable

### YahooFinanceClient

Collects market data from Yahoo Finance.

**Key Methods:**
- `get_real_time_price(symbol)`: Get current price data
- `get_historical_data(symbol, period)`: Get historical prices
- `get_multiple_prices(symbols)`: Get prices for multiple symbols
- `get_financial_data(symbol)`: Get financial ratios and metrics

**No API Key Required:** Uses yfinance library

## Error Handling

All clients implement comprehensive error handling:

```python
from services.base_api_client import APIError, RateLimitError

try:
    async with TwitterClient() as client:
        data = await client.search_tweets("AAPL")
except RateLimitError as e:
    print(f"Rate limited: {e}")
except APIError as e:
    print(f"API error: {e}")
```

## Circuit Breaker

Automatic circuit breaker protection:
- **Failure Threshold**: 5 consecutive failures
- **Recovery Timeout**: 60 seconds
- **Automatic Recovery**: Resumes after timeout

## Testing

Run the test suite:

```bash
cd backend/services
python test_api_clients.py
```

Run the demo:

```bash
python demo_api_usage.py
```

## Data Formats

### Tweet Data
```python
{
    'id': 'tweet_id',
    'text': 'Tweet content',
    'created_at': '2024-01-01T12:00:00Z',
    'author_verified': True,
    'retweet_count': 10,
    'like_count': 50,
    'source': 'twitter'
}
```

### News Article Data
```python
{
    'id': 'article_id',
    'title': 'Article title',
    'description': 'Article description',
    'url': 'https://...',
    'source': 'Reuters',
    'published_at': '2024-01-01T12:00:00Z',
    'relevance_score': 0.85,
    'source_type': 'newsapi'
}
```

### Market Data
```python
{
    'symbol': 'AAPL',
    'price': 150.25,
    'change': 2.50,
    'change_percent': 1.69,
    'volume': 50000000,
    'market_cap': 2500000000000,
    'pe_ratio': 25.5,
    'timestamp': 1704110400
}
```

## Integration with Lambda Processors

These API clients are designed to work with the Lambda data processors:

- **sentiment-processor**: Uses TwitterClient for social media data
- **news-processor**: Uses NewsClient for financial news
- **market-processor**: Uses YahooFinanceClient for market data

## Performance Considerations

- **Concurrent Requests**: All clients support concurrent operations
- **Rate Limiting**: Built-in rate limiting prevents API throttling
- **Caching**: Consider implementing caching for frequently requested data
- **Batch Processing**: Use batch methods for multiple symbols

## Monitoring

All clients provide detailed logging:

```python
import logging
logging.basicConfig(level=logging.INFO)

# Logs include:
# - API request/response details
# - Rate limiting information
# - Error conditions
# - Performance metrics
```

## Requirements

See `backend/requirements.txt` for all dependencies:
- `tweepy`: Twitter API client
- `feedparser`: RSS feed parsing
- `yfinance`: Yahoo Finance data
- `aiohttp`: Async HTTP client
- `tenacity`: Retry logic
- `circuitbreaker`: Circuit breaker pattern