"""
Demonstration script showing how to use the external API clients
"""
import asyncio
import os
from datetime import datetime, timedelta

# Set up basic environment for demo
os.environ.setdefault('TWITTER_BEARER_TOKEN', 'demo_token')
os.environ.setdefault('NEWS_API_KEY', 'demo_key')

from data_integration import DataIntegrationService
from twitter_client import TwitterClient
from news_client import NewsClient
from yahoo_finance_client import YahooFinanceClient


async def demo_twitter_client():
    """Demonstrate Twitter API client usage"""
    print("🐦 Twitter API Client Demo")
    print("-" * 40)
    
    try:
        async with TwitterClient() as twitter:
            # Search for tweets about Apple stock
            tweets = await twitter.search_tweets("$AAPL", max_results=5)
            print(f"Found {len(tweets)} tweets about AAPL")
            
            # Get tweets for multiple symbols
            symbols = ['AAPL', 'GOOGL']
            multi_tweets = await twitter.get_tweets_by_symbols(symbols, max_results_per_symbol=3)
            
            for symbol, symbol_tweets in multi_tweets.items():
                print(f"{symbol}: {len(symbol_tweets)} tweets")
                
    except Exception as e:
        print(f"Twitter demo error (expected without real API key): {e}")


async def demo_news_client():
    """Demonstrate News API client usage"""
    print("\n📰 News API Client Demo")
    print("-" * 40)
    
    try:
        async with NewsClient() as news:
            # Search for news about Apple
            articles = await news.search_news("Apple stock", page_size=5)
            print(f"Found {len(articles)} news articles about Apple")
            
            # Get news for multiple symbols
            symbols = ['AAPL', 'TSLA']
            multi_news = await news.get_news_by_symbols(symbols, max_articles_per_symbol=3)
            
            for symbol, symbol_articles in multi_news.items():
                print(f"{symbol}: {len(symbol_articles)} articles")
                if symbol_articles:
                    print(f"  Top article: {symbol_articles[0]['title']}")
                    
    except Exception as e:
        print(f"News demo error: {e}")


async def demo_yahoo_finance_client():
    """Demonstrate Yahoo Finance API client usage"""
    print("\n📈 Yahoo Finance API Client Demo")
    print("-" * 40)
    
    try:
        async with YahooFinanceClient() as yahoo:
            # Get real-time price for Apple
            price_data = await yahoo.get_real_time_price("AAPL")
            print(f"AAPL current price: ${price_data['price']:.2f}")
            print(f"Change: {price_data['change_percent']:.2f}%")
            
            # Get prices for multiple symbols
            symbols = ['AAPL', 'GOOGL', 'TSLA']
            multi_prices = await yahoo.get_multiple_prices(symbols)
            
            print("\nMultiple stock prices:")
            for symbol, data in multi_prices.items():
                if data:
                    print(f"  {symbol}: ${data['price']:.2f} ({data['change_percent']:+.2f}%)")
                else:
                    print(f"  {symbol}: No data available")
                    
            # Get historical data
            historical = await yahoo.get_historical_data("AAPL", period="5d")
            print(f"\nHistorical data points: {len(historical)}")
            if historical:
                latest = historical[-1]
                print(f"Latest close: ${latest['close']:.2f} on {latest['date'][:10]}")
                
    except Exception as e:
        print(f"Yahoo Finance demo error: {e}")


async def demo_integration_service():
    """Demonstrate the integrated data collection service"""
    print("\n🔄 Data Integration Service Demo")
    print("-" * 40)
    
    try:
        async with DataIntegrationService() as service:
            # Health check
            health = await service.health_check()
            print(f"Service health: {health['overall_status']}")
            
            # Collect comprehensive data for a symbol
            symbols = ['AAPL']
            data = await service.collect_all_data_for_symbols(
                symbols,
                max_tweets_per_symbol=3,
                max_news_per_symbol=3
            )
            
            aapl_data = data.get('AAPL', {})
            print(f"\nComprehensive data for AAPL:")
            print(f"  Social media posts: {len(aapl_data.get('social_media', []))}")
            print(f"  News articles: {len(aapl_data.get('news', []))}")
            print(f"  Market data: {'Available' if aapl_data.get('market_data') else 'Not available'}")
            
            if aapl_data.get('market_data'):
                market = aapl_data['market_data']
                print(f"  Current price: ${market['price']:.2f}")
                
            # Market overview
            overview = await service.get_market_overview()
            print(f"\nMarket overview:")
            print(f"  News articles: {overview['summary']['news_count']}")
            print(f"  Trending topics: {overview['summary']['trending_count']}")
            print(f"  Major indices: {overview['summary']['indices_available']}")
            
    except Exception as e:
        print(f"Integration service demo error: {e}")


async def main():
    """Run all demonstrations"""
    print("🚀 External API Clients Demonstration")
    print("=" * 50)
    print("Note: Some demos may show errors without real API keys")
    print("=" * 50)
    
    await demo_twitter_client()
    await demo_news_client()
    await demo_yahoo_finance_client()
    await demo_integration_service()
    
    print("\n✅ Demo completed!")
    print("\nTo use with real data:")
    print("1. Set TWITTER_BEARER_TOKEN environment variable")
    print("2. Set NEWS_API_KEY environment variable (optional)")
    print("3. Yahoo Finance works without API key")


if __name__ == "__main__":
    asyncio.run(main())