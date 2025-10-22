"""
Test script for external API clients
"""
import asyncio
import logging
from datetime import datetime

from data_integration import DataIntegrationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_individual_clients():
    """Test each API client individually"""
    print("Testing individual API clients...")
    
    # Test Twitter client
    print("\n1. Testing Twitter API...")
    try:
        from twitter_client import TwitterClient
        async with TwitterClient() as twitter:
            tweets = await twitter.search_tweets("AAPL", max_results=5)
            print(f"✓ Twitter API: Retrieved {len(tweets)} tweets")
            if tweets:
                print(f"  Sample tweet: {tweets[0]['text'][:100]}...")
    except Exception as e:
        print(f"✗ Twitter API error: {e}")
        
    # Test News client
    print("\n2. Testing News API...")
    try:
        from news_client import NewsClient
        async with NewsClient() as news:
            articles = await news.search_news("Apple stock", page_size=5)
            print(f"✓ News API: Retrieved {len(articles)} articles")
            if articles:
                print(f"  Sample article: {articles[0]['title']}")
    except Exception as e:
        print(f"✗ News API error: {e}")
        
    # Test Yahoo Finance client
    print("\n3. Testing Yahoo Finance API...")
    try:
        from yahoo_finance_client import YahooFinanceClient
        async with YahooFinanceClient() as yahoo:
            price_data = await yahoo.get_real_time_price("AAPL")
            print(f"✓ Yahoo Finance API: AAPL price = ${price_data['price']}")
    except Exception as e:
        print(f"✗ Yahoo Finance API error: {e}")


async def test_integration_service():
    """Test the integrated data collection service"""
    print("\n\nTesting integrated data collection...")
    
    try:
        async with DataIntegrationService() as service:
            # Test health check
            print("\n1. Health check...")
            health = await service.health_check()
            print(f"Overall status: {health['overall_status']}")
            for service_name, status in health['services'].items():
                print(f"  {service_name}: {status['status']}")
                
            # Test data collection for a single symbol
            print("\n2. Collecting data for AAPL...")
            data = await service.collect_all_data_for_symbols(
                ['AAPL'], 
                max_tweets_per_symbol=5,
                max_news_per_symbol=5
            )
            
            aapl_data = data.get('AAPL', {})
            print(f"✓ Social media posts: {len(aapl_data.get('social_media', []))}")
            print(f"✓ News articles: {len(aapl_data.get('news', []))}")
            print(f"✓ Market data available: {'Yes' if aapl_data.get('market_data') else 'No'}")
            
            if aapl_data.get('errors'):
                print(f"⚠ Errors: {aapl_data['errors']}")
                
            # Test market overview
            print("\n3. Getting market overview...")
            overview = await service.get_market_overview()
            print(f"✓ Market news: {overview['summary']['news_count']} articles")
            print(f"✓ Trending topics: {overview['summary']['trending_count']} topics")
            print(f"✓ Major indices: {overview['summary']['indices_available']} available")
            
    except Exception as e:
        print(f"✗ Integration service error: {e}")


async def main():
    """Main test function"""
    print("=" * 60)
    print("External API Clients Test Suite")
    print("=" * 60)
    
    await test_individual_clients()
    await test_integration_service()
    
    print("\n" + "=" * 60)
    print("Test suite completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())