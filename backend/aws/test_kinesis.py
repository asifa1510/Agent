"""
Test script for Kinesis stream configuration and basic functionality.
"""

import logging
import json
from datetime import datetime
from .config import aws_kinesis_config
from .kinesis_producer import send_twitter_sentiment, send_news_article, send_price_update

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_stream_creation():
    """Test creating all required streams."""
    logger.info("Testing stream creation...")
    
    try:
        results = aws_kinesis_config.initialize_streams()
        
        for stream_name, success in results.items():
            status = "SUCCESS" if success else "FAILED"
            logger.info(f"Stream {stream_name}: {status}")
        
        return all(results.values())
        
    except Exception as e:
        logger.error(f"Stream creation test failed: {e}")
        return False

def test_data_production():
    """Test sending sample data to streams."""
    logger.info("Testing data production...")
    
    try:
        # Test social media data
        twitter_success = send_twitter_sentiment(
            symbol="AAPL",
            content="Apple stock looking bullish today! Great earnings report.",
            source="twitter"
        )
        logger.info(f"Twitter data send: {'SUCCESS' if twitter_success else 'FAILED'}")
        
        # Test news data
        news_success = send_news_article(
            symbol="AAPL",
            title="Apple Reports Strong Q4 Earnings",
            content="Apple Inc. reported better than expected earnings for Q4...",
            source="reuters",
            url="https://example.com/news"
        )
        logger.info(f"News data send: {'SUCCESS' if news_success else 'FAILED'}")
        
        # Test market data
        market_success = send_price_update(
            symbol="AAPL",
            price=150.25,
            volume=1000000,
            additional_data={"high": 151.0, "low": 149.5}
        )
        logger.info(f"Market data send: {'SUCCESS' if market_success else 'FAILED'}")
        
        return twitter_success and news_success and market_success
        
    except Exception as e:
        logger.error(f"Data production test failed: {e}")
        return False

def test_stream_health():
    """Test stream health and status."""
    logger.info("Testing stream health...")
    
    try:
        manager = aws_kinesis_config.get_stream_manager()
        stream_names = aws_kinesis_config.get_all_stream_names()
        
        all_healthy = True
        
        for stream_name in stream_names:
            if manager.stream_exists(stream_name):
                info = manager.get_stream_info(stream_name)
                if info:
                    status = info['StreamStatus']
                    shard_count = len(info['Shards'])
                    logger.info(f"Stream {stream_name}: {status} ({shard_count} shards)")
                    
                    if status != 'ACTIVE':
                        all_healthy = False
                else:
                    logger.error(f"Could not get info for stream {stream_name}")
                    all_healthy = False
            else:
                logger.error(f"Stream {stream_name} does not exist")
                all_healthy = False
        
        return all_healthy
        
    except Exception as e:
        logger.error(f"Stream health test failed: {e}")
        return False

def run_all_tests():
    """Run all Kinesis tests."""
    logger.info("Starting Kinesis configuration tests...")
    
    tests = [
        ("Stream Creation", test_stream_creation),
        ("Stream Health", test_stream_health),
        ("Data Production", test_data_production)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"{test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("\n--- Test Results Summary ---")
    passed = 0
    for test_name, success in results.items():
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)