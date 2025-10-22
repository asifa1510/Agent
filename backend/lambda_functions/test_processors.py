#!/usr/bin/env python3
"""
Test script for Lambda data processors.
Tests each processor with sample data to ensure they work correctly.
"""

import json
import base64
import logging
import os
from datetime import datetime
from typing import Dict, Any

# Set AWS region for testing
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
os.environ['AWS_REGION'] = 'us-east-1'

# Set mock environment variables for testing
os.environ['BERT_ENDPOINT'] = 'test-bert-endpoint'
os.environ['SENTIMENT_TABLE'] = 'test-sentiment-scores'
os.environ['NEWS_SCORES_TABLE'] = 'test-news-scores'
os.environ['MARKET_DATA_TABLE'] = 'test-market-data'
os.environ['TECHNICAL_INDICATORS_TABLE'] = 'test-technical-indicators'
os.environ['DLQ_QUEUE_URL'] = 'https://sqs.us-east-1.amazonaws.com/123456789/test-dlq'

# Import the Lambda handlers after setting environment variables
from sentiment_processor import lambda_handler as sentiment_handler
from news_processor import lambda_handler as news_handler
from market_processor import lambda_handler as market_handler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_kinesis_event(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a mock Kinesis event for testing."""
    encoded_data = base64.b64encode(json.dumps(data).encode()).decode()
    
    return {
        'Records': [
            {
                'kinesis': {
                    'data': encoded_data,
                    'partitionKey': data.get('symbol', 'TEST'),
                    'sequenceNumber': '12345',
                    'approximateArrivalTimestamp': datetime.utcnow().timestamp()
                },
                'eventSource': 'aws:kinesis',
                'eventVersion': '1.0',
                'eventID': 'test-event-id',
                'eventName': 'aws:kinesis:record',
                'invokeIdentityArn': 'test-arn',
                'awsRegion': 'us-east-1',
                'eventSourceARN': 'test-stream-arn'
            }
        ]
    }

def test_sentiment_processor():
    """Test the sentiment processor Lambda function."""
    logger.info("Testing sentiment processor...")
    
    # Sample social media data
    sample_data = {
        'symbol': 'AAPL',
        'content': 'Apple stock is looking great today! Bullish trend continues. $AAPL to the moon!',
        'source': 'twitter',
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': 'test_user_123',
        'post_id': 'tweet_456'
    }
    
    event = create_kinesis_event(sample_data)
    
    try:
        result = sentiment_handler(event, None)
        logger.info(f"Sentiment processor result: {result}")
        
        # Validate result
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert 'processed' in body
        assert 'failed' in body
        assert 'timestamp' in body
        
        logger.info("✓ Sentiment processor test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Sentiment processor test failed: {e}")
        return False

def test_news_processor():
    """Test the news processor Lambda function."""
    logger.info("Testing news processor...")
    
    # Sample news data
    sample_data = {
        'symbol': 'AAPL',
        'title': 'Apple Reports Strong Q4 Earnings Beat',
        'content': 'Apple Inc. reported quarterly earnings that beat analyst expectations with revenue of $89.5 billion, up 8% year-over-year. The company showed strong iPhone sales and services growth.',
        'source': 'reuters',
        'url': 'https://example.com/apple-earnings',
        'timestamp': datetime.utcnow().isoformat(),
        'author': 'Financial Reporter'
    }
    
    event = create_kinesis_event(sample_data)
    
    try:
        result = news_handler(event, None)
        logger.info(f"News processor result: {result}")
        
        # Validate result
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert 'processed' in body
        assert 'filtered' in body
        assert 'failed' in body
        assert 'timestamp' in body
        
        logger.info("✓ News processor test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ News processor test failed: {e}")
        return False

def test_market_processor():
    """Test the market processor Lambda function."""
    logger.info("Testing market processor...")
    
    # Sample market data
    sample_data = {
        'symbol': 'AAPL',
        'price': 150.25,
        'volume': 1000000,
        'high': 151.0,
        'low': 149.5,
        'open': 150.0,
        'timestamp': datetime.utcnow().isoformat(),
        'exchange': 'NASDAQ'
    }
    
    event = create_kinesis_event(sample_data)
    
    try:
        result = market_handler(event, None)
        logger.info(f"Market processor result: {result}")
        
        # Validate result
        assert result['statusCode'] == 200
        body = json.loads(result['body'])
        assert 'processed' in body
        assert 'failed' in body
        assert 'timestamp' in body
        
        logger.info("✓ Market processor test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Market processor test failed: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid data."""
    logger.info("Testing error handling...")
    
    # Invalid data (missing required fields)
    invalid_data = {
        'symbol': 'AAPL'
        # Missing other required fields
    }
    
    event = create_kinesis_event(invalid_data)
    
    try:
        # Test sentiment processor with invalid data
        result = sentiment_handler(event, None)
        body = json.loads(result['body'])
        
        # Should have failed records
        assert body['failed'] > 0
        logger.info("✓ Error handling test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Error handling test failed: {e}")
        return False

def run_all_tests():
    """Run all Lambda processor tests."""
    logger.info("Starting Lambda processor tests...")
    
    tests = [
        ("Sentiment Processor", test_sentiment_processor),
        ("News Processor", test_news_processor),
        ("Market Processor", test_market_processor),
        ("Error Handling", test_error_handling)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} Test")
        logger.info(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        logger.info(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)