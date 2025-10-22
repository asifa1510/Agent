"""
Lambda function for processing financial news data and scoring relevance/impact.
Processes data from the financial-news Kinesis stream.
"""

import json
import logging
import boto3
import base64
from typing import Dict, Any, List
from datetime import datetime
import os
import re

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
dynamodb = boto3.resource('dynamodb')
bedrock_runtime = boto3.client('bedrock-runtime')

# Configuration from environment variables
BERT_ENDPOINT = os.environ.get('BERT_ENDPOINT', 'sentiment-bert-endpoint')
NEWS_SCORES_TABLE = os.environ.get('NEWS_SCORES_TABLE', 'news-scores')
DLQ_QUEUE_URL = os.environ.get('DLQ_QUEUE_URL')

# Initialize DynamoDB table
news_table = dynamodb.Table(NEWS_SCORES_TABLE)

# Relevance threshold
RELEVANCE_THRESHOLD = 0.3

def lambda_handler(event, context):
    """
    Main Lambda handler for processing Kinesis records.
    
    Args:
        event: Kinesis event containing records
        context: Lambda context
        
    Returns:
        Dict with processing results
    """
    logger.info(f"Processing {len(event['Records'])} news records")
    
    processed_count = 0
    failed_count = 0
    filtered_count = 0
    failed_records = []
    
    for record in event['Records']:
        try:
            # Decode Kinesis data
            payload = base64.b64decode(record['kinesis']['data']).decode('utf-8')
            data = json.loads(payload)
            
            # Process the news record
            result = process_news_record(data)
            
            if result == 'processed':
                processed_count += 1
                logger.info(f"Successfully processed news for symbol: {data.get('symbol', 'unknown')}")
            elif result == 'filtered':
                filtered_count += 1
                logger.info(f"Filtered low relevance news for symbol: {data.get('symbol', 'unknown')}")
            else:
                failed_count += 1
                failed_records.append({
                    'record': data,
                    'error': 'Processing failed'
                })
                
        except Exception as e:
            failed_count += 1
            error_msg = f"Error processing record: {str(e)}"
            logger.error(error_msg)
            
            failed_records.append({
                'record': record,
                'error': error_msg
            })
    
    # Send failed records to DLQ if configured
    if failed_records and DLQ_QUEUE_URL:
        send_to_dlq(failed_records)
    
    logger.info(f"Processing complete: {processed_count} processed, {filtered_count} filtered, {failed_count} failed")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed': processed_count,
            'filtered': filtered_count,
            'failed': failed_count,
            'timestamp': datetime.utcnow().isoformat()
        })
    }

def process_news_record(data: Dict[str, Any]) -> str:
    """
    Process a single news record.
    
    Args:
        data: News data record
        
    Returns:
        str: 'processed', 'filtered', or 'failed'
    """
    try:
        # Validate required fields
        required_fields = ['symbol', 'title', 'content', 'source', 'timestamp']
        if not all(field in data for field in required_fields):
            logger.error(f"Missing required fields: {required_fields}")
            return 'failed'
        
        # Calculate relevance score
        relevance_score = calculate_relevance_score(data)
        
        # Filter out low relevance news
        if relevance_score < RELEVANCE_THRESHOLD:
            logger.info(f"Filtering news with low relevance: {relevance_score:.3f}")
            return 'filtered'
        
        # Calculate impact score
        impact_score = calculate_impact_score(data)
        
        # Perform sentiment analysis on news content
        sentiment_result = analyze_news_sentiment(data['title'] + ' ' + data['content'])
        
        # Prepare news score record
        news_record = {
            'symbol': data['symbol'],
            'timestamp': int(datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')).timestamp()),
            'relevance_score': relevance_score,
            'impact_score': impact_score,
            'sentiment_score': sentiment_result['score'] if sentiment_result else 0.0,
            'sentiment_confidence': sentiment_result['confidence'] if sentiment_result else 0.0,
            'source': data['source'],
            'title': data['title'],
            'url': data.get('url', ''),
            'content_preview': data['content'][:200],
            'processing_timestamp': int(datetime.utcnow().timestamp()),
            'ttl': int(datetime.utcnow().timestamp()) + (7 * 24 * 60 * 60)  # 7 days TTL
        }
        
        # Store in DynamoDB
        news_table.put_item(Item=news_record)
        
        logger.info(f"Stored news score for {data['symbol']}: relevance={relevance_score:.3f}, impact={impact_score:.3f}")
        
        return 'processed'
        
    except Exception as e:
        logger.error(f"Error processing news record: {e}")
        return 'failed'

def calculate_relevance_score(data: Dict[str, Any]) -> float:
    """
    Calculate relevance score for news article.
    
    Args:
        data: News data
        
    Returns:
        float: Relevance score (0.0 to 1.0)
    """
    try:
        symbol = data['symbol'].upper()
        title = data['title'].lower()
        content = data['content'].lower()
        
        # Company name mapping (simplified)
        company_names = {
            'AAPL': ['apple', 'iphone', 'ipad', 'mac', 'tim cook'],
            'GOOGL': ['google', 'alphabet', 'android', 'youtube', 'sundar pichai'],
            'MSFT': ['microsoft', 'windows', 'azure', 'office', 'satya nadella'],
            'TSLA': ['tesla', 'elon musk', 'electric vehicle', 'ev', 'model'],
            'AMZN': ['amazon', 'aws', 'prime', 'bezos', 'alexa']
        }
        
        # Base relevance from symbol mention
        relevance_score = 0.0
        
        # Direct symbol mention
        if symbol.lower() in title or symbol.lower() in content:
            relevance_score += 0.4
        
        # Company name mentions
        if symbol in company_names:
            for name in company_names[symbol]:
                if name in title:
                    relevance_score += 0.3
                elif name in content:
                    relevance_score += 0.2
        
        # Financial keywords boost
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'forecast',
            'stock', 'shares', 'market', 'trading', 'investment', 'analyst',
            'upgrade', 'downgrade', 'target price', 'dividend', 'split'
        ]
        
        keyword_count = sum(1 for keyword in financial_keywords if keyword in title or keyword in content)
        relevance_score += min(0.3, keyword_count * 0.05)
        
        # Source credibility boost
        credible_sources = ['reuters', 'bloomberg', 'wsj', 'cnbc', 'marketwatch', 'yahoo finance']
        source = data['source'].lower()
        
        if any(credible in source for credible in credible_sources):
            relevance_score += 0.1
        
        # Clamp to [0, 1]
        return min(1.0, max(0.0, relevance_score))
        
    except Exception as e:
        logger.error(f"Error calculating relevance score: {e}")
        return 0.0

def calculate_impact_score(data: Dict[str, Any]) -> float:
    """
    Calculate potential market impact score.
    
    Args:
        data: News data
        
    Returns:
        float: Impact score (0.0 to 1.0)
    """
    try:
        title = data['title'].lower()
        content = data['content'].lower()
        
        # High impact keywords
        high_impact_keywords = [
            'earnings', 'acquisition', 'merger', 'bankruptcy', 'lawsuit',
            'fda approval', 'recall', 'ceo', 'resignation', 'fired',
            'investigation', 'fraud', 'scandal', 'breakthrough'
        ]
        
        # Medium impact keywords
        medium_impact_keywords = [
            'revenue', 'guidance', 'forecast', 'upgrade', 'downgrade',
            'partnership', 'contract', 'launch', 'product', 'expansion'
        ]
        
        # Low impact keywords
        low_impact_keywords = [
            'conference', 'interview', 'statement', 'comment', 'opinion',
            'analysis', 'review', 'update', 'announcement'
        ]
        
        impact_score = 0.0
        
        # Check for high impact keywords
        for keyword in high_impact_keywords:
            if keyword in title:
                impact_score += 0.4
            elif keyword in content:
                impact_score += 0.2
        
        # Check for medium impact keywords
        for keyword in medium_impact_keywords:
            if keyword in title:
                impact_score += 0.2
            elif keyword in content:
                impact_score += 0.1
        
        # Check for low impact keywords
        for keyword in low_impact_keywords:
            if keyword in title:
                impact_score += 0.1
            elif keyword in content:
                impact_score += 0.05
        
        # Time sensitivity boost (recent news has higher impact)
        try:
            news_time = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            time_diff = datetime.utcnow() - news_time
            hours_old = time_diff.total_seconds() / 3600
            
            if hours_old < 1:
                impact_score += 0.2
            elif hours_old < 6:
                impact_score += 0.1
            elif hours_old < 24:
                impact_score += 0.05
        except:
            pass
        
        # Clamp to [0, 1]
        return min(1.0, max(0.0, impact_score))
        
    except Exception as e:
        logger.error(f"Error calculating impact score: {e}")
        return 0.0

def analyze_news_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment of news content.
    
    Args:
        text: News text to analyze
        
    Returns:
        Dict with sentiment score and confidence
    """
    try:
        # Try SageMaker BERT endpoint first
        input_data = {
            'inputs': text[:512],  # Limit text length for BERT
            'parameters': {
                'return_all_scores': True
            }
        }
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=BERT_ENDPOINT,
            ContentType='application/json',
            Body=json.dumps(input_data)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        if isinstance(result, list) and len(result) > 0:
            scores = result[0]
            
            negative_score = scores.get('NEGATIVE', 0)
            positive_score = scores.get('POSITIVE', 0)
            neutral_score = scores.get('NEUTRAL', 0)
            
            sentiment_score = positive_score - negative_score
            confidence = max(negative_score, positive_score, neutral_score)
            
            return {
                'score': sentiment_score,
                'confidence': confidence
            }
        
    except Exception as e:
        logger.warning(f"BERT sentiment analysis failed, using fallback: {e}")
    
    # Fallback to keyword-based analysis
    return fallback_news_sentiment(text)

def fallback_news_sentiment(text: str) -> Dict[str, float]:
    """
    Fallback sentiment analysis for news.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dict with sentiment score and confidence
    """
    positive_words = [
        'growth', 'profit', 'gain', 'rise', 'increase', 'strong', 'beat',
        'exceed', 'outperform', 'bullish', 'positive', 'success', 'win'
    ]
    
    negative_words = [
        'loss', 'decline', 'fall', 'drop', 'weak', 'miss', 'disappoint',
        'underperform', 'bearish', 'negative', 'fail', 'concern', 'risk'
    ]
    
    text_lower = text.lower()
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total_words = len(text.split())
    
    if total_words == 0:
        return {'score': 0.0, 'confidence': 0.0}
    
    sentiment_score = (positive_count - negative_count) / max(total_words, 1)
    sentiment_score = max(-1.0, min(1.0, sentiment_score * 10))
    
    confidence = min(1.0, (positive_count + negative_count) / max(total_words, 1) * 20)
    
    return {
        'score': sentiment_score,
        'confidence': max(0.1, confidence)
    }

def send_to_dlq(failed_records: List[Dict]) -> None:
    """Send failed records to Dead Letter Queue."""
    if not DLQ_QUEUE_URL:
        logger.warning("DLQ URL not configured, skipping failed record handling")
        return
    
    try:
        sqs = boto3.client('sqs')
        
        for failed_record in failed_records:
            message_body = {
                'source': 'news-processor',
                'timestamp': datetime.utcnow().isoformat(),
                'failed_record': failed_record
            }
            
            sqs.send_message(
                QueueUrl=DLQ_QUEUE_URL,
                MessageBody=json.dumps(message_body)
            )
        
        logger.info(f"Sent {len(failed_records)} failed records to DLQ")
        
    except Exception as e:
        logger.error(f"Error sending to DLQ: {e}")

# For local testing
if __name__ == "__main__":
    test_event = {
        'Records': [
            {
                'kinesis': {
                    'data': base64.b64encode(json.dumps({
                        'symbol': 'AAPL',
                        'title': 'Apple Reports Strong Q4 Earnings Beat',
                        'content': 'Apple Inc. reported quarterly earnings that beat analyst expectations...',
                        'source': 'reuters',
                        'url': 'https://example.com/news',
                        'timestamp': datetime.utcnow().isoformat()
                    }).encode()).decode()
                }
            }
        ]
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))