"""
Lambda function for processing social media data and performing sentiment analysis.
Processes data from the social-media Kinesis stream.
"""

import json
import logging
import boto3
import base64
from typing import Dict, Any, List
from datetime import datetime
import os

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
dynamodb = boto3.resource('dynamodb')

# Configuration from environment variables
BERT_ENDPOINT = os.environ.get('BERT_ENDPOINT', 'sentiment-bert-endpoint')
SENTIMENT_TABLE = os.environ.get('SENTIMENT_TABLE', 'sentiment-scores')
DLQ_QUEUE_URL = os.environ.get('DLQ_QUEUE_URL')

# Initialize DynamoDB table
sentiment_table = dynamodb.Table(SENTIMENT_TABLE)

def lambda_handler(event, context):
    """
    Main Lambda handler for processing Kinesis records.
    
    Args:
        event: Kinesis event containing records
        context: Lambda context
        
    Returns:
        Dict with processing results
    """
    logger.info(f"Processing {len(event['Records'])} records")
    
    processed_count = 0
    failed_count = 0
    failed_records = []
    
    for record in event['Records']:
        try:
            # Decode Kinesis data
            payload = base64.b64decode(record['kinesis']['data']).decode('utf-8')
            data = json.loads(payload)
            
            # Process the social media data
            result = process_social_media_record(data)
            
            if result:
                processed_count += 1
                logger.info(f"Successfully processed record for symbol: {data.get('symbol', 'unknown')}")
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
    
    logger.info(f"Processing complete: {processed_count} success, {failed_count} failed")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'processed': processed_count,
            'failed': failed_count,
            'timestamp': datetime.utcnow().isoformat()
        })
    }

def process_social_media_record(data: Dict[str, Any]) -> bool:
    """
    Process a single social media record.
    
    Args:
        data: Social media data record
        
    Returns:
        bool: True if processed successfully
    """
    try:
        # Validate required fields
        required_fields = ['symbol', 'content', 'source', 'timestamp']
        if not all(field in data for field in required_fields):
            logger.error(f"Missing required fields: {required_fields}")
            return False
        
        # Perform sentiment analysis
        sentiment_result = analyze_sentiment(data['content'])
        
        if not sentiment_result:
            logger.error("Sentiment analysis failed")
            return False
        
        # Prepare sentiment score record
        sentiment_record = {
            'symbol': data['symbol'],
            'timestamp': int(datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')).timestamp()),
            'score': sentiment_result['score'],
            'confidence': sentiment_result['confidence'],
            'volume': 1,  # Single post
            'source': data['source'],
            'content_preview': data['content'][:100],  # Store preview for debugging
            'processing_timestamp': int(datetime.utcnow().timestamp()),
            'ttl': int(datetime.utcnow().timestamp()) + (30 * 24 * 60 * 60)  # 30 days TTL
        }
        
        # Store in DynamoDB
        sentiment_table.put_item(Item=sentiment_record)
        
        logger.info(f"Stored sentiment for {data['symbol']}: score={sentiment_result['score']:.3f}, confidence={sentiment_result['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing social media record: {e}")
        return False

def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Analyze sentiment using SageMaker BERT endpoint.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Dict with sentiment score and confidence
    """
    try:
        # Prepare input for BERT model
        input_data = {
            'inputs': text,
            'parameters': {
                'return_all_scores': True
            }
        }
        
        # Call SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=BERT_ENDPOINT,
            ContentType='application/json',
            Body=json.dumps(input_data)
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        
        # Extract sentiment score and confidence
        # Assuming BERT returns scores for [negative, neutral, positive]
        if isinstance(result, list) and len(result) > 0:
            scores = result[0]
            
            # Convert to sentiment score (-1 to 1)
            negative_score = scores.get('NEGATIVE', 0)
            positive_score = scores.get('POSITIVE', 0)
            neutral_score = scores.get('NEUTRAL', 0)
            
            # Calculate overall sentiment score
            sentiment_score = positive_score - negative_score
            
            # Confidence is the maximum score
            confidence = max(negative_score, positive_score, neutral_score)
            
            return {
                'score': sentiment_score,
                'confidence': confidence
            }
        else:
            logger.error(f"Unexpected BERT response format: {result}")
            return None
            
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        
        # Fallback to simple keyword-based sentiment
        return fallback_sentiment_analysis(text)

def fallback_sentiment_analysis(text: str) -> Dict[str, float]:
    """
    Fallback sentiment analysis using simple keyword matching.
    
    Args:
        text: Text to analyze
        
    Returns:
        Dict with sentiment score and confidence
    """
    positive_words = ['good', 'great', 'excellent', 'bullish', 'up', 'rise', 'gain', 'profit', 'buy']
    negative_words = ['bad', 'terrible', 'bearish', 'down', 'fall', 'loss', 'sell', 'crash', 'drop']
    
    text_lower = text.lower()
    
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    total_words = len(text.split())
    
    if total_words == 0:
        return {'score': 0.0, 'confidence': 0.0}
    
    # Calculate sentiment score
    sentiment_score = (positive_count - negative_count) / max(total_words, 1)
    sentiment_score = max(-1.0, min(1.0, sentiment_score * 5))  # Scale and clamp
    
    # Calculate confidence based on sentiment word density
    confidence = min(1.0, (positive_count + negative_count) / max(total_words, 1) * 10)
    
    return {
        'score': sentiment_score,
        'confidence': max(0.1, confidence)  # Minimum confidence for fallback
    }

def send_to_dlq(failed_records: List[Dict]) -> None:
    """
    Send failed records to Dead Letter Queue.
    
    Args:
        failed_records: List of failed record data
    """
    if not DLQ_QUEUE_URL:
        logger.warning("DLQ URL not configured, skipping failed record handling")
        return
    
    try:
        sqs = boto3.client('sqs')
        
        for failed_record in failed_records:
            message_body = {
                'source': 'sentiment-processor',
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
    # Sample test event
    test_event = {
        'Records': [
            {
                'kinesis': {
                    'data': base64.b64encode(json.dumps({
                        'symbol': 'AAPL',
                        'content': 'Apple stock is looking great today! Bullish trend continues.',
                        'source': 'twitter',
                        'timestamp': datetime.utcnow().isoformat()
                    }).encode()).decode()
                }
            }
        ]
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))