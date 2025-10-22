"""
Lambda function for processing market data and calculating technical indicators.
Processes data from the market-data Kinesis stream.
"""

import json
import logging
import boto3
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os
import statistics

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource('dynamodb')

# Configuration from environment variables
MARKET_DATA_TABLE = os.environ.get('MARKET_DATA_TABLE', 'market-data')
TECHNICAL_INDICATORS_TABLE = os.environ.get('TECHNICAL_INDICATORS_TABLE', 'technical-indicators')
DLQ_QUEUE_URL = os.environ.get('DLQ_QUEUE_URL')

# Initialize DynamoDB tables
market_data_table = dynamodb.Table(MARKET_DATA_TABLE)
indicators_table = dynamodb.Table(TECHNICAL_INDICATORS_TABLE)

def lambda_handler(event, context):
    """
    Main Lambda handler for processing Kinesis records.
    
    Args:
        event: Kinesis event containing records
        context: Lambda context
        
    Returns:
        Dict with processing results
    """
    logger.info(f"Processing {len(event['Records'])} market data records")
    
    processed_count = 0
    failed_count = 0
    failed_records = []
    
    for record in event['Records']:
        try:
            # Decode Kinesis data
            payload = base64.b64decode(record['kinesis']['data']).decode('utf-8')
            data = json.loads(payload)
            
            # Process the market data record
            result = process_market_record(data)
            
            if result:
                processed_count += 1
                logger.info(f"Successfully processed market data for symbol: {data.get('symbol', 'unknown')}")
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

def process_market_record(data: Dict[str, Any]) -> bool:
    """
    Process a single market data record.
    
    Args:
        data: Market data record
        
    Returns:
        bool: True if processed successfully
    """
    try:
        # Validate required fields
        required_fields = ['symbol', 'price', 'volume', 'timestamp']
        if not all(field in data for field in required_fields):
            logger.error(f"Missing required fields: {required_fields}")
            return False
        
        # Store raw market data
        market_record = {
            'symbol': data['symbol'],
            'timestamp': int(datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')).timestamp()),
            'price': float(data['price']),
            'volume': int(data['volume']),
            'high': float(data.get('high', data['price'])),
            'low': float(data.get('low', data['price'])),
            'open': float(data.get('open', data['price'])),
            'close': float(data['price']),  # Current price as close
            'processing_timestamp': int(datetime.utcnow().timestamp()),
            'ttl': int(datetime.utcnow().timestamp()) + (90 * 24 * 60 * 60)  # 90 days TTL
        }
        
        # Store market data
        market_data_table.put_item(Item=market_record)
        
        # Calculate and store technical indicators
        calculate_technical_indicators(data['symbol'], market_record)
        
        logger.info(f"Stored market data for {data['symbol']}: price=${data['price']}, volume={data['volume']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing market record: {e}")
        return False

def calculate_technical_indicators(symbol: str, current_data: Dict[str, Any]) -> None:
    """
    Calculate technical indicators for the symbol.
    
    Args:
        symbol: Stock symbol
        current_data: Current market data record
    """
    try:
        # Get historical data for calculations
        historical_data = get_historical_data(symbol, days=50)  # 50 days for indicators
        
        if len(historical_data) < 2:
            logger.warning(f"Insufficient historical data for {symbol}, skipping indicators")
            return
        
        # Add current data to historical data
        all_data = historical_data + [current_data]
        all_data.sort(key=lambda x: x['timestamp'])
        
        # Calculate various technical indicators
        indicators = {}
        
        # Simple Moving Averages
        indicators.update(calculate_moving_averages(all_data))
        
        # RSI (Relative Strength Index)
        indicators['rsi_14'] = calculate_rsi(all_data, period=14)
        
        # MACD
        macd_data = calculate_macd(all_data)
        indicators.update(macd_data)
        
        # Bollinger Bands
        bollinger_data = calculate_bollinger_bands(all_data, period=20)
        indicators.update(bollinger_data)
        
        # Volume indicators
        indicators.update(calculate_volume_indicators(all_data))
        
        # Price change indicators
        indicators.update(calculate_price_changes(all_data))
        
        # Store indicators
        indicator_record = {
            'symbol': symbol,
            'timestamp': current_data['timestamp'],
            'price': current_data['price'],
            **indicators,
            'processing_timestamp': int(datetime.utcnow().timestamp()),
            'ttl': int(datetime.utcnow().timestamp()) + (30 * 24 * 60 * 60)  # 30 days TTL
        }
        
        indicators_table.put_item(Item=indicator_record)
        
        logger.info(f"Calculated technical indicators for {symbol}")
        
    except Exception as e:
        logger.error(f"Error calculating technical indicators for {symbol}: {e}")

def get_historical_data(symbol: str, days: int = 50) -> List[Dict[str, Any]]:
    """
    Get historical market data for the symbol.
    
    Args:
        symbol: Stock symbol
        days: Number of days of historical data
        
    Returns:
        List of historical market data records
    """
    try:
        # Calculate timestamp range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        start_timestamp = int(start_time.timestamp())
        end_timestamp = int(end_time.timestamp())
        
        # Query DynamoDB for historical data
        response = market_data_table.query(
            KeyConditionExpression='symbol = :symbol AND #ts BETWEEN :start_ts AND :end_ts',
            ExpressionAttributeNames={'#ts': 'timestamp'},
            ExpressionAttributeValues={
                ':symbol': symbol,
                ':start_ts': start_timestamp,
                ':end_ts': end_timestamp
            },
            ScanIndexForward=True,  # Sort by timestamp ascending
            Limit=days * 24  # Approximate limit for hourly data
        )
        
        return response.get('Items', [])
        
    except Exception as e:
        logger.error(f"Error getting historical data for {symbol}: {e}")
        return []

def calculate_moving_averages(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate simple moving averages."""
    prices = [float(item['price']) for item in data]
    
    indicators = {}
    
    # Calculate different period moving averages
    periods = [5, 10, 20, 50]
    
    for period in periods:
        if len(prices) >= period:
            ma_value = statistics.mean(prices[-period:])
            indicators[f'sma_{period}'] = round(ma_value, 4)
    
    return indicators

def calculate_rsi(data: List[Dict[str, Any]], period: int = 14) -> Optional[float]:
    """Calculate Relative Strength Index."""
    if len(data) < period + 1:
        return None
    
    prices = [float(item['price']) for item in data]
    
    # Calculate price changes
    changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    
    if len(changes) < period:
        return None
    
    # Separate gains and losses
    gains = [change if change > 0 else 0 for change in changes[-period:]]
    losses = [-change if change < 0 else 0 for change in changes[-period:]]
    
    # Calculate average gain and loss
    avg_gain = statistics.mean(gains) if gains else 0
    avg_loss = statistics.mean(losses) if losses else 0
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 4)

def calculate_macd(data: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    if len(data) < 26:
        return {'macd': None, 'macd_signal': None, 'macd_histogram': None}
    
    prices = [float(item['price']) for item in data]
    
    # Calculate EMAs
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    
    if ema_12 is None or ema_26 is None:
        return {'macd': None, 'macd_signal': None, 'macd_histogram': None}
    
    macd_line = ema_12 - ema_26
    
    # For signal line, we'd need historical MACD values
    # Simplified version - just return MACD line
    return {
        'macd': round(macd_line, 4),
        'macd_signal': None,  # Would need historical MACD data
        'macd_histogram': None
    }

def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return None
    
    multiplier = 2 / (period + 1)
    ema = statistics.mean(prices[:period])  # Start with SMA
    
    for price in prices[period:]:
        ema = (price * multiplier) + (ema * (1 - multiplier))
    
    return ema

def calculate_bollinger_bands(data: List[Dict[str, Any]], period: int = 20) -> Dict[str, Optional[float]]:
    """Calculate Bollinger Bands."""
    if len(data) < period:
        return {'bb_upper': None, 'bb_middle': None, 'bb_lower': None}
    
    prices = [float(item['price']) for item in data[-period:]]
    
    middle = statistics.mean(prices)
    std_dev = statistics.stdev(prices) if len(prices) > 1 else 0
    
    upper = middle + (2 * std_dev)
    lower = middle - (2 * std_dev)
    
    return {
        'bb_upper': round(upper, 4),
        'bb_middle': round(middle, 4),
        'bb_lower': round(lower, 4)
    }

def calculate_volume_indicators(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate volume-based indicators."""
    if len(data) < 2:
        return {}
    
    volumes = [int(item['volume']) for item in data]
    
    indicators = {}
    
    # Volume moving averages
    periods = [10, 20]
    for period in periods:
        if len(volumes) >= period:
            vol_ma = statistics.mean(volumes[-period:])
            indicators[f'volume_ma_{period}'] = round(vol_ma, 0)
    
    # Current volume vs average
    if len(volumes) >= 20:
        avg_volume = statistics.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        indicators['volume_ratio'] = round(volume_ratio, 4)
    
    return indicators

def calculate_price_changes(data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate price change indicators."""
    if len(data) < 2:
        return {}
    
    current_price = float(data[-1]['price'])
    
    indicators = {}
    
    # Price changes over different periods
    periods = [1, 5, 10, 20]  # 1 period, 5 periods, etc.
    
    for period in periods:
        if len(data) > period:
            old_price = float(data[-(period + 1)]['price'])
            price_change = current_price - old_price
            price_change_pct = (price_change / old_price) * 100 if old_price > 0 else 0
            
            indicators[f'price_change_{period}'] = round(price_change, 4)
            indicators[f'price_change_pct_{period}'] = round(price_change_pct, 4)
    
    return indicators

def send_to_dlq(failed_records: List[Dict]) -> None:
    """Send failed records to Dead Letter Queue."""
    if not DLQ_QUEUE_URL:
        logger.warning("DLQ URL not configured, skipping failed record handling")
        return
    
    try:
        sqs = boto3.client('sqs')
        
        for failed_record in failed_records:
            message_body = {
                'source': 'market-processor',
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
                        'price': 150.25,
                        'volume': 1000000,
                        'high': 151.0,
                        'low': 149.5,
                        'open': 150.0,
                        'timestamp': datetime.utcnow().isoformat()
                    }).encode()).decode()
                }
            }
        ]
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))