"""
Kinesis Data Producer for sending data to streams with proper partitioning.
"""

import boto3
import json
import logging
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class KinesisDataProducer:
    """Produces data to Kinesis streams with symbol-based partitioning."""
    
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize Kinesis producer."""
        self.kinesis_client = boto3.client('kinesis', region_name=region_name)
        self.region_name = region_name
    
    def _generate_partition_key(self, symbol: str, stream_name: str) -> str:
        """
        Generate partition key based on symbol for even distribution.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            stream_name: Name of the target stream
            
        Returns:
            str: Partition key for the record
        """
        # Use symbol as base, add hash for better distribution
        base_key = f"{symbol}_{stream_name}"
        hash_suffix = hashlib.md5(base_key.encode()).hexdigest()[:4]
        return f"{symbol}_{hash_suffix}"
    
    def send_social_media_data(self, data: Dict[str, Any]) -> bool:
        """
        Send social media data to social-media stream.
        
        Args:
            data: Social media data containing symbol and content
            
        Returns:
            bool: True if sent successfully
        """
        required_fields = ['symbol', 'content', 'source', 'timestamp']
        if not all(field in data for field in required_fields):
            logger.error(f"Missing required fields in social media data: {required_fields}")
            return False
            
        return self._send_to_stream('social-media', data, data['symbol'])
    
    def send_financial_news_data(self, data: Dict[str, Any]) -> bool:
        """
        Send financial news data to financial-news stream.
        
        Args:
            data: News data containing symbol and article info
            
        Returns:
            bool: True if sent successfully
        """
        required_fields = ['symbol', 'title', 'content', 'source', 'timestamp']
        if not all(field in data for field in required_fields):
            logger.error(f"Missing required fields in news data: {required_fields}")
            return False
            
        return self._send_to_stream('financial-news', data, data['symbol'])
    
    def send_market_data(self, data: Dict[str, Any]) -> bool:
        """
        Send market data to market-data stream.
        
        Args:
            data: Market data containing symbol and price info
            
        Returns:
            bool: True if sent successfully
        """
        required_fields = ['symbol', 'price', 'volume', 'timestamp']
        if not all(field in data for field in required_fields):
            logger.error(f"Missing required fields in market data: {required_fields}")
            return False
            
        return self._send_to_stream('market-data', data, data['symbol'])
    
    def _send_to_stream(self, stream_name: str, data: Dict[str, Any], symbol: str) -> bool:
        """
        Send data to specified Kinesis stream.
        
        Args:
            stream_name: Target stream name
            data: Data to send
            symbol: Stock symbol for partitioning
            
        Returns:
            bool: True if sent successfully
        """
        try:
            # Add metadata
            enriched_data = {
                **data,
                'stream_name': stream_name,
                'ingestion_timestamp': datetime.utcnow().isoformat()
            }
            
            # Generate partition key
            partition_key = self._generate_partition_key(symbol, stream_name)
            
            # Send to Kinesis
            response = self.kinesis_client.put_record(
                StreamName=stream_name,
                Data=json.dumps(enriched_data),
                PartitionKey=partition_key
            )
            
            logger.info(f"Sent data to {stream_name} with sequence number: {response['SequenceNumber']}")
            return True
            
        except ClientError as e:
            logger.error(f"Error sending data to {stream_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error sending data to {stream_name}: {e}")
            return False
    
    def send_batch_records(self, stream_name: str, records: list) -> Dict[str, Any]:
        """
        Send multiple records to a stream in batch.
        
        Args:
            stream_name: Target stream name
            records: List of records, each containing 'data' and 'symbol'
            
        Returns:
            Dict with success count and failed records
        """
        if not records:
            return {'success_count': 0, 'failed_records': []}
            
        batch_records = []
        for i, record in enumerate(records):
            if 'data' not in record or 'symbol' not in record:
                logger.error(f"Invalid record format at index {i}")
                continue
                
            partition_key = self._generate_partition_key(record['symbol'], stream_name)
            
            # Add metadata to data
            enriched_data = {
                **record['data'],
                'stream_name': stream_name,
                'ingestion_timestamp': datetime.utcnow().isoformat()
            }
            
            batch_records.append({
                'Data': json.dumps(enriched_data),
                'PartitionKey': partition_key
            })
        
        if not batch_records:
            return {'success_count': 0, 'failed_records': []}
        
        try:
            response = self.kinesis_client.put_records(
                StreamName=stream_name,
                Records=batch_records
            )
            
            success_count = len(batch_records) - response['FailedRecordCount']
            failed_records = []
            
            # Collect failed records for retry
            for i, record_result in enumerate(response['Records']):
                if 'ErrorCode' in record_result:
                    failed_records.append({
                        'index': i,
                        'error_code': record_result['ErrorCode'],
                        'error_message': record_result.get('ErrorMessage', ''),
                        'original_record': records[i] if i < len(records) else None
                    })
            
            logger.info(f"Batch send to {stream_name}: {success_count} success, {len(failed_records)} failed")
            
            return {
                'success_count': success_count,
                'failed_records': failed_records
            }
            
        except ClientError as e:
            logger.error(f"Error sending batch to {stream_name}: {e}")
            return {
                'success_count': 0,
                'failed_records': [{'error': str(e), 'records': records}]
            }


# Convenience functions for common use cases
def send_twitter_sentiment(symbol: str, content: str, source: str = 'twitter') -> bool:
    """Send Twitter data for sentiment analysis."""
    producer = KinesisDataProducer()
    data = {
        'symbol': symbol,
        'content': content,
        'source': source,
        'timestamp': datetime.utcnow().isoformat()
    }
    return producer.send_social_media_data(data)


def send_news_article(symbol: str, title: str, content: str, source: str, url: str = None) -> bool:
    """Send news article for processing."""
    producer = KinesisDataProducer()
    data = {
        'symbol': symbol,
        'title': title,
        'content': content,
        'source': source,
        'url': url,
        'timestamp': datetime.utcnow().isoformat()
    }
    return producer.send_financial_news_data(data)


def send_price_update(symbol: str, price: float, volume: int, additional_data: Dict = None) -> bool:
    """Send market price update."""
    producer = KinesisDataProducer()
    data = {
        'symbol': symbol,
        'price': price,
        'volume': volume,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if additional_data:
        data.update(additional_data)
        
    return producer.send_market_data(data)