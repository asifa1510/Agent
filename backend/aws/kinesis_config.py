"""
Kinesis Data Streams configuration and management utilities.
Handles creation and management of three streams: social-media, financial-news, market-data.
"""

import boto3
import logging
from typing import Dict, List, Optional
from botocore.exceptions import ClientError
import time

logger = logging.getLogger(__name__)

class KinesisStreamManager:
    """Manages Kinesis Data Streams for the trading agent."""
    
    # Stream configurations
    STREAM_CONFIGS = {
        'social-media': {
            'shard_count': 2,
            'retention_hours': 24,
            'partition_key': 'symbol'
        },
        'financial-news': {
            'shard_count': 1,
            'retention_hours': 24,
            'partition_key': 'symbol'
        },
        'market-data': {
            'shard_count': 3,
            'retention_hours': 24,
            'partition_key': 'symbol'
        }
    }
    
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize Kinesis client."""
        self.kinesis_client = boto3.client('kinesis', region_name=region_name)
        self.region_name = region_name
    
    def create_stream(self, stream_name: str) -> bool:
        """
        Create a Kinesis stream with configuration.
        
        Args:
            stream_name: Name of the stream to create
            
        Returns:
            bool: True if created successfully, False otherwise
        """
        if stream_name not in self.STREAM_CONFIGS:
            logger.error(f"Unknown stream configuration: {stream_name}")
            return False
            
        config = self.STREAM_CONFIGS[stream_name]
        
        try:
            # Check if stream already exists
            if self.stream_exists(stream_name):
                logger.info(f"Stream {stream_name} already exists")
                return True
                
            # Create the stream
            response = self.kinesis_client.create_stream(
                StreamName=stream_name,
                ShardCount=config['shard_count']
            )
            
            logger.info(f"Creating stream {stream_name} with {config['shard_count']} shards")
            
            # Wait for stream to become active
            if self.wait_for_stream_active(stream_name):
                # Set retention period
                self.kinesis_client.increase_stream_retention_period(
                    StreamName=stream_name,
                    RetentionPeriodHours=config['retention_hours']
                )
                logger.info(f"Stream {stream_name} created successfully")
                return True
            else:
                logger.error(f"Stream {stream_name} failed to become active")
                return False
                
        except ClientError as e:
            logger.error(f"Error creating stream {stream_name}: {e}")
            return False
    
    def stream_exists(self, stream_name: str) -> bool:
        """Check if a stream exists."""
        try:
            self.kinesis_client.describe_stream(StreamName=stream_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                return False
            raise
    
    def wait_for_stream_active(self, stream_name: str, timeout: int = 300) -> bool:
        """
        Wait for stream to become active.
        
        Args:
            stream_name: Name of the stream
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if stream becomes active, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.kinesis_client.describe_stream(StreamName=stream_name)
                status = response['StreamDescription']['StreamStatus']
                
                if status == 'ACTIVE':
                    return True
                elif status in ['DELETING', 'FAILED']:
                    logger.error(f"Stream {stream_name} is in {status} state")
                    return False
                    
                logger.info(f"Stream {stream_name} status: {status}, waiting...")
                time.sleep(10)
                
            except ClientError as e:
                logger.error(f"Error checking stream status: {e}")
                return False
                
        logger.error(f"Timeout waiting for stream {stream_name} to become active")
        return False
    
    def create_all_streams(self) -> Dict[str, bool]:
        """
        Create all required streams for the trading agent.
        
        Returns:
            Dict[str, bool]: Results for each stream creation
        """
        results = {}
        
        for stream_name in self.STREAM_CONFIGS.keys():
            logger.info(f"Creating stream: {stream_name}")
            results[stream_name] = self.create_stream(stream_name)
            
        return results
    
    def get_stream_info(self, stream_name: str) -> Optional[Dict]:
        """Get detailed information about a stream."""
        try:
            response = self.kinesis_client.describe_stream(StreamName=stream_name)
            return response['StreamDescription']
        except ClientError as e:
            logger.error(f"Error getting stream info for {stream_name}: {e}")
            return None
    
    def list_streams(self) -> List[str]:
        """List all Kinesis streams."""
        try:
            response = self.kinesis_client.list_streams()
            return response['StreamNames']
        except ClientError as e:
            logger.error(f"Error listing streams: {e}")
            return []
    
    def delete_stream(self, stream_name: str) -> bool:
        """Delete a Kinesis stream."""
        try:
            self.kinesis_client.delete_stream(StreamName=stream_name)
            logger.info(f"Stream {stream_name} deletion initiated")
            return True
        except ClientError as e:
            logger.error(f"Error deleting stream {stream_name}: {e}")
            return False
    
    def get_partition_key(self, stream_name: str) -> str:
        """Get the partition key for a stream."""
        if stream_name in self.STREAM_CONFIGS:
            return self.STREAM_CONFIGS[stream_name]['partition_key']
        return 'default'


def create_trading_streams(region_name: str = 'us-east-1') -> Dict[str, bool]:
    """
    Convenience function to create all trading agent streams.
    
    Args:
        region_name: AWS region for stream creation
        
    Returns:
        Dict[str, bool]: Results for each stream creation
    """
    manager = KinesisStreamManager(region_name)
    return manager.create_all_streams()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create all streams
    results = create_trading_streams()
    
    print("Stream creation results:")
    for stream_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {stream_name}: {status}")