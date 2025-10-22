"""
AWS-specific configuration and utilities for Kinesis streams.
"""

from typing import Dict, List
from backend.config import settings
from .kinesis_config import KinesisStreamManager
from .kinesis_producer import KinesisDataProducer

class AWSKinesisConfig:
    """AWS Kinesis configuration manager."""
    
    def __init__(self):
        self.region = settings.aws_region
        self.stream_names = {
            'social_media': settings.social_media_stream,
            'financial_news': settings.financial_news_stream,
            'market_data': settings.market_data_stream
        }
    
    def get_stream_manager(self) -> KinesisStreamManager:
        """Get configured Kinesis stream manager."""
        return KinesisStreamManager(region_name=self.region)
    
    def get_data_producer(self) -> KinesisDataProducer:
        """Get configured Kinesis data producer."""
        return KinesisDataProducer(region_name=self.region)
    
    def get_all_stream_names(self) -> List[str]:
        """Get list of all configured stream names."""
        return list(self.stream_names.values())
    
    def get_stream_config_mapping(self) -> Dict[str, str]:
        """Get mapping of logical names to actual stream names."""
        return self.stream_names.copy()
    
    def initialize_streams(self) -> Dict[str, bool]:
        """Initialize all required Kinesis streams."""
        manager = self.get_stream_manager()
        
        # Update stream manager with our configured names
        updated_configs = {}
        for logical_name, stream_name in self.stream_names.items():
            if stream_name in manager.STREAM_CONFIGS:
                updated_configs[stream_name] = manager.STREAM_CONFIGS[stream_name]
            else:
                # Use default config for custom stream names
                updated_configs[stream_name] = {
                    'shard_count': 2,
                    'retention_hours': 24,
                    'partition_key': 'symbol'
                }
        
        # Temporarily update the manager's config
        original_configs = manager.STREAM_CONFIGS
        manager.STREAM_CONFIGS = updated_configs
        
        try:
            results = manager.create_all_streams()
        finally:
            # Restore original configs
            manager.STREAM_CONFIGS = original_configs
            
        return results


# Global AWS Kinesis configuration instance
aws_kinesis_config = AWSKinesisConfig()