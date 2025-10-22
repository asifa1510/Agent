"""
Command-line utility for managing Kinesis streams.
"""

import argparse
import logging
import sys
from typing import Dict, Any
from .config import aws_kinesis_config
from .kinesis_config import KinesisStreamManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_streams() -> bool:
    """Create all required Kinesis streams."""
    logger.info("Creating Kinesis streams for trading agent...")
    
    try:
        results = aws_kinesis_config.initialize_streams()
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        logger.info(f"Stream creation completed: {success_count}/{total_count} successful")
        
        for stream_name, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"  {stream_name}: {status}")
        
        return success_count == total_count
        
    except Exception as e:
        logger.error(f"Error creating streams: {e}")
        return False

def list_streams() -> bool:
    """List all Kinesis streams."""
    logger.info("Listing Kinesis streams...")
    
    try:
        manager = aws_kinesis_config.get_stream_manager()
        streams = manager.list_streams()
        
        if not streams:
            logger.info("No Kinesis streams found")
            return True
        
        logger.info(f"Found {len(streams)} streams:")
        for stream_name in streams:
            info = manager.get_stream_info(stream_name)
            if info:
                status = info['StreamStatus']
                shard_count = len(info['Shards'])
                logger.info(f"  {stream_name}: {status} ({shard_count} shards)")
            else:
                logger.info(f"  {stream_name}: Unable to get info")
        
        return True
        
    except Exception as e:
        logger.error(f"Error listing streams: {e}")
        return False

def describe_stream(stream_name: str) -> bool:
    """Describe a specific Kinesis stream."""
    logger.info(f"Describing stream: {stream_name}")
    
    try:
        manager = aws_kinesis_config.get_stream_manager()
        info = manager.get_stream_info(stream_name)
        
        if not info:
            logger.error(f"Stream {stream_name} not found or inaccessible")
            return False
        
        logger.info(f"Stream Details for {stream_name}:")
        logger.info(f"  Status: {info['StreamStatus']}")
        logger.info(f"  ARN: {info['StreamARN']}")
        logger.info(f"  Creation Time: {info['StreamCreationTimestamp']}")
        logger.info(f"  Retention Period: {info['RetentionPeriodHours']} hours")
        logger.info(f"  Shard Count: {len(info['Shards'])}")
        
        if info['Shards']:
            logger.info("  Shards:")
            for shard in info['Shards']:
                shard_id = shard['ShardId']
                hash_range = f"{shard['HashKeyRange']['StartingHashKey']}-{shard['HashKeyRange']['EndingHashKey']}"
                logger.info(f"    {shard_id}: {hash_range}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error describing stream {stream_name}: {e}")
        return False

def delete_stream(stream_name: str) -> bool:
    """Delete a Kinesis stream."""
    logger.warning(f"Deleting stream: {stream_name}")
    
    try:
        manager = aws_kinesis_config.get_stream_manager()
        
        if not manager.stream_exists(stream_name):
            logger.error(f"Stream {stream_name} does not exist")
            return False
        
        success = manager.delete_stream(stream_name)
        
        if success:
            logger.info(f"Stream {stream_name} deletion initiated")
        else:
            logger.error(f"Failed to delete stream {stream_name}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error deleting stream {stream_name}: {e}")
        return False

def check_streams_health() -> bool:
    """Check health of all configured streams."""
    logger.info("Checking health of configured streams...")
    
    try:
        manager = aws_kinesis_config.get_stream_manager()
        stream_names = aws_kinesis_config.get_all_stream_names()
        
        all_healthy = True
        
        for stream_name in stream_names:
            if manager.stream_exists(stream_name):
                info = manager.get_stream_info(stream_name)
                if info and info['StreamStatus'] == 'ACTIVE':
                    logger.info(f"  {stream_name}: ✓ HEALTHY")
                else:
                    status = info['StreamStatus'] if info else 'UNKNOWN'
                    logger.warning(f"  {stream_name}: ⚠ {status}")
                    all_healthy = False
            else:
                logger.error(f"  {stream_name}: ✗ NOT FOUND")
                all_healthy = False
        
        if all_healthy:
            logger.info("All streams are healthy")
        else:
            logger.warning("Some streams have issues")
        
        return all_healthy
        
    except Exception as e:
        logger.error(f"Error checking stream health: {e}")
        return False

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Kinesis Stream Management Utility')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create streams command
    subparsers.add_parser('create', help='Create all required streams')
    
    # List streams command
    subparsers.add_parser('list', help='List all streams')
    
    # Describe stream command
    describe_parser = subparsers.add_parser('describe', help='Describe a specific stream')
    describe_parser.add_argument('stream_name', help='Name of the stream to describe')
    
    # Delete stream command
    delete_parser = subparsers.add_parser('delete', help='Delete a stream')
    delete_parser.add_argument('stream_name', help='Name of the stream to delete')
    delete_parser.add_argument('--confirm', action='store_true', help='Confirm deletion')
    
    # Health check command
    subparsers.add_parser('health', help='Check health of all configured streams')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    success = False
    
    if args.command == 'create':
        success = create_streams()
    elif args.command == 'list':
        success = list_streams()
    elif args.command == 'describe':
        success = describe_stream(args.stream_name)
    elif args.command == 'delete':
        if not args.confirm:
            logger.error("Deletion requires --confirm flag")
            return 1
        success = delete_stream(args.stream_name)
    elif args.command == 'health':
        success = check_streams_health()
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())