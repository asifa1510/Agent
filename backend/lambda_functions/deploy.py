#!/usr/bin/env python3
"""
Deployment script for Lambda data processors.
Deploys all Lambda functions with proper error handling and DLQ configuration.
"""

import logging
import sys
import argparse
from deploy_config import LambdaDeploymentManager
from backend.aws.kinesis_config import KinesisStreamManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Deploy Lambda data processors')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deployed without actually deploying')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Lambda deployment in region: {args.region}")
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual deployment will occur")
        print_deployment_plan()
        return
    
    # Verify Kinesis streams exist
    logger.info("Verifying Kinesis streams...")
    kinesis_manager = KinesisStreamManager(args.region)
    
    required_streams = ['social-media', 'financial-news', 'market-data']
    missing_streams = []
    
    for stream_name in required_streams:
        if not kinesis_manager.stream_exists(stream_name):
            missing_streams.append(stream_name)
    
    if missing_streams:
        logger.error(f"Missing required Kinesis streams: {missing_streams}")
        logger.info("Please create the streams first using the Kinesis configuration")
        sys.exit(1)
    
    # Deploy Lambda functions
    logger.info("Deploying Lambda functions...")
    deployment_manager = LambdaDeploymentManager(args.region)
    
    results = deployment_manager.deploy_all_functions()
    
    # Print results
    print("\n" + "="*50)
    print("DEPLOYMENT RESULTS")
    print("="*50)
    
    success_count = 0
    total_count = len(results)
    
    for function_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{function_name:20} {status}")
        if success:
            success_count += 1
    
    print(f"\nSummary: {success_count}/{total_count} functions deployed successfully")
    
    if success_count == total_count:
        print("\n🎉 All Lambda functions deployed successfully!")
        print("\nNext steps:")
        print("1. Verify SageMaker endpoints are deployed")
        print("2. Create DynamoDB tables")
        print("3. Test the data pipeline with sample data")
    else:
        print(f"\n⚠️  {total_count - success_count} functions failed to deploy")
        print("Check the logs above for error details")
        sys.exit(1)

def print_deployment_plan():
    """Print what would be deployed in dry-run mode."""
    print("\n" + "="*50)
    print("DEPLOYMENT PLAN")
    print("="*50)
    
    print("\n1. Infrastructure Components:")
    print("   - Dead Letter Queue (SQS)")
    print("   - Lambda Execution Role (IAM)")
    print("   - Lambda Function Policies")
    
    print("\n2. Lambda Functions:")
    manager = LambdaDeploymentManager()
    
    for function_name, config in manager.LAMBDA_CONFIGS.items():
        print(f"   - {function_name}")
        print(f"     Handler: {config['handler']}")
        print(f"     Runtime: {config['runtime']}")
        print(f"     Memory: {config['memory_size']}MB")
        print(f"     Timeout: {config['timeout']}s")
        print(f"     Trigger: {config['trigger_stream']} Kinesis stream")
        print()
    
    print("3. Event Source Mappings:")
    print("   - sentiment-processor <- social-media stream")
    print("   - news-processor <- financial-news stream")
    print("   - market-processor <- market-data stream")
    
    print("\n4. Error Handling:")
    print("   - Dead Letter Queue for failed executions")
    print("   - Retry logic with exponential backoff")
    print("   - Circuit breaker patterns for external APIs")

if __name__ == "__main__":
    main()