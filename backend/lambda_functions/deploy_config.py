"""
Deployment configuration for Lambda functions.
Contains settings for creating and deploying the Lambda functions with DLQ support.
"""

import boto3
import json
import logging
import zipfile
import os
from typing import Dict, List, Optional
from botocore.exceptions import ClientError
import time

logger = logging.getLogger(__name__)

class LambdaDeploymentManager:
    """Manages deployment of Lambda functions for the trading agent."""
    
    # Lambda function configurations
    LAMBDA_CONFIGS = {
        'sentiment-processor': {
            'handler': 'sentiment_processor.lambda_handler',
            'runtime': 'python3.9',
            'timeout': 300,  # 5 minutes
            'memory_size': 512,
            'description': 'Processes social media data for sentiment analysis',
            'environment_variables': {
                'BERT_ENDPOINT': 'sentiment-bert-endpoint',
                'SENTIMENT_TABLE': 'sentiment-scores'
            },
            'trigger_stream': 'social-media'
        },
        'news-processor': {
            'handler': 'news_processor.lambda_handler',
            'runtime': 'python3.9',
            'timeout': 300,  # 5 minutes
            'memory_size': 512,
            'description': 'Processes financial news data and scores relevance/impact',
            'environment_variables': {
                'BERT_ENDPOINT': 'sentiment-bert-endpoint',
                'NEWS_SCORES_TABLE': 'news-scores'
            },
            'trigger_stream': 'financial-news'
        },
        'market-processor': {
            'handler': 'market_processor.lambda_handler',
            'runtime': 'python3.9',
            'timeout': 180,  # 3 minutes
            'memory_size': 256,
            'description': 'Processes market data and calculates technical indicators',
            'environment_variables': {
                'MARKET_DATA_TABLE': 'market-data',
                'TECHNICAL_INDICATORS_TABLE': 'technical-indicators'
            },
            'trigger_stream': 'market-data'
        }
    }
    
    def __init__(self, region_name: str = 'us-east-1'):
        """Initialize AWS clients."""
        self.region_name = region_name
        self.lambda_client = boto3.client('lambda', region_name=region_name)
        self.iam_client = boto3.client('iam', region_name=region_name)
        self.sqs_client = boto3.client('sqs', region_name=region_name)
        self.kinesis_client = boto3.client('kinesis', region_name=region_name)
        
        # DLQ configuration
        self.dlq_name = 'trading-agent-dlq'
        self.dlq_url = None
    
    def create_dead_letter_queue(self) -> Optional[str]:
        """
        Create SQS Dead Letter Queue for failed Lambda executions.
        
        Returns:
            str: DLQ URL if created successfully, None otherwise
        """
        try:
            # Check if DLQ already exists
            try:
                response = self.sqs_client.get_queue_url(QueueName=self.dlq_name)
                self.dlq_url = response['QueueUrl']
                logger.info(f"DLQ already exists: {self.dlq_url}")
                return self.dlq_url
            except ClientError as e:
                if e.response['Error']['Code'] != 'AWS.SimpleQueueService.NonExistentQueue':
                    raise
            
            # Create DLQ
            response = self.sqs_client.create_queue(
                QueueName=self.dlq_name,
                Attributes={
                    'MessageRetentionPeriod': '1209600',  # 14 days
                    'VisibilityTimeoutSeconds': '60',
                    'ReceiveMessageWaitTimeSeconds': '20'  # Long polling
                }
            )
            
            self.dlq_url = response['QueueUrl']
            logger.info(f"Created DLQ: {self.dlq_url}")
            
            return self.dlq_url
            
        except ClientError as e:
            logger.error(f"Error creating DLQ: {e}")
            return None
    
    def create_lambda_execution_role(self) -> Optional[str]:
        """
        Create IAM role for Lambda execution with necessary permissions.
        
        Returns:
            str: Role ARN if created successfully, None otherwise
        """
        role_name = 'TradingAgentLambdaExecutionRole'
        
        try:
            # Check if role already exists
            try:
                response = self.iam_client.get_role(RoleName=role_name)
                role_arn = response['Role']['Arn']
                logger.info(f"Lambda execution role already exists: {role_arn}")
                return role_arn
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchEntity':
                    raise
            
            # Trust policy for Lambda
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "lambda.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            # Create role
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description='Execution role for Trading Agent Lambda functions'
            )
            
            role_arn = response['Role']['Arn']
            
            # Attach basic Lambda execution policy
            self.iam_client.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole'
            )
            
            # Create and attach custom policy for trading agent permissions
            custom_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "kinesis:DescribeStream",
                            "kinesis:GetShardIterator",
                            "kinesis:GetRecords",
                            "kinesis:ListStreams"
                        ],
                        "Resource": "*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "dynamodb:PutItem",
                            "dynamodb:GetItem",
                            "dynamodb:Query",
                            "dynamodb:Scan",
                            "dynamodb:UpdateItem"
                        ],
                        "Resource": "*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sagemaker:InvokeEndpoint"
                        ],
                        "Resource": "*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "bedrock:InvokeModel"
                        ],
                        "Resource": "*"
                    },
                    {
                        "Effect": "Allow",
                        "Action": [
                            "sqs:SendMessage",
                            "sqs:GetQueueUrl"
                        ],
                        "Resource": "*"
                    }
                ]
            }
            
            policy_name = 'TradingAgentLambdaPolicy'
            
            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName=policy_name,
                PolicyDocument=json.dumps(custom_policy)
            )
            
            logger.info(f"Created Lambda execution role: {role_arn}")
            
            # Wait for role to be available
            time.sleep(10)
            
            return role_arn
            
        except ClientError as e:
            logger.error(f"Error creating Lambda execution role: {e}")
            return None
    
    def create_deployment_package(self, function_name: str) -> Optional[str]:
        """
        Create deployment package for Lambda function.
        
        Args:
            function_name: Name of the Lambda function
            
        Returns:
            str: Path to deployment package if created successfully, None otherwise
        """
        try:
            package_path = f"/tmp/{function_name}-deployment.zip"
            
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add the main Lambda function file
                function_file = f"{function_name.replace('-', '_')}.py"
                function_path = os.path.join(os.path.dirname(__file__), function_file)
                
                if os.path.exists(function_path):
                    zip_file.write(function_path, function_file)
                else:
                    logger.error(f"Function file not found: {function_path}")
                    return None
                
                # Add any additional dependencies if needed
                # For now, we rely on Lambda's built-in libraries
                
            logger.info(f"Created deployment package: {package_path}")
            return package_path
            
        except Exception as e:
            logger.error(f"Error creating deployment package for {function_name}: {e}")
            return None
    
    def deploy_lambda_function(self, function_name: str, role_arn: str) -> bool:
        """
        Deploy a Lambda function.
        
        Args:
            function_name: Name of the Lambda function
            role_arn: ARN of the execution role
            
        Returns:
            bool: True if deployed successfully, False otherwise
        """
        try:
            config = self.LAMBDA_CONFIGS[function_name]
            
            # Create deployment package
            package_path = self.create_deployment_package(function_name)
            if not package_path:
                return False
            
            # Read deployment package
            with open(package_path, 'rb') as zip_file:
                zip_content = zip_file.read()
            
            # Prepare environment variables
            env_vars = config['environment_variables'].copy()
            if self.dlq_url:
                env_vars['DLQ_QUEUE_URL'] = self.dlq_url
            
            try:
                # Check if function already exists
                self.lambda_client.get_function(FunctionName=function_name)
                
                # Update existing function
                self.lambda_client.update_function_code(
                    FunctionName=function_name,
                    ZipFile=zip_content
                )
                
                self.lambda_client.update_function_configuration(
                    FunctionName=function_name,
                    Runtime=config['runtime'],
                    Role=role_arn,
                    Handler=config['handler'],
                    Description=config['description'],
                    Timeout=config['timeout'],
                    MemorySize=config['memory_size'],
                    Environment={'Variables': env_vars}
                )
                
                logger.info(f"Updated Lambda function: {function_name}")
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    # Create new function
                    response = self.lambda_client.create_function(
                        FunctionName=function_name,
                        Runtime=config['runtime'],
                        Role=role_arn,
                        Handler=config['handler'],
                        Code={'ZipFile': zip_content},
                        Description=config['description'],
                        Timeout=config['timeout'],
                        MemorySize=config['memory_size'],
                        Environment={'Variables': env_vars}
                    )
                    
                    logger.info(f"Created Lambda function: {function_name}")
                else:
                    raise
            
            # Clean up deployment package
            os.remove(package_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deploying Lambda function {function_name}: {e}")
            return False
    
    def create_kinesis_trigger(self, function_name: str, stream_name: str) -> bool:
        """
        Create Kinesis trigger for Lambda function.
        
        Args:
            function_name: Name of the Lambda function
            stream_name: Name of the Kinesis stream
            
        Returns:
            bool: True if trigger created successfully, False otherwise
        """
        try:
            # Get stream ARN
            response = self.kinesis_client.describe_stream(StreamName=stream_name)
            stream_arn = response['StreamDescription']['StreamARN']
            
            # Create event source mapping
            response = self.lambda_client.create_event_source_mapping(
                EventSourceArn=stream_arn,
                FunctionName=function_name,
                StartingPosition='LATEST',
                BatchSize=10,
                MaximumBatchingWindowInSeconds=5,
                ParallelizationFactor=1,
                MaximumRecordAgeInSeconds=3600,  # 1 hour
                BisectBatchOnFunctionError=True,
                MaximumRetryAttempts=3
            )
            
            logger.info(f"Created Kinesis trigger for {function_name} -> {stream_name}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceConflictException':
                logger.info(f"Kinesis trigger already exists for {function_name} -> {stream_name}")
                return True
            else:
                logger.error(f"Error creating Kinesis trigger for {function_name}: {e}")
                return False
    
    def deploy_all_functions(self) -> Dict[str, bool]:
        """
        Deploy all Lambda functions for the trading agent.
        
        Returns:
            Dict[str, bool]: Results for each function deployment
        """
        results = {}
        
        # Create DLQ
        dlq_url = self.create_dead_letter_queue()
        if not dlq_url:
            logger.error("Failed to create DLQ, continuing without it")
        
        # Create execution role
        role_arn = self.create_lambda_execution_role()
        if not role_arn:
            logger.error("Failed to create execution role")
            return {name: False for name in self.LAMBDA_CONFIGS.keys()}
        
        # Deploy each function
        for function_name, config in self.LAMBDA_CONFIGS.items():
            logger.info(f"Deploying Lambda function: {function_name}")
            
            # Deploy function
            success = self.deploy_lambda_function(function_name, role_arn)
            results[function_name] = success
            
            if success:
                # Create Kinesis trigger
                trigger_success = self.create_kinesis_trigger(
                    function_name, 
                    config['trigger_stream']
                )
                if not trigger_success:
                    logger.warning(f"Failed to create trigger for {function_name}")
            
        return results
    
    def get_function_info(self, function_name: str) -> Optional[Dict]:
        """Get information about a deployed Lambda function."""
        try:
            response = self.lambda_client.get_function(FunctionName=function_name)
            return response
        except ClientError as e:
            logger.error(f"Error getting function info for {function_name}: {e}")
            return None


def deploy_trading_lambda_functions(region_name: str = 'us-east-1') -> Dict[str, bool]:
    """
    Convenience function to deploy all trading agent Lambda functions.
    
    Args:
        region_name: AWS region for deployment
        
    Returns:
        Dict[str, bool]: Results for each function deployment
    """
    manager = LambdaDeploymentManager(region_name)
    return manager.deploy_all_functions()


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Deploy all functions
    results = deploy_trading_lambda_functions()
    
    print("Lambda deployment results:")
    for function_name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {function_name}: {status}")