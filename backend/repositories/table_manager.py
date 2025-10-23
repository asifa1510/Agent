"""
DynamoDB table management utilities.
Handles table creation and schema management.
"""

import boto3
import logging
from typing import Dict, Any, List
from botocore.exceptions import ClientError

from ..config import settings

logger = logging.getLogger(__name__)

class TableManager:
    """Manages DynamoDB table creation and configuration."""
    
    def __init__(self):
        self.dynamodb = boto3.resource(
            'dynamodb',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
        self.client = boto3.client(
            'dynamodb',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
    
    def get_table_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Get table schema definitions for all required tables.
        
        Returns:
            Dictionary mapping table names to their schema definitions
        """
        return {
            settings.sentiment_table: {
                'AttributeDefinitions': [
                    {'AttributeName': 'symbol', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'N'}
                ],
                'KeySchema': [
                    {'AttributeName': 'symbol', 'KeyType': 'HASH'},
                    {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                ],
                'BillingMode': 'PAY_PER_REQUEST'
            },
            settings.predictions_table: {
                'AttributeDefinitions': [
                    {'AttributeName': 'symbol', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'N'}
                ],
                'KeySchema': [
                    {'AttributeName': 'symbol', 'KeyType': 'HASH'},
                    {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                ],
                'BillingMode': 'PAY_PER_REQUEST'
            },
            settings.trades_table: {
                'AttributeDefinitions': [
                    {'AttributeName': 'id', 'AttributeType': 'S'}
                ],
                'KeySchema': [
                    {'AttributeName': 'id', 'KeyType': 'HASH'}
                ],
                'BillingMode': 'PAY_PER_REQUEST'
            },
            settings.explanations_table: {
                'AttributeDefinitions': [
                    {'AttributeName': 'id', 'AttributeType': 'S'}
                ],
                'KeySchema': [
                    {'AttributeName': 'id', 'KeyType': 'HASH'}
                ],
                'BillingMode': 'PAY_PER_REQUEST'
            },
            settings.portfolio_table: {
                'AttributeDefinitions': [
                    {'AttributeName': 'symbol', 'AttributeType': 'S'}
                ],
                'KeySchema': [
                    {'AttributeName': 'symbol', 'KeyType': 'HASH'}
                ],
                'BillingMode': 'PAY_PER_REQUEST'
            }
        }
    
    async def create_table(self, table_name: str, schema: Dict[str, Any]) -> bool:
        """
        Create a DynamoDB table with the given schema.
        
        Args:
            table_name: Name of the table to create
            schema: Table schema definition
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.create_table(
                TableName=table_name,
                **schema
            )
            
            # Wait for table to be created
            waiter = self.client.get_waiter('table_exists')
            waiter.wait(TableName=table_name)
            
            logger.info(f"Successfully created table: {table_name}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceInUseException':
                logger.info(f"Table {table_name} already exists")
                return True
            else:
                logger.error(f"Error creating table {table_name}: {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error creating table {table_name}: {e}")
            return False
    
    async def create_all_tables(self) -> bool:
        """
        Create all required tables for the application.
        
        Returns:
            True if all tables created successfully, False otherwise
        """
        schemas = self.get_table_schemas()
        success = True
        
        for table_name, schema in schemas.items():
            if not await self.create_table(table_name, schema):
                success = False
        
        return success
    
    async def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists, False otherwise
        """
        try:
            self.client.describe_table(TableName=table_name)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                return False
            else:
                logger.error(f"Error checking table {table_name}: {e}")
                return False
    
    async def delete_table(self, table_name: str) -> bool:
        """
        Delete a DynamoDB table.
        
        Args:
            table_name: Name of the table to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_table(TableName=table_name)
            
            # Wait for table to be deleted
            waiter = self.client.get_waiter('table_not_exists')
            waiter.wait(TableName=table_name)
            
            logger.info(f"Successfully deleted table: {table_name}")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.info(f"Table {table_name} does not exist")
                return True
            else:
                logger.error(f"Error deleting table {table_name}: {e}")
                return False
    
    async def list_tables(self) -> List[str]:
        """
        List all DynamoDB tables.
        
        Returns:
            List of table names
        """
        try:
            response = self.client.list_tables()
            return response.get('TableNames', [])
        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []