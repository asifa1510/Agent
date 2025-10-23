"""
Base repository class for DynamoDB operations.
Provides common functionality for all data repositories.
"""

import boto3
import logging
from typing import Dict, List, Optional, Any, Type, TypeVar
from abc import ABC, abstractmethod
from botocore.exceptions import ClientError
from pydantic import BaseModel

from ..config import settings

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class BaseRepository(ABC):
    """Base repository class for DynamoDB operations."""
    
    def __init__(self, table_name: str):
        """
        Initialize repository with DynamoDB table.
        
        Args:
            table_name: Name of the DynamoDB table
        """
        self.table_name = table_name
        self.dynamodb = boto3.resource(
            'dynamodb',
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key
        )
        self.table = self.dynamodb.Table(table_name)
        
    @abstractmethod
    def get_model_class(self) -> Type[T]:
        """Return the Pydantic model class for this repository."""
        pass
    
    async def create_item(self, item: T) -> bool:
        """
        Create a new item in the table.
        
        Args:
            item: Pydantic model instance to create
            
        Returns:
            True if successful, False otherwise
        """
        try:
            item_dict = item.model_dump()
            self.table.put_item(Item=item_dict)
            logger.info(f"Created item in {self.table_name}")
            return True
        except ClientError as e:
            logger.error(f"Error creating item in {self.table_name}: {e}")
            return False
    
    async def get_item(self, key: Dict[str, Any]) -> Optional[T]:
        """
        Get an item by its primary key.
        
        Args:
            key: Primary key dictionary
            
        Returns:
            Pydantic model instance if found, None otherwise
        """
        try:
            response = self.table.get_item(Key=key)
            if 'Item' in response:
                model_class = self.get_model_class()
                return model_class(**response['Item'])
            return None
        except ClientError as e:
            logger.error(f"Error getting item from {self.table_name}: {e}")
            return None
    
    async def update_item(self, key: Dict[str, Any], updates: Dict[str, Any]) -> bool:
        """
        Update an item with the given updates.
        
        Args:
            key: Primary key dictionary
            updates: Dictionary of field updates
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build update expression
            update_expression = "SET "
            expression_values = {}
            
            for field, value in updates.items():
                update_expression += f"{field} = :{field}, "
                expression_values[f":{field}"] = value
            
            update_expression = update_expression.rstrip(", ")
            
            self.table.update_item(
                Key=key,
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_values
            )
            logger.info(f"Updated item in {self.table_name}")
            return True
        except ClientError as e:
            logger.error(f"Error updating item in {self.table_name}: {e}")
            return False
    
    async def delete_item(self, key: Dict[str, Any]) -> bool:
        """
        Delete an item by its primary key.
        
        Args:
            key: Primary key dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.table.delete_item(Key=key)
            logger.info(f"Deleted item from {self.table_name}")
            return True
        except ClientError as e:
            logger.error(f"Error deleting item from {self.table_name}: {e}")
            return False
    
    async def query_items(
        self, 
        key_condition: str,
        expression_values: Dict[str, Any],
        limit: Optional[int] = None,
        scan_forward: bool = True
    ) -> List[T]:
        """
        Query items using a key condition.
        
        Args:
            key_condition: DynamoDB key condition expression
            expression_values: Values for the expression
            limit: Maximum number of items to return
            scan_forward: Whether to scan forward or backward
            
        Returns:
            List of Pydantic model instances
        """
        try:
            query_params = {
                'KeyConditionExpression': key_condition,
                'ExpressionAttributeValues': expression_values,
                'ScanIndexForward': scan_forward
            }
            
            if limit:
                query_params['Limit'] = limit
            
            response = self.table.query(**query_params)
            
            model_class = self.get_model_class()
            return [model_class(**item) for item in response.get('Items', [])]
        except ClientError as e:
            logger.error(f"Error querying items from {self.table_name}: {e}")
            return []
    
    async def scan_items(
        self,
        filter_expression: Optional[str] = None,
        expression_values: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> List[T]:
        """
        Scan items with optional filtering.
        
        Args:
            filter_expression: Optional filter expression
            expression_values: Values for the expression
            limit: Maximum number of items to return
            
        Returns:
            List of Pydantic model instances
        """
        try:
            scan_params = {}
            
            if filter_expression:
                scan_params['FilterExpression'] = filter_expression
            if expression_values:
                scan_params['ExpressionAttributeValues'] = expression_values
            if limit:
                scan_params['Limit'] = limit
            
            response = self.table.scan(**scan_params)
            
            model_class = self.get_model_class()
            return [model_class(**item) for item in response.get('Items', [])]
        except ClientError as e:
            logger.error(f"Error scanning items from {self.table_name}: {e}")
            return []