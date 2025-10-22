"""
SageMaker deployment utilities for BERT sentiment analysis model.
Handles model deployment, endpoint management, and inference.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
import boto3
from sagemaker import Model, Predictor
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

logger = logging.getLogger(__name__)


class SageMakerBERTDeployment:
    """
    Manages BERT sentiment model deployment on AWS SageMaker.
    """
    
    def __init__(self, 
                 role_arn: str,
                 region: str = "us-east-1",
                 instance_type: str = "ml.t2.medium"):
        """
        Initialize SageMaker deployment manager.
        
        Args:
            role_arn: AWS IAM role ARN for SageMaker
            region: AWS region for deployment
            instance_type: EC2 instance type for endpoint
        """
        self.role_arn = role_arn
        self.region = region
        self.instance_type = instance_type
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.endpoint_name = None
        self.model_name = None
        
    def create_model_package(self, 
                           model_name: str,
                           model_data_url: Optional[str] = None) -> str:
        """
        Create SageMaker model package for BERT sentiment analysis.
        
        Args:
            model_name: Name for the SageMaker model
            model_data_url: S3 URL for model artifacts (optional for pre-trained)
            
        Returns:
            Model name in SageMaker
        """
        try:
            # Use PyTorch framework for BERT model
            pytorch_model = PyTorchModel(
                name=model_name,
                model_data=model_data_url,
                role=self.role_arn,
                entry_point='bert_model.py',
                source_dir='ml-models/sentiment',
                framework_version='2.1.0',
                py_version='py310',
                predictor_cls=Predictor,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
            
            self.model_name = model_name
            logger.info(f"Created SageMaker model: {model_name}")
            return model_name
            
        except Exception as e:
            logger.error(f"Failed to create model package: {str(e)}")
            raise
    
    def deploy_endpoint(self, 
                       model_name: str,
                       endpoint_name: str,
                       initial_instance_count: int = 1) -> str:
        """
        Deploy BERT model to SageMaker endpoint.
        
        Args:
            model_name: Name of the SageMaker model
            endpoint_name: Name for the endpoint
            initial_instance_count: Number of instances
            
        Returns:
            Endpoint name
        """
        try:
            # Create endpoint configuration
            endpoint_config_name = f"{endpoint_name}-config"
            
            self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': initial_instance_count,
                        'InstanceType': self.instance_type,
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            
            # Create endpoint
            self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            # Wait for endpoint to be in service
            logger.info(f"Creating endpoint: {endpoint_name}")
            self._wait_for_endpoint(endpoint_name)
            
            self.endpoint_name = endpoint_name
            logger.info(f"Endpoint deployed successfully: {endpoint_name}")
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Failed to deploy endpoint: {str(e)}")
            raise
    
    def _wait_for_endpoint(self, endpoint_name: str, timeout: int = 600) -> None:
        """
        Wait for endpoint to be in service.
        
        Args:
            endpoint_name: Name of the endpoint
            timeout: Maximum wait time in seconds
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.sagemaker_client.describe_endpoint(
                    EndpointName=endpoint_name
                )
                status = response['EndpointStatus']
                
                if status == 'InService':
                    logger.info(f"Endpoint {endpoint_name} is in service")
                    return
                elif status == 'Failed':
                    raise Exception(f"Endpoint {endpoint_name} failed to deploy")
                
                logger.info(f"Endpoint status: {status}, waiting...")
                time.sleep(30)
                
            except Exception as e:
                if "does not exist" in str(e):
                    logger.info("Endpoint not yet created, waiting...")
                    time.sleep(30)
                else:
                    raise
        
        raise TimeoutError(f"Endpoint {endpoint_name} did not become ready within {timeout} seconds")
    
    def create_predictor(self, endpoint_name: str) -> Predictor:
        """
        Create predictor for inference.
        
        Args:
            endpoint_name: Name of the deployed endpoint
            
        Returns:
            SageMaker predictor instance
        """
        predictor = Predictor(
            endpoint_name=endpoint_name,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        return predictor
    
    def delete_endpoint(self, endpoint_name: str) -> None:
        """
        Delete SageMaker endpoint and configuration.
        
        Args:
            endpoint_name: Name of the endpoint to delete
        """
        try:
            # Delete endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Deleted endpoint: {endpoint_name}")
            
            # Delete endpoint configuration
            endpoint_config_name = f"{endpoint_name}-config"
            self.sagemaker_client.delete_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
            logger.info(f"Deleted endpoint configuration: {endpoint_config_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {str(e)}")
            raise


class SentimentInferencePipeline:
    """
    Inference pipeline for BERT sentiment analysis using SageMaker endpoint.
    """
    
    def __init__(self, endpoint_name: str, region: str = "us-east-1"):
        """
        Initialize inference pipeline.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            region: AWS region
        """
        self.endpoint_name = endpoint_name
        self.region = region
        self.predictor = None
        self._initialize_predictor()
    
    def _initialize_predictor(self) -> None:
        """Initialize SageMaker predictor."""
        try:
            self.predictor = Predictor(
                endpoint_name=self.endpoint_name,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
            logger.info(f"Initialized predictor for endpoint: {self.endpoint_name}")
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {str(e)}")
            raise
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for single text.
        
        Args:
            text: Input text for sentiment analysis
            
        Returns:
            Sentiment prediction with score and confidence
        """
        try:
            payload = {"text": text}
            response = self.predictor.predict(payload)
            
            # Validate response
            if "error" in response:
                raise Exception(f"Prediction error: {response['error']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Sentiment prediction failed: {str(e)}")
            raise
    
    def batch_predict_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Batch sentiment predictions
        """
        try:
            payload = {"texts": texts}
            response = self.predictor.predict(payload)
            
            # Validate response
            if "error" in response:
                raise Exception(f"Batch prediction error: {response['error']}")
            
            return response
            
        except Exception as e:
            logger.error(f"Batch sentiment prediction failed: {str(e)}")
            raise
    
    def process_social_media_post(self, post_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process social media post for sentiment analysis.
        
        Args:
            post_data: Social media post data with text and metadata
            
        Returns:
            Processed sentiment data with confidence scoring
        """
        try:
            text = post_data.get("text", "")
            if not text:
                raise ValueError("Post data must contain 'text' field")
            
            # Get sentiment prediction
            prediction = self.predict_sentiment(text)
            
            # Filter by confidence threshold (0.7+ as per requirements)
            confidence_threshold = 0.7
            if prediction["confidence"] < confidence_threshold:
                logger.info(f"Low confidence prediction filtered: {prediction['confidence']}")
                return None
            
            # Prepare result with metadata
            result = {
                "symbol": post_data.get("symbol"),
                "timestamp": post_data.get("timestamp"),
                "text": text,
                "sentiment_score": prediction["sentiment_score"],
                "confidence": prediction["confidence"],
                "classification": prediction["classification"],
                "source": post_data.get("source", "unknown"),
                "post_id": post_data.get("post_id")
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Social media post processing failed: {str(e)}")
            raise


def deploy_bert_sentiment_model(role_arn: str, 
                               model_name: str = "bert-sentiment-model",
                               endpoint_name: str = "bert-sentiment-endpoint") -> str:
    """
    Deploy BERT sentiment model to SageMaker endpoint.
    
    Args:
        role_arn: AWS IAM role ARN
        model_name: Name for the model
        endpoint_name: Name for the endpoint
        
    Returns:
        Deployed endpoint name
    """
    deployment = SageMakerBERTDeployment(role_arn)
    
    # Create model package
    deployment.create_model_package(model_name)
    
    # Deploy endpoint
    endpoint_name = deployment.deploy_endpoint(model_name, endpoint_name)
    
    return endpoint_name