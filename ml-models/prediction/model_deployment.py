"""
SageMaker deployment utilities for LSTM and XGBoost prediction models.
Handles model deployment, versioning, and endpoint management.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
import boto3
from sagemaker import Model, Predictor
from sagemaker.sklearn import SKLearnModel
from sagemaker.tensorflow import TensorFlowModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

logger = logging.getLogger(__name__)


class PredictionModelDeployment:
    """
    Manages deployment of LSTM and XGBoost prediction models on AWS SageMaker.
    Supports model versioning and A/B testing capabilities.
    """
    
    def __init__(self, 
                 role_arn: str,
                 region: str = "us-east-1",
                 instance_type: str = "ml.m5.large"):
        """
        Initialize prediction model deployment manager.
        
        Args:
            role_arn: AWS IAM role ARN for SageMaker
            region: AWS region for deployment
            instance_type: EC2 instance type for endpoint
        """
        self.role_arn = role_arn
        self.region = region
        self.instance_type = instance_type
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        
    def deploy_lstm_model(self, 
                         model_name: str,
                         model_data_url: str,
                         endpoint_name: str,
                         framework_version: str = "2.11") -> str:
        """
        Deploy LSTM model to SageMaker endpoint.
        
        Args:
            model_name: Name for the SageMaker model
            model_data_url: S3 URL for model artifacts
            endpoint_name: Name for the endpoint
            framework_version: TensorFlow framework version
            
        Returns:
            Endpoint name
        """
        try:
            # Create TensorFlow model
            tensorflow_model = TensorFlowModel(
                name=model_name,
                model_data=model_data_url,
                role=self.role_arn,
                entry_point='lstm_model.py',
                source_dir='ml-models/prediction',
                framework_version=framework_version,
                py_version='py39',
                predictor_cls=Predictor,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
            
            # Deploy to endpoint
            predictor = tensorflow_model.deploy(
                initial_instance_count=1,
                instance_type=self.instance_type,
                endpoint_name=endpoint_name
            )
            
            logger.info(f"LSTM model deployed to endpoint: {endpoint_name}")
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Failed to deploy LSTM model: {str(e)}")
            raise
    
    def deploy_xgboost_model(self, 
                           model_name: str,
                           model_data_url: str,
                           endpoint_name: str,
                           sklearn_version: str = "1.0-1") -> str:
        """
        Deploy XGBoost model to SageMaker endpoint.
        
        Args:
            model_name: Name for the SageMaker model
            model_data_url: S3 URL for model artifacts
            endpoint_name: Name for the endpoint
            sklearn_version: Scikit-learn framework version
            
        Returns:
            Endpoint name
        """
        try:
            # Create SKLearn model (XGBoost runs on SKLearn container)
            sklearn_model = SKLearnModel(
                name=model_name,
                model_data=model_data_url,
                role=self.role_arn,
                entry_point='xgboost_model.py',
                source_dir='ml-models/prediction',
                framework_version=sklearn_version,
                py_version='py39',
                predictor_cls=Predictor,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
            
            # Deploy to endpoint
            predictor = sklearn_model.deploy(
                initial_instance_count=1,
                instance_type=self.instance_type,
                endpoint_name=endpoint_name
            )
            
            logger.info(f"XGBoost model deployed to endpoint: {endpoint_name}")
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Failed to deploy XGBoost model: {str(e)}")
            raise
    
    def create_multi_model_endpoint(self, 
                                  endpoint_name: str,
                                  models: List[Dict[str, Any]]) -> str:
        """
        Create endpoint with multiple model variants for A/B testing.
        
        Args:
            endpoint_name: Name for the endpoint
            models: List of model configurations
            
        Returns:
            Endpoint name
        """
        try:
            endpoint_config_name = f"{endpoint_name}-config"
            
            # Create production variants
            production_variants = []
            total_weight = sum(model.get('weight', 1) for model in models)
            
            for i, model_config in enumerate(models):
                variant_name = model_config.get('variant_name', f'variant-{i}')
                model_name = model_config['model_name']
                weight = model_config.get('weight', 1) / total_weight
                
                production_variants.append({
                    'VariantName': variant_name,
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': self.instance_type,
                    'InitialVariantWeight': weight
                })
            
            # Create endpoint configuration
            self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=production_variants
            )
            
            # Create endpoint
            self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            # Wait for endpoint to be ready
            self._wait_for_endpoint(endpoint_name)
            
            logger.info(f"Multi-model endpoint created: {endpoint_name}")
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Failed to create multi-model endpoint: {str(e)}")
            raise
    
    def update_endpoint_weights(self, 
                              endpoint_name: str,
                              variant_weights: Dict[str, float]) -> None:
        """
        Update traffic weights for endpoint variants.
        
        Args:
            endpoint_name: Name of the endpoint
            variant_weights: Dictionary of variant names and weights
        """
        try:
            # Normalize weights
            total_weight = sum(variant_weights.values())
            normalized_weights = {k: v/total_weight for k, v in variant_weights.items()}
            
            # Update endpoint
            desired_weights_and_capacities = [
                {
                    'VariantName': variant_name,
                    'DesiredWeight': weight
                }
                for variant_name, weight in normalized_weights.items()
            ]
            
            self.sagemaker_client.update_endpoint_weights_and_capacities(
                EndpointName=endpoint_name,
                DesiredWeightsAndCapacities=desired_weights_and_capacities
            )
            
            logger.info(f"Updated endpoint weights for {endpoint_name}")
            
        except Exception as e:
            logger.error(f"Failed to update endpoint weights: {str(e)}")
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


class PredictionInferencePipeline:
    """
    Inference pipeline for price prediction using LSTM and XGBoost models.
    Supports ensemble predictions and model comparison.
    """
    
    def __init__(self, 
                 lstm_endpoint: str,
                 xgboost_endpoint: str,
                 region: str = "us-east-1"):
        """
        Initialize prediction inference pipeline.
        
        Args:
            lstm_endpoint: LSTM model endpoint name
            xgboost_endpoint: XGBoost model endpoint name
            region: AWS region
        """
        self.lstm_endpoint = lstm_endpoint
        self.xgboost_endpoint = xgboost_endpoint
        self.region = region
        
        # Initialize predictors
        self.lstm_predictor = Predictor(
            endpoint_name=lstm_endpoint,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
        
        self.xgboost_predictor = Predictor(
            endpoint_name=xgboost_endpoint,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
    
    def predict_lstm(self, recent_data: List[List[float]], horizons: List[int] = [1, 3, 7]) -> Dict[str, Any]:
        """
        Get LSTM predictions for price forecasting.
        
        Args:
            recent_data: Recent price data for LSTM input
            horizons: Prediction horizons in days
            
        Returns:
            LSTM prediction results
        """
        try:
            payload = {
                "recent_data": recent_data,
                "horizons": horizons
            }
            
            response = self.lstm_predictor.predict(payload)
            return response
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {str(e)}")
            raise
    
    def predict_xgboost(self, features: List[float], horizons: List[int] = [1, 3, 7]) -> Dict[str, Any]:
        """
        Get XGBoost predictions for price forecasting.
        
        Args:
            features: Engineered features for XGBoost input
            horizons: Prediction horizons in days
            
        Returns:
            XGBoost prediction results
        """
        try:
            payload = {
                "features": features,
                "horizons": horizons
            }
            
            response = self.xgboost_predictor.predict(payload)
            return response
            
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {str(e)}")
            raise
    
    def ensemble_predict(self, 
                        recent_data: List[List[float]],
                        features: List[float],
                        horizons: List[int] = [1, 3, 7],
                        lstm_weight: float = 0.6,
                        xgboost_weight: float = 0.4) -> Dict[str, Any]:
        """
        Generate ensemble predictions from both LSTM and XGBoost models.
        
        Args:
            recent_data: Recent price data for LSTM
            features: Engineered features for XGBoost
            horizons: Prediction horizons
            lstm_weight: Weight for LSTM predictions
            xgboost_weight: Weight for XGBoost predictions
            
        Returns:
            Ensemble prediction results
        """
        try:
            # Get predictions from both models
            lstm_response = self.predict_lstm(recent_data, horizons)
            xgboost_response = self.predict_xgboost(features, horizons)
            
            # Extract predictions
            lstm_predictions = lstm_response.get("predictions", {})
            xgboost_predictions = xgboost_response.get("predictions", {})
            
            # Create ensemble predictions
            ensemble_predictions = {}
            
            for horizon in horizons:
                if str(horizon) in lstm_predictions and horizon in xgboost_predictions:
                    lstm_pred = lstm_predictions[str(horizon)]
                    xgboost_pred = xgboost_predictions[horizon]
                    
                    # Weighted average
                    ensemble_price = (
                        lstm_pred["predicted_price"] * lstm_weight +
                        xgboost_pred["predicted_price"] * xgboost_weight
                    )
                    
                    # Conservative confidence interval (wider of the two)
                    lstm_range = lstm_pred["confidence_upper"] - lstm_pred["confidence_lower"]
                    xgboost_range = xgboost_pred["confidence_upper"] - xgboost_pred["confidence_lower"]
                    max_range = max(lstm_range, xgboost_range)
                    
                    ensemble_predictions[horizon] = {
                        "predicted_price": ensemble_price,
                        "confidence_lower": ensemble_price - max_range / 2,
                        "confidence_upper": ensemble_price + max_range / 2,
                        "lstm_prediction": lstm_pred["predicted_price"],
                        "xgboost_prediction": xgboost_pred["predicted_price"],
                        "ensemble_weight_lstm": lstm_weight,
                        "ensemble_weight_xgboost": xgboost_weight
                    }
            
            return {
                "ensemble_predictions": ensemble_predictions,
                "individual_predictions": {
                    "lstm": lstm_predictions,
                    "xgboost": xgboost_predictions
                },
                "model_weights": {
                    "lstm": lstm_weight,
                    "xgboost": xgboost_weight
                },
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on both prediction models.
        
        Returns:
            Health check results
        """
        results = {
            "lstm": {"status": "unknown"},
            "xgboost": {"status": "unknown"},
            "overall": {"status": "unknown"}
        }
        
        # Test LSTM
        try:
            test_data = [[1.0, 2.0, 3.0, 4.0, 5.0] for _ in range(60)]  # 60 time steps
            lstm_result = self.predict_lstm(test_data, [1])
            results["lstm"]["status"] = "healthy" if "predictions" in lstm_result else "unhealthy"
        except Exception as e:
            results["lstm"]["status"] = "unhealthy"
            results["lstm"]["error"] = str(e)
        
        # Test XGBoost
        try:
            test_features = [1.0] * 50  # Assuming 50 features
            xgboost_result = self.predict_xgboost(test_features, [1])
            results["xgboost"]["status"] = "healthy" if "predictions" in xgboost_result else "unhealthy"
        except Exception as e:
            results["xgboost"]["status"] = "unhealthy"
            results["xgboost"]["error"] = str(e)
        
        # Overall status
        if results["lstm"]["status"] == "healthy" and results["xgboost"]["status"] == "healthy":
            results["overall"]["status"] = "healthy"
        elif results["lstm"]["status"] == "healthy" or results["xgboost"]["status"] == "healthy":
            results["overall"]["status"] = "partial"
        else:
            results["overall"]["status"] = "unhealthy"
        
        results["timestamp"] = time.time()
        return results


def deploy_prediction_models(role_arn: str,
                           lstm_model_data: str,
                           xgboost_model_data: str) -> Dict[str, str]:
    """
    Deploy both LSTM and XGBoost prediction models.
    
    Args:
        role_arn: AWS IAM role ARN
        lstm_model_data: S3 URL for LSTM model artifacts
        xgboost_model_data: S3 URL for XGBoost model artifacts
        
    Returns:
        Dictionary with endpoint names
    """
    deployment = PredictionModelDeployment(role_arn)
    
    # Deploy LSTM model
    lstm_endpoint = deployment.deploy_lstm_model(
        model_name="lstm-price-predictor",
        model_data_url=lstm_model_data,
        endpoint_name="lstm-prediction-endpoint"
    )
    
    # Deploy XGBoost model
    xgboost_endpoint = deployment.deploy_xgboost_model(
        model_name="xgboost-price-predictor",
        model_data_url=xgboost_model_data,
        endpoint_name="xgboost-prediction-endpoint"
    )
    
    return {
        "lstm_endpoint": lstm_endpoint,
        "xgboost_endpoint": xgboost_endpoint
    }