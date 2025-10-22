"""
Configuration settings for BERT sentiment analysis model.
"""

import os
from typing import Dict, Any

# Model configuration
BERT_MODEL_CONFIG = {
    "model_name": "nlptown/bert-base-multilingual-uncased-sentiment",
    "max_length": 512,
    "confidence_threshold": 0.7,
    "batch_size": 10,
    "device": "auto"  # auto-detect cuda/cpu
}

# SageMaker deployment configuration
SAGEMAKER_CONFIG = {
    "instance_type": "ml.t2.medium",
    "initial_instance_count": 1,
    "framework_version": "2.1.0",
    "py_version": "py310",
    "region": os.getenv("AWS_REGION", "us-east-1")
}

# Processing configuration
PROCESSING_CONFIG = {
    "confidence_threshold": 0.7,
    "batch_size": 10,
    "max_workers": 4,
    "time_window_minutes": 5,
    "max_text_length": 512
}

# Endpoint configuration
ENDPOINT_CONFIG = {
    "model_name": "bert-sentiment-model",
    "endpoint_name": "bert-sentiment-endpoint",
    "endpoint_config_name": "bert-sentiment-config"
}

def get_model_config() -> Dict[str, Any]:
    """Get BERT model configuration."""
    return BERT_MODEL_CONFIG.copy()

def get_sagemaker_config() -> Dict[str, Any]:
    """Get SageMaker deployment configuration."""
    return SAGEMAKER_CONFIG.copy()

def get_processing_config() -> Dict[str, Any]:
    """Get processing configuration."""
    return PROCESSING_CONFIG.copy()

def get_endpoint_config() -> Dict[str, Any]:
    """Get endpoint configuration."""
    return ENDPOINT_CONFIG.copy()