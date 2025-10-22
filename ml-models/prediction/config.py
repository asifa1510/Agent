"""
Configuration settings for LSTM and XGBoost prediction models.
"""

import os
from typing import Dict, Any, List

# LSTM model configuration
LSTM_CONFIG = {
    "sequence_length": 60,
    "lstm_units": 50,
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 32,
    "validation_split": 0.2
}

# XGBoost model configuration
XGBOOST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

# Feature engineering configuration
FEATURE_CONFIG = {
    "ma_windows": [5, 10, 20, 50],
    "volatility_windows": [5, 20],
    "volume_windows": [5, 20],
    "lag_periods": [1, 2, 3, 5],
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2
}

# SageMaker deployment configuration
SAGEMAKER_PREDICTION_CONFIG = {
    "instance_type": "ml.m5.large",
    "initial_instance_count": 1,
    "tensorflow_version": "2.11",
    "sklearn_version": "1.0-1",
    "py_version": "py39",
    "region": os.getenv("AWS_REGION", "us-east-1")
}

# Prediction configuration
PREDICTION_CONFIG = {
    "horizons": [1, 3, 7],  # Days
    "ensemble_weights": {
        "lstm": 0.6,
        "xgboost": 0.4
    },
    "confidence_level": 0.95,
    "update_frequency_hours": 1
}

# Model versioning configuration
VERSIONING_CONFIG = {
    "model_registry": "prediction-models",
    "approval_status": "PendingManualApproval",
    "model_package_group": "price-prediction-models"
}

# Endpoint configuration
ENDPOINT_CONFIG = {
    "lstm_model_name": "lstm-price-predictor",
    "xgboost_model_name": "xgboost-price-predictor",
    "lstm_endpoint_name": "lstm-prediction-endpoint",
    "xgboost_endpoint_name": "xgboost-prediction-endpoint",
    "ensemble_endpoint_name": "ensemble-prediction-endpoint"
}

def get_lstm_config() -> Dict[str, Any]:
    """Get LSTM model configuration."""
    return LSTM_CONFIG.copy()

def get_xgboost_config() -> Dict[str, Any]:
    """Get XGBoost model configuration."""
    return XGBOOST_CONFIG.copy()

def get_feature_config() -> Dict[str, Any]:
    """Get feature engineering configuration."""
    return FEATURE_CONFIG.copy()

def get_sagemaker_config() -> Dict[str, Any]:
    """Get SageMaker deployment configuration."""
    return SAGEMAKER_PREDICTION_CONFIG.copy()

def get_prediction_config() -> Dict[str, Any]:
    """Get prediction configuration."""
    return PREDICTION_CONFIG.copy()

def get_endpoint_config() -> Dict[str, Any]:
    """Get endpoint configuration."""
    return ENDPOINT_CONFIG.copy()

def get_versioning_config() -> Dict[str, Any]:
    """Get model versioning configuration."""
    return VERSIONING_CONFIG.copy()