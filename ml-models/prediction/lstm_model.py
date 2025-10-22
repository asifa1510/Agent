"""
LSTM model for time-series price prediction.
Implements deep learning approach for stock price forecasting.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json

logger = logging.getLogger(__name__)


class LSTMPricePredictor:
    """
    LSTM model for stock price prediction with time-series forecasting.
    Supports 1-day, 3-day, and 7-day price predictions.
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 lstm_units: int = 50,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM price predictor.
        
        Args:
            sequence_length: Number of time steps to look back
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_trained = False
        
    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Build LSTM model architecture.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
        """
        try:
            model = Sequential([
                Input(shape=input_shape),
                LSTM(self.lstm_units, return_sequences=True),
                Dropout(self.dropout_rate),
                LSTM(self.lstm_units, return_sequences=True),
                Dropout(self.dropout_rate),
                LSTM(self.lstm_units),
                Dropout(self.dropout_rate),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mean_squared_error',
                metrics=['mae']
            )
            
            self.model = model
            logger.info(f"Built LSTM model with input shape: {input_shape}")
            
        except Exception as e:
            logger.error(f"Failed to build LSTM model: {str(e)}")
            raise
    
    def prepare_data(self, 
                    price_data: pd.DataFrame,
                    target_column: str = 'close',
                    feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time-series data for LSTM training.
        
        Args:
            price_data: DataFrame with price and feature data
            target_column: Column name for target variable
            feature_columns: List of feature column names
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        try:
            if feature_columns is None:
                feature_columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Select features
            features = price_data[feature_columns].values
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(scaled_features)):
                X.append(scaled_features[i-self.sequence_length:i])
                y.append(scaled_features[i, feature_columns.index(target_column)])
            
            X, y = np.array(X), np.array(y)
            
            logger.info(f"Prepared data: X shape {X.shape}, y shape {y.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Data preparation failed: {str(e)}")
            raise
    
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100,
              batch_size: int = 32) -> Dict[str, Any]:
        """
        Train LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        try:
            if self.model is None:
                self.build_model((X_train.shape[1], X_train.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
            
            # Train model
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            
            history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            logger.info("LSTM model training completed")
            
            return history.history
            
        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained LSTM model.
        
        Args:
            X: Input sequences for prediction
            
        Returns:
            Predicted values
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            predictions = self.model.predict(X)
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {str(e)}")
            raise
    
    def predict_future_prices(self, 
                            recent_data: np.ndarray,
                            horizons: List[int] = [1, 3, 7]) -> Dict[int, Dict[str, float]]:
        """
        Predict future prices for multiple horizons.
        
        Args:
            recent_data: Recent price data for prediction
            horizons: List of prediction horizons in days
            
        Returns:
            Dictionary with predictions for each horizon
        """
        try:
            predictions = {}
            
            # Ensure recent_data has correct shape
            if len(recent_data.shape) == 2:
                recent_data = recent_data.reshape(1, recent_data.shape[0], recent_data.shape[1])
            
            current_sequence = recent_data.copy()
            
            for horizon in horizons:
                horizon_predictions = []
                temp_sequence = current_sequence.copy()
                
                # Predict step by step for the horizon
                for step in range(horizon):
                    pred = self.model.predict(temp_sequence, verbose=0)
                    horizon_predictions.append(pred[0, 0])
                    
                    # Update sequence for next prediction
                    # Shift sequence and add prediction
                    new_row = temp_sequence[0, -1, :].copy()
                    new_row[3] = pred[0, 0]  # Assuming close price is at index 3
                    
                    temp_sequence = np.roll(temp_sequence, -1, axis=1)
                    temp_sequence[0, -1, :] = new_row
                
                # Calculate confidence intervals (simple approach)
                final_prediction = horizon_predictions[-1]
                prediction_std = np.std(horizon_predictions) if len(horizon_predictions) > 1 else 0.01
                
                predictions[horizon] = {
                    "predicted_price": float(final_prediction),
                    "confidence_lower": float(final_prediction - 1.96 * prediction_std),
                    "confidence_upper": float(final_prediction + 1.96 * prediction_std),
                    "prediction_path": [float(p) for p in horizon_predictions]
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Future price prediction failed: {str(e)}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics
        """
        try:
            predictions = self.predict(X_test)
            
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mse)
            
            # Calculate percentage error
            mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
            
            metrics = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape)
            }
            
            logger.info(f"LSTM evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"LSTM evaluation failed: {str(e)}")
            raise
    
    def save_model(self, model_path: str, scaler_path: str) -> None:
        """
        Save trained model and scaler.
        
        Args:
            model_path: Path to save model
            scaler_path: Path to save scaler
        """
        try:
            if self.model is not None:
                self.model.save(model_path)
                logger.info(f"Saved LSTM model to: {model_path}")
            
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to: {scaler_path}")
            
        except Exception as e:
            logger.error(f"Failed to save LSTM model: {str(e)}")
            raise
    
    def load_model(self, model_path: str, scaler_path: str) -> None:
        """
        Load trained model and scaler.
        
        Args:
            model_path: Path to load model from
            scaler_path: Path to load scaler from
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            
            logger.info(f"Loaded LSTM model from: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {str(e)}")
            raise


def model_fn(model_dir: str) -> LSTMPricePredictor:
    """
    SageMaker model loading function for LSTM.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Loaded LSTM model
    """
    model = LSTMPricePredictor()
    model.load_model(
        f"{model_dir}/lstm_model.h5",
        f"{model_dir}/scaler.pkl"
    )
    return model


def input_fn(request_body: str, content_type: str = "application/json") -> Dict:
    """
    SageMaker input processing function for LSTM.
    
    Args:
        request_body: Raw request body
        content_type: Content type of request
        
    Returns:
        Parsed input data
    """
    if content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: Dict, model: LSTMPricePredictor) -> Dict:
    """
    SageMaker prediction function for LSTM.
    
    Args:
        input_data: Parsed input data
        model: Loaded LSTM model
        
    Returns:
        Prediction results
    """
    try:
        if "recent_data" in input_data:
            recent_data = np.array(input_data["recent_data"])
            horizons = input_data.get("horizons", [1, 3, 7])
            
            predictions = model.predict_future_prices(recent_data, horizons)
            
            return {
                "predictions": predictions,
                "model": "lstm",
                "timestamp": pd.Timestamp.now().isoformat()
            }
        else:
            raise ValueError("Input must contain 'recent_data' field")
            
    except Exception as e:
        logger.error(f"LSTM prediction failed: {str(e)}")
        return {"error": str(e)}


def output_fn(prediction: Dict, accept: str = "application/json") -> str:
    """
    SageMaker output processing function for LSTM.
    
    Args:
        prediction: Model prediction results
        accept: Accepted response format
        
    Returns:
        Formatted response
    """
    if accept == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")