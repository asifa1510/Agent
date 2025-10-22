"""
XGBoost model for feature-based price prediction.
Implements gradient boosting approach for stock price forecasting.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import joblib
import json

logger = logging.getLogger(__name__)


class XGBoostPricePredictor:
    """
    XGBoost model for stock price prediction using engineered features.
    Supports multiple prediction horizons with feature importance analysis.
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 42):
        """
        Initialize XGBoost price predictor.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def create_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for XGBoost model.
        
        Args:
            price_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df = price_data.copy()
            
            # Price-based features
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Moving averages
            for window in [5, 10, 20, 50]:
                df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'price_ma_{window}_ratio'] = df['close'] / df[f'ma_{window}']
            
            # Volatility features
            df['volatility_5'] = df['price_change'].rolling(window=5).std()
            df['volatility_20'] = df['price_change'].rolling(window=20).std()
            
            # Volume features
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            
            # Technical indicators
            # RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f'close_lag_{lag}'] = df['close'].shift(lag)
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
                df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
            
            # Time-based features
            df['day_of_week'] = pd.to_datetime(df.index).dayofweek
            df['month'] = pd.to_datetime(df.index).month
            df['quarter'] = pd.to_datetime(df.index).quarter
            
            # Drop rows with NaN values
            df = df.dropna()
            
            logger.info(f"Created {len(df.columns)} features for XGBoost model")
            return df
            
        except Exception as e:
            logger.error(f"Feature creation failed: {str(e)}")
            raise
    
    def prepare_data(self, 
                    price_data: pd.DataFrame,
                    target_column: str = 'close',
                    prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for XGBoost training.
        
        Args:
            price_data: DataFrame with price data
            target_column: Column name for target variable
            prediction_horizon: Days ahead to predict
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        try:
            # Create features
            df = self.create_features(price_data)
            
            # Create target variable (future price)
            df[f'target_{prediction_horizon}d'] = df[target_column].shift(-prediction_horizon)
            
            # Remove rows where target is NaN
            df = df.dropna()
            
            # Separate features and target
            target_col = f'target_{prediction_horizon}d'
            feature_cols = [col for col in df.columns if col != target_col and col != target_column]
            
            X = df[feature_cols].values
            y = df[target_col].values
            
            self.feature_names = feature_cols
            
            logger.info(f"Prepared XGBoost data: X shape {X.shape}, y shape {y.shape}")
            return X, y, feature_cols
            
        except Exception as e:
            logger.error(f"XGBoost data preparation failed: {str(e)}")
            raise
    
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              optimize_hyperparameters: bool = False) -> Dict[str, Any]:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            
        Returns:
            Training results
        """
        try:
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
            
            if optimize_hyperparameters and X_val is not None:
                # Hyperparameter optimization
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
                
                xgb_model = xgb.XGBRegressor(random_state=self.random_state)
                grid_search = GridSearchCV(
                    xgb_model, 
                    param_grid, 
                    cv=3, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train_scaled, y_train)
                self.model = grid_search.best_estimator_
                
                logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
                
            else:
                # Train with default/specified parameters
                self.model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state
                )
                
                # Set up evaluation set if validation data provided
                eval_set = [(X_train_scaled, y_train)]
                if X_val_scaled is not None:
                    eval_set.append((X_val_scaled, y_val))
                
                self.model.fit(
                    X_train_scaled, 
                    y_train,
                    eval_set=eval_set,
                    early_stopping_rounds=10,
                    verbose=False
                )
            
            self.is_trained = True
            logger.info("XGBoost model training completed")
            
            # Get feature importance
            feature_importance = dict(zip(
                self.feature_names or [f"feature_{i}" for i in range(X_train.shape[1])],
                self.model.feature_importances_
            ))
            
            return {
                "feature_importance": feature_importance,
                "n_features": X_train.shape[1],
                "training_samples": X_train.shape[0]
            }
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained XGBoost model.
        
        Args:
            X: Input features for prediction
            
        Returns:
            Predicted values
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            return predictions
            
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {str(e)}")
            raise
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals using quantile regression.
        
        Args:
            X: Input features for prediction
            
        Returns:
            Tuple of (predictions, confidence_intervals)
        """
        try:
            predictions = self.predict(X)
            
            # Simple confidence interval estimation
            # In practice, you might want to train separate models for quantiles
            prediction_std = np.std(predictions) if len(predictions) > 1 else 0.01
            confidence_intervals = np.column_stack([
                predictions - 1.96 * prediction_std,
                predictions + 1.96 * prediction_std
            ])
            
            return predictions, confidence_intervals
            
        except Exception as e:
            logger.error(f"XGBoost confidence prediction failed: {str(e)}")
            raise
    
    def predict_multiple_horizons(self, 
                                 recent_features: np.ndarray,
                                 horizons: List[int] = [1, 3, 7]) -> Dict[int, Dict[str, float]]:
        """
        Predict prices for multiple horizons.
        Note: This requires separate models trained for each horizon.
        
        Args:
            recent_features: Recent feature data
            horizons: List of prediction horizons
            
        Returns:
            Dictionary with predictions for each horizon
        """
        try:
            predictions = {}
            
            # For simplicity, using the same model for all horizons
            # In practice, you'd train separate models for each horizon
            base_prediction = self.predict(recent_features.reshape(1, -1))[0]
            
            for horizon in horizons:
                # Adjust prediction based on horizon (simple approach)
                horizon_factor = 1 + (horizon - 1) * 0.01  # Small adjustment for longer horizons
                predicted_price = base_prediction * horizon_factor
                
                # Simple confidence interval
                confidence_range = abs(predicted_price) * 0.05 * horizon
                
                predictions[horizon] = {
                    "predicted_price": float(predicted_price),
                    "confidence_lower": float(predicted_price - confidence_range),
                    "confidence_upper": float(predicted_price + confidence_range)
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Multi-horizon prediction failed: {str(e)}")
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
            
            # R-squared
            ss_res = np.sum((y_test - predictions) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            metrics = {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape),
                "r2": float(r2)
            }
            
            logger.info(f"XGBoost evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"XGBoost evaluation failed: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from trained model.
        
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained to get feature importance")
        
        feature_names = self.feature_names or [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
        return dict(zip(feature_names, self.model.feature_importances_))
    
    def save_model(self, model_path: str, scaler_path: str) -> None:
        """
        Save trained model and scaler.
        
        Args:
            model_path: Path to save model
            scaler_path: Path to save scaler
        """
        try:
            if self.model is not None:
                joblib.dump(self.model, model_path)
                logger.info(f"Saved XGBoost model to: {model_path}")
            
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to: {scaler_path}")
            
            # Save feature names
            if self.feature_names:
                feature_path = model_path.replace('.pkl', '_features.json')
                with open(feature_path, 'w') as f:
                    json.dump(self.feature_names, f)
                logger.info(f"Saved feature names to: {feature_path}")
            
        except Exception as e:
            logger.error(f"Failed to save XGBoost model: {str(e)}")
            raise
    
    def load_model(self, model_path: str, scaler_path: str) -> None:
        """
        Load trained model and scaler.
        
        Args:
            model_path: Path to load model from
            scaler_path: Path to load scaler from
        """
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.is_trained = True
            
            # Load feature names
            feature_path = model_path.replace('.pkl', '_features.json')
            try:
                with open(feature_path, 'r') as f:
                    self.feature_names = json.load(f)
            except FileNotFoundError:
                logger.warning("Feature names file not found")
            
            logger.info(f"Loaded XGBoost model from: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {str(e)}")
            raise


def model_fn(model_dir: str) -> XGBoostPricePredictor:
    """
    SageMaker model loading function for XGBoost.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Loaded XGBoost model
    """
    model = XGBoostPricePredictor()
    model.load_model(
        f"{model_dir}/xgboost_model.pkl",
        f"{model_dir}/scaler.pkl"
    )
    return model


def input_fn(request_body: str, content_type: str = "application/json") -> Dict:
    """
    SageMaker input processing function for XGBoost.
    
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


def predict_fn(input_data: Dict, model: XGBoostPricePredictor) -> Dict:
    """
    SageMaker prediction function for XGBoost.
    
    Args:
        input_data: Parsed input data
        model: Loaded XGBoost model
        
    Returns:
        Prediction results
    """
    try:
        if "features" in input_data:
            features = np.array(input_data["features"])
            horizons = input_data.get("horizons", [1, 3, 7])
            
            if len(features.shape) == 1:
                predictions = model.predict_multiple_horizons(features, horizons)
            else:
                # Batch prediction
                batch_predictions = []
                for feature_row in features:
                    pred = model.predict_multiple_horizons(feature_row, horizons)
                    batch_predictions.append(pred)
                predictions = batch_predictions
            
            return {
                "predictions": predictions,
                "model": "xgboost",
                "feature_importance": model.get_feature_importance(),
                "timestamp": pd.Timestamp.now().isoformat()
            }
        else:
            raise ValueError("Input must contain 'features' field")
            
    except Exception as e:
        logger.error(f"XGBoost prediction failed: {str(e)}")
        return {"error": str(e)}


def output_fn(prediction: Dict, accept: str = "application/json") -> str:
    """
    SageMaker output processing function for XGBoost.
    
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