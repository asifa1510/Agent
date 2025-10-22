# ML Models

This directory contains machine learning model implementations and configurations for the sentiment trading agent.

## Structure

- `sentiment/` - BERT-based sentiment analysis models
- `prediction/` - LSTM and XGBoost price prediction models
- `utils/` - Common utilities for model training and inference
- `notebooks/` - Jupyter notebooks for model development and analysis

## Models

### Sentiment Analysis
- Pre-trained BERT model for social media sentiment classification
- Deployed on AWS SageMaker for real-time inference

### Price Prediction
- LSTM model for time-series price forecasting
- XGBoost model for feature-based predictions
- Both deployed on SageMaker with versioning support

## Deployment

Models are deployed to AWS SageMaker endpoints for scalable inference.