"""
Configuration settings for the Sentiment Trading Agent backend
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_title: str = "Sentiment Trading Agent API"
    api_description: str = "AI trading agent integrating sentiment, news, and market data"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # AWS Configuration
    aws_region: str = "us-east-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    
    # DynamoDB Tables
    sentiment_table: str = "sentiment-scores"
    predictions_table: str = "predictions"
    trades_table: str = "trades"
    explanations_table: str = "explanations"
    portfolio_table: str = "portfolio"
    
    # Kinesis Streams
    social_media_stream: str = "social-media"
    financial_news_stream: str = "financial-news"
    market_data_stream: str = "market-data"
    
    # SageMaker Endpoints
    bert_endpoint: str = "sentiment-bert-endpoint"
    lstm_endpoint: str = "price-lstm-endpoint"
    xgboost_endpoint: str = "price-xgboost-endpoint"
    
    # SageMaker Endpoint Names (for prediction service)
    lstm_endpoint_name: str = "lstm-price-prediction"
    xgboost_endpoint_name: str = "xgboost-price-prediction"
    
    # Risk Management
    max_position_allocation: float = 0.05  # 5% max per position
    stop_loss_threshold: float = 0.02  # 2% stop loss
    
    # External APIs
    twitter_bearer_token: Optional[str] = None
    news_api_key: Optional[str] = None
    
    # AWS Bedrock Configuration
    bedrock_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    bedrock_max_tokens: int = 1000
    bedrock_temperature: float = 0.3
    explanation_timeout_seconds: int = 15
    
    # API Rate Limits (requests per minute)
    twitter_rate_limit: int = 20
    news_api_rate_limit: int = 30
    yahoo_finance_rate_limit: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()