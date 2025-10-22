# Sentiment Trading Agent - Project Structure

## Overview
This document outlines the complete project structure for the Sentiment Trading Agent system.

## Directory Structure

```
sentiment-trading-agent/
├── backend/                    # FastAPI backend services
│   ├── api/                   # API route handlers
│   ├── models/                # Pydantic data models
│   │   └── data_models.py     # Core data models (SentimentScore, PricePrediction, Trade, etc.)
│   ├── repositories/          # Data access layer
│   ├── services/              # Business logic services
│   ├── config.py              # Application configuration
│   ├── main.py                # FastAPI application entry point
│   ├── requirements.txt       # Python dependencies
│   └── .env.example          # Environment variables template
│
├── frontend/                  # React dashboard
│   ├── public/               # Static assets
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── hooks/           # Custom React hooks
│   │   ├── services/        # API clients
│   │   ├── types/           # TypeScript type definitions
│   │   │   └── index.ts     # Core TypeScript interfaces
│   │   ├── utils/           # Utility functions
│   │   ├── config/          # Frontend configuration
│   │   └── App.tsx          # Main React component
│   ├── package.json         # Node.js dependencies
│   └── .env.example        # Environment variables template
│
├── ml-models/                # Machine learning models
│   ├── sentiment/           # BERT sentiment analysis models
│   ├── prediction/          # LSTM/XGBoost prediction models
│   ├── utils/              # ML utilities and helpers
│   ├── requirements.txt    # ML dependencies
│   └── README.md           # ML models documentation
│
└── .kiro/                  # Kiro specification files
    └── specs/
        └── sentiment-trading-agent/
            ├── requirements.md  # Feature requirements
            ├── design.md       # System design
            └── tasks.md        # Implementation tasks
```

## Core Data Models

### Backend (Python/Pydantic)
- `SentimentScore`: Social media sentiment analysis results
- `PricePrediction`: ML-generated price forecasts
- `Trade`: Trading transaction records
- `TradeExplanation`: AI-generated trade explanations
- `PortfolioPosition`: Current portfolio positions
- `RiskMetrics`: Portfolio risk calculations

### Frontend (TypeScript)
- Matching TypeScript interfaces for all backend models
- Additional UI-specific types for charts and API responses
- Configuration types for application settings

## Configuration

### Backend Configuration (`backend/config.py`)
- AWS service endpoints and credentials
- DynamoDB table names
- Kinesis stream names
- SageMaker endpoint configurations
- Risk management parameters
- External API credentials

### Frontend Configuration (`frontend/src/config/index.ts`)
- API base URLs
- Chart update intervals
- WebSocket configuration
- UI theme settings
- Data refresh intervals

## Key Features Implemented

✅ **Project Structure**: Complete directory organization
✅ **Data Models**: Comprehensive Pydantic and TypeScript models
✅ **FastAPI Setup**: Basic application with CORS and health endpoints
✅ **Configuration Management**: Environment-based settings
✅ **Type Safety**: Full TypeScript interface definitions
✅ **Development Setup**: Requirements files and example environments

## Next Steps

The project structure and core interfaces are now complete. The next tasks will involve:
1. Setting up data ingestion pipeline (Kinesis streams)
2. Implementing ML processing components
3. Building backend services and APIs
4. Creating the React dashboard components
5. Integrating all system components

## Requirements Satisfied

This implementation satisfies requirements:
- **8.1**: AWS infrastructure foundation with proper service configuration
- **8.2**: System architecture with scalable component organization