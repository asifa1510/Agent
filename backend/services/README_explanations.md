# AWS Bedrock Explanation Service

## Overview

The Explanation Service provides AI-powered explanations for trading decisions and portfolio performance using AWS Bedrock. It integrates with Large Language Models (LLMs) to generate human-readable explanations that help users understand the reasoning behind automated trading decisions.

## Features

- **Trade Explanations**: Generate detailed explanations for individual buy/sell decisions
- **Portfolio Analysis**: Provide comprehensive portfolio performance analysis
- **Context Injection**: Incorporate sentiment data, price predictions, and market data
- **Batch Processing**: Generate explanations for multiple trades efficiently
- **Confidence Scoring**: Provide confidence metrics for generated explanations

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Explanation    │    │   AWS Bedrock   │
│   Endpoints     │───▶│   Service        │───▶│   Claude/GPT    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   Context Data   │
                       │   - Sentiment    │
                       │   - Predictions  │
                       │   - Market Data  │
                       └──────────────────┘
```

## Configuration

Add the following environment variables to your `.env` file:

```env
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_MAX_TOKENS=1000
BEDROCK_TEMPERATURE=0.3
EXPLANATION_TIMEOUT_SECONDS=15
```

## API Endpoints

### Generate Trade Explanation
```http
POST /explanations/trade
```

Generate explanation for a single trade decision.

**Request Body:**
```json
{
  "trade": {
    "id": "trade-001",
    "symbol": "AAPL",
    "action": "buy",
    "quantity": 100,
    "price": 150.25,
    "signal_strength": 0.85,
    "timestamp": 1703097600
  },
  "include_context": true,
  "sentiment_data": {
    "score": 0.65,
    "confidence": 0.82,
    "volume": 1250,
    "source": "twitter"
  },
  "prediction_data": {
    "model": "lstm",
    "predicted_price": 155.50,
    "confidence_lower": 148.00,
    "confidence_upper": 163.00,
    "horizon": "3d"
  },
  "market_data": {
    "current_price": 150.25,
    "volume": 45000000,
    "price_change_percent": 2.1,
    "volatility": 22.5
  }
}
```

### Generate Portfolio Explanation
```http
POST /explanations/portfolio
```

Generate explanation for portfolio performance and composition.

### Batch Generate Explanations
```http
POST /explanations/batch
```

Generate explanations for multiple trades in batch.

### Service Status
```http
GET /explanations/status
```

Get current status of the explanation service including Bedrock connection health.

### Test Connection
```http
GET /explanations/test-connection
```

Test connection to AWS Bedrock service.

## Usage Examples

### Basic Trade Explanation

```python
from services.explanation_service import ExplanationService
from models.data_models import Trade

# Initialize service
service = ExplanationService()

# Create trade
trade = Trade(
    id="trade-001",
    symbol="AAPL",
    action="buy",
    quantity=100,
    price=150.25,
    signal_strength=0.85,
    timestamp=1703097600
)

# Generate explanation
explanation = await service.generate_trade_explanation(trade)
print(f"Explanation: {explanation.explanation}")
print(f"Confidence: {explanation.confidence}")
```

### Portfolio Analysis

```python
# Generate portfolio explanation
positions = [...]  # List of PortfolioPosition objects
risk_metrics = RiskMetrics(...)  # Risk metrics object

result = await service.generate_portfolio_explanation(
    positions=positions,
    risk_metrics=risk_metrics
)

print(f"Portfolio Analysis: {result['explanation']}")
```

## Prompt Templates

The service uses structured prompt templates to ensure consistent and informative explanations:

### Trade Explanation Template
- Trade details (symbol, action, quantity, price, signal strength)
- Supporting data (sentiment, predictions, market data)
- Risk/reward analysis
- Clear reasoning for the decision

### Portfolio Explanation Template
- Portfolio overview (total value, P&L, positions)
- Risk metrics analysis
- Top performing/underperforming positions
- Diversification assessment
- Improvement recommendations

## Error Handling

The service includes comprehensive error handling:

- **Connection Errors**: Graceful handling of AWS Bedrock connection issues
- **Rate Limiting**: Automatic retry with exponential backoff
- **Timeout Handling**: Configurable timeout for explanation generation
- **Validation Errors**: Input validation for trade and portfolio data
- **Fallback Mechanisms**: Degraded service mode when Bedrock is unavailable

## Performance Considerations

- **Response Time**: Target <15 seconds for explanation generation (configurable)
- **Batch Processing**: Efficient handling of multiple explanation requests
- **Caching**: Consider implementing caching for frequently requested explanations
- **Rate Limits**: Respect AWS Bedrock service limits and quotas

## Testing

Run the test suite to verify functionality:

```bash
cd backend
python test_explanation_service.py
```

The test suite includes:
- Connection testing
- Trade explanation generation
- Portfolio analysis
- Batch processing
- Service status checks

## Monitoring

Monitor the following metrics:
- Explanation generation success rate
- Average response time
- Bedrock API usage and costs
- Error rates and types
- User satisfaction with explanations

## Security

- AWS credentials are managed through IAM roles and policies
- No sensitive data is logged in explanations
- Input validation prevents injection attacks
- Rate limiting prevents abuse

## Future Enhancements

- Support for additional LLM models
- Custom prompt templates per user
- Explanation quality scoring
- Multi-language support
- Integration with feedback systems