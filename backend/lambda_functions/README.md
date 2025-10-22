# Lambda Data Processors

This directory contains the Lambda functions that process data from Kinesis streams for the sentiment-driven trading agent.

## Overview

The data processing pipeline consists of three Lambda functions:

1. **sentiment-processor**: Processes social media data from the `social-media` Kinesis stream
2. **news-processor**: Processes financial news from the `financial-news` Kinesis stream  
3. **market-processor**: Processes market data from the `market-data` Kinesis stream

## Architecture

```
Kinesis Streams → Lambda Processors → DynamoDB Tables
     ↓                    ↓               ↓
Data Sources         Processing        Storage
```

### Error Handling

- **Dead Letter Queue (SQS)**: Failed records are sent to a DLQ for manual inspection
- **Retry Logic**: Built-in retry with exponential backoff
- **Circuit Breaker**: Fallback mechanisms for external API failures
- **Monitoring**: CloudWatch logs and metrics for all functions

## Lambda Functions

### sentiment-processor

**Purpose**: Analyzes sentiment of social media posts using SageMaker BERT model

**Input**: Social media data from `social-media` Kinesis stream
```json
{
  "symbol": "AAPL",
  "content": "Apple stock is looking great today!",
  "source": "twitter",
  "timestamp": "2023-10-22T10:30:00Z",
  "user_id": "user123",
  "post_id": "tweet456"
}
```

**Output**: Sentiment scores stored in `sentiment-scores` DynamoDB table
```json
{
  "symbol": "AAPL",
  "timestamp": 1698062400,
  "score": 0.75,
  "confidence": 0.85,
  "volume": 1,
  "source": "twitter"
}
```

**Features**:
- SageMaker BERT model integration
- Fallback keyword-based sentiment analysis
- Confidence scoring and filtering
- Real-time processing with 30-second SLA

### news-processor

**Purpose**: Scores financial news relevance and impact, filters low-quality content

**Input**: Financial news data from `financial-news` Kinesis stream
```json
{
  "symbol": "AAPL",
  "title": "Apple Reports Strong Q4 Earnings Beat",
  "content": "Apple Inc. reported quarterly earnings...",
  "source": "reuters",
  "url": "https://example.com/news",
  "timestamp": "2023-10-22T10:30:00Z"
}
```

**Output**: News scores stored in `news-scores` DynamoDB table
```json
{
  "symbol": "AAPL",
  "timestamp": 1698062400,
  "relevance_score": 0.85,
  "impact_score": 0.70,
  "sentiment_score": 0.60,
  "sentiment_confidence": 0.75,
  "source": "reuters"
}
```

**Features**:
- Relevance scoring based on company mentions and financial keywords
- Impact scoring using keyword analysis and source credibility
- Sentiment analysis of news content
- Filtering of low-relevance content (< 0.3 threshold)
- Processing within 60-second SLA

### market-processor

**Purpose**: Processes market data and calculates technical indicators

**Input**: Market data from `market-data` Kinesis stream
```json
{
  "symbol": "AAPL",
  "price": 150.25,
  "volume": 1000000,
  "high": 151.0,
  "low": 149.5,
  "open": 150.0,
  "timestamp": "2023-10-22T10:30:00Z"
}
```

**Output**: 
- Raw market data in `market-data` DynamoDB table
- Technical indicators in `technical-indicators` DynamoDB table

**Features**:
- Technical indicator calculations (SMA, RSI, MACD, Bollinger Bands)
- Volume analysis and price change tracking
- Historical data integration for indicator calculations
- Real-time processing and storage

## Deployment

### Prerequisites

1. **Kinesis Streams**: Ensure the required streams exist:
   - `social-media`
   - `financial-news` 
   - `market-data`

2. **DynamoDB Tables**: Create the required tables:
   - `sentiment-scores`
   - `news-scores`
   - `market-data`
   - `technical-indicators`

3. **SageMaker Endpoints**: Deploy BERT model endpoint for sentiment analysis

### Deploy Lambda Functions

```bash
# Deploy all functions
python deploy.py --region us-east-1

# Dry run to see what would be deployed
python deploy.py --dry-run

# Deploy to specific region
python deploy.py --region us-west-2
```

The deployment script will:
1. Create Dead Letter Queue (SQS)
2. Create Lambda execution role with necessary permissions
3. Deploy all three Lambda functions
4. Set up Kinesis event source mappings
5. Configure error handling and monitoring

### Test Deployment

```bash
# Run local tests
python test_processors.py

# Test individual processors
python -c "from test_processors import test_sentiment_processor; test_sentiment_processor()"
```

## Configuration

### Environment Variables

Each Lambda function uses these environment variables:

**sentiment-processor**:
- `BERT_ENDPOINT`: SageMaker BERT endpoint name
- `SENTIMENT_TABLE`: DynamoDB table for sentiment scores
- `DLQ_QUEUE_URL`: Dead Letter Queue URL

**news-processor**:
- `BERT_ENDPOINT`: SageMaker BERT endpoint name  
- `NEWS_SCORES_TABLE`: DynamoDB table for news scores
- `DLQ_QUEUE_URL`: Dead Letter Queue URL

**market-processor**:
- `MARKET_DATA_TABLE`: DynamoDB table for market data
- `TECHNICAL_INDICATORS_TABLE`: DynamoDB table for indicators
- `DLQ_QUEUE_URL`: Dead Letter Queue URL

### Performance Settings

| Function | Memory | Timeout | Batch Size |
|----------|--------|---------|------------|
| sentiment-processor | 512 MB | 5 min | 10 |
| news-processor | 512 MB | 5 min | 10 |
| market-processor | 256 MB | 3 min | 10 |

## Monitoring

### CloudWatch Metrics

- **Invocations**: Number of function executions
- **Duration**: Execution time per invocation
- **Errors**: Failed executions
- **Throttles**: Rate-limited executions

### CloudWatch Logs

Each function logs:
- Processing statistics (success/failure counts)
- Error details for failed records
- Performance metrics
- DLQ message details

### Alarms

Recommended CloudWatch alarms:
- Error rate > 5%
- Duration > 80% of timeout
- DLQ message count > 0

## Troubleshooting

### Common Issues

1. **SageMaker Endpoint Not Found**
   - Verify BERT endpoint is deployed and active
   - Check endpoint name in environment variables

2. **DynamoDB Access Denied**
   - Verify Lambda execution role has DynamoDB permissions
   - Check table names match environment variables

3. **Kinesis Stream Not Found**
   - Ensure streams are created before deploying functions
   - Verify stream names match configuration

4. **High Error Rates**
   - Check CloudWatch logs for specific error messages
   - Monitor DLQ for failed records
   - Verify external API connectivity

### Performance Optimization

1. **Memory Allocation**: Increase memory for CPU-intensive operations
2. **Batch Size**: Adjust based on processing time and throughput needs
3. **Parallelization**: Use multiple shards for high-volume streams
4. **Caching**: Implement caching for frequently accessed data

## Security

### IAM Permissions

The Lambda execution role includes minimal required permissions:
- Kinesis: Read from streams
- DynamoDB: Read/write to tables
- SageMaker: Invoke endpoints
- Bedrock: Invoke models
- SQS: Send messages to DLQ
- CloudWatch: Write logs and metrics

### Data Protection

- All data in transit is encrypted (HTTPS/TLS)
- DynamoDB tables use encryption at rest
- Sensitive data is not logged
- TTL configured for automatic data cleanup

## Development

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python test_processors.py

# Test individual functions
python sentiment_processor.py
python news_processor.py  
python market_processor.py
```

### Adding New Processors

1. Create new processor file following the existing pattern
2. Add configuration to `deploy_config.py`
3. Update deployment script
4. Add tests to `test_processors.py`
5. Update this README

## Requirements

See `requirements.txt` for Python dependencies. The Lambda functions use:
- `boto3`: AWS SDK
- `botocore`: AWS core library
- Built-in Python libraries for data processing