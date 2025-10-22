# Requirements Document

## Introduction
AI trading agent integrating sentiment, news, and market data for autonomous trading decisions.

## Glossary
- **Trading_Agent**: Core AI system executing trades
- **Sentiment_Analyzer**: NLP component for social media sentiment
- **News_Processor**: Financial news analysis component
- **Risk_Controller**: Risk and position management
- **Explanation_Engine**: AWS Bedrock trade explanations
- **Dashboard**: React UI for insights

## Requirements

### Requirement 1
**User Story:** As a retail investor, I want sentiment analysis, so that I can gauge market mood.
1. WHEN X/Twitter data is received, THE Sentiment_Analyzer SHALL process within 30 seconds
2. THE Sentiment_Analyzer SHALL classify with 0.7+ confidence scores
3. THE Sentiment_Analyzer SHALL update every 5 minutes

### Requirement 2
**User Story:** As a retail investor, I want news scoring, so that I can focus on impact.
1. THE News_Processor SHALL score relevance within 60 seconds
2. THE News_Processor SHALL assign 0.0-1.0 impact scores
3. THE News_Processor SHALL filter below 0.3 relevance

### Requirement 3
**User Story:** As a retail investor, I want price predictions, so that I can understand trends.
1. THE Trading_Agent SHALL generate LSTM/XGBoost predictions
2. THE Trading_Agent SHALL provide 1/3/7-day forecasts
3. THE Trading_Agent SHALL update hourly

### Requirement 4
**User Story:** As a retail investor, I want autonomous trading, so that I can protect capital.
1. THE Risk_Controller SHALL validate position sizes
2. THE Trading_Agent SHALL execute within 10 seconds
3. THE Risk_Controller SHALL limit 5% allocation per position

### Requirement 5
**User Story:** As a retail investor, I want explainable decisions, so that I understand reasoning.
1. THE Explanation_Engine SHALL generate explanations within 15 seconds
2. THE Dashboard SHALL display with visualizations

### Requirement 6
**User Story:** As a retail investor, I want simulation, so that I can evaluate performance.
1. THE Portfolio_Simulator SHALL run 1000+ Monte Carlo iterations
2. THE Portfolio_Simulator SHALL provide Sharpe/drawdown metrics

### Requirement 7
**User Story:** As a retail investor, I want a dashboard, so that I can monitor insights.
1. THE Dashboard SHALL display real-time data
2. THE Dashboard SHALL update every 30 seconds

### Requirement 8
**User Story:** As an administrator, I want AWS infrastructure, so that system scales.
1. THE Data_Pipeline SHALL process 10,000+ posts/hour via Kinesis
2. THE system SHALL maintain 99.5% uptime