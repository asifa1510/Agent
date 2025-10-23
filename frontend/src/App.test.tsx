import React from 'react';
import { render, screen } from '@testing-library/react';
import App from './App';

// Mock the API client to avoid network calls in tests
jest.mock('./services/api', () => ({
  apiClient: {
    getSentimentScores: jest.fn().mockResolvedValue([]),
    getPredictions: jest.fn().mockResolvedValue([]),
    getTrades: jest.fn().mockResolvedValue([]),
    getPortfolioPositions: jest.fn().mockResolvedValue([]),
    getRiskMetrics: jest.fn().mockResolvedValue({
      total_value: 0,
      total_pnl: 0,
      max_drawdown: 0,
      volatility: 0,
      var_95: 0,
      positions_count: 0,
      timestamp: Date.now() / 1000
    }),
  }
}));

test('renders sentiment trading agent header', () => {
  render(<App />);
  const headerElement = screen.getByText(/Sentiment Trading Agent/i);
  expect(headerElement).toBeInTheDocument();
});

test('renders navigation menu', () => {
  render(<App />);
  const dashboardLink = screen.getByRole('link', { name: /📊 Dashboard/i });
  const sentimentLink = screen.getByRole('link', { name: /💭 Sentiment/i });
  const predictionsLink = screen.getByRole('link', { name: /📈 Predictions/i });
  const tradesLink = screen.getByRole('link', { name: /💰 Trades/i });
  const portfolioLink = screen.getByRole('link', { name: /📋 Portfolio/i });
  
  expect(dashboardLink).toBeInTheDocument();
  expect(sentimentLink).toBeInTheDocument();
  expect(predictionsLink).toBeInTheDocument();
  expect(tradesLink).toBeInTheDocument();
  expect(portfolioLink).toBeInTheDocument();
});

test('renders symbol selector', () => {
  render(<App />);
  const symbolSelectorTitle = screen.getByText(/Symbol Selection/i);
  expect(symbolSelectorTitle).toBeInTheDocument();
});