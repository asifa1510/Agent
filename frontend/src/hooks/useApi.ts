import { useCallback } from 'react';
import { useAppContext } from '../context/AppContext';
import { apiClient } from '../services/api';

export function useApi() {
  const { dispatch } = useAppContext();

  // Sentiment data hooks
  const fetchSentimentScores = useCallback(async (symbol?: string) => {
    dispatch({ type: 'SET_LOADING', payload: { key: 'sentiment', value: true } });
    try {
      const data = await apiClient.getSentimentScores(symbol);
      dispatch({ type: 'SET_SENTIMENT_SCORES', payload: data });
      dispatch({ type: 'CLEAR_ERROR' });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: `Failed to fetch sentiment data: ${error}` });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { key: 'sentiment', value: false } });
    }
  }, [dispatch]);

  // Predictions data hooks
  const fetchPredictions = useCallback(async (symbol?: string) => {
    dispatch({ type: 'SET_LOADING', payload: { key: 'predictions', value: true } });
    try {
      const data = await apiClient.getPredictions(symbol);
      dispatch({ type: 'SET_PREDICTIONS', payload: data });
      dispatch({ type: 'CLEAR_ERROR' });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: `Failed to fetch predictions: ${error}` });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { key: 'predictions', value: false } });
    }
  }, [dispatch]);

  // Trades data hooks
  const fetchTrades = useCallback(async (limit: number = 100) => {
    dispatch({ type: 'SET_LOADING', payload: { key: 'trades', value: true } });
    try {
      const data = await apiClient.getTrades(limit);
      dispatch({ type: 'SET_TRADES', payload: data });
      dispatch({ type: 'CLEAR_ERROR' });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: `Failed to fetch trades: ${error}` });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { key: 'trades', value: false } });
    }
  }, [dispatch]);

  // Portfolio data hooks
  const fetchPortfolioData = useCallback(async () => {
    dispatch({ type: 'SET_LOADING', payload: { key: 'portfolio', value: true } });
    try {
      const [positions, riskMetrics] = await Promise.all([
        apiClient.getPortfolioPositions(),
        apiClient.getRiskMetrics()
      ]);
      dispatch({ type: 'SET_PORTFOLIO_POSITIONS', payload: positions });
      dispatch({ type: 'SET_RISK_METRICS', payload: riskMetrics });
      dispatch({ type: 'CLEAR_ERROR' });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: `Failed to fetch portfolio data: ${error}` });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { key: 'portfolio', value: false } });
    }
  }, [dispatch]);

  // Explanations data hooks
  const fetchExplanations = useCallback(async (limit: number = 50) => {
    dispatch({ type: 'SET_LOADING', payload: { key: 'explanations', value: true } });
    try {
      const data = await apiClient.getExplanations(limit);
      dispatch({ type: 'SET_EXPLANATIONS', payload: data });
      dispatch({ type: 'CLEAR_ERROR' });
    } catch (error) {
      dispatch({ type: 'SET_ERROR', payload: `Failed to fetch explanations: ${error}` });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { key: 'explanations', value: false } });
    }
  }, [dispatch]);

  // Symbol selection
  const setSelectedSymbol = useCallback((symbol: string | null) => {
    dispatch({ type: 'SET_SELECTED_SYMBOL', payload: symbol });
  }, [dispatch]);

  // Error handling
  const clearError = useCallback(() => {
    dispatch({ type: 'CLEAR_ERROR' });
  }, [dispatch]);

  return {
    fetchSentimentScores,
    fetchPredictions,
    fetchTrades,
    fetchPortfolioData,
    fetchExplanations,
    setSelectedSymbol,
    clearError,
  };
}

export default useApi;