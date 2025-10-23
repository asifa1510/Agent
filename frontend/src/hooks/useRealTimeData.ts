import { useEffect, useRef } from 'react';
import { useApi } from './useApi';
import { useAppContext } from '../context/AppContext';
import { config } from '../config';

export function useRealTimeData() {
  const { state } = useAppContext();
  const { 
    fetchSentimentScores, 
    fetchPredictions, 
    fetchTrades, 
    fetchPortfolioData 
  } = useApi();
  
  const intervalsRef = useRef<{ [key: string]: NodeJS.Timeout }>({});

  // Start real-time data fetching
  const startRealTimeUpdates = () => {
    // Clear existing intervals
    stopRealTimeUpdates();

    // Sentiment data updates
    intervalsRef.current.sentiment = setInterval(() => {
      fetchSentimentScores(state.selectedSymbol || undefined);
    }, config.refreshIntervals.sentiment);

    // Predictions updates
    intervalsRef.current.predictions = setInterval(() => {
      fetchPredictions(state.selectedSymbol || undefined);
    }, config.refreshIntervals.predictions);

    // Trades updates
    intervalsRef.current.trades = setInterval(() => {
      fetchTrades();
    }, config.refreshIntervals.trades);

    // Portfolio updates
    intervalsRef.current.portfolio = setInterval(() => {
      fetchPortfolioData();
    }, config.refreshIntervals.portfolio);
  };

  // Stop real-time data fetching
  const stopRealTimeUpdates = () => {
    Object.values(intervalsRef.current).forEach(interval => {
      if (interval) clearInterval(interval);
    });
    intervalsRef.current = {};
  };

  // Initial data fetch and start real-time updates
  useEffect(() => {
    // Initial data fetch
    fetchSentimentScores();
    fetchPredictions();
    fetchTrades();
    fetchPortfolioData();

    // Start real-time updates
    startRealTimeUpdates();

    // Cleanup on unmount
    return () => {
      stopRealTimeUpdates();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update data when selected symbol changes
  useEffect(() => {
    if (state.selectedSymbol) {
      fetchSentimentScores(state.selectedSymbol);
      fetchPredictions(state.selectedSymbol);
    }
  }, [state.selectedSymbol, fetchSentimentScores, fetchPredictions]);

  return {
    startRealTimeUpdates,
    stopRealTimeUpdates,
  };
}

export default useRealTimeData;