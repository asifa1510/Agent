import React, { useState, useEffect, useCallback } from 'react';
import { Trade, TradeExplanation, SentimentScore, PricePrediction } from '../../types';
import { apiClient } from '../../services/api';
import LoadingSpinner from '../LoadingSpinner';
import ExplanationCard from './ExplanationCard';
import { formatTimestamp, formatCurrency } from '../../utils/formatters';

interface TradeReasoningPanelProps {
  trade: Trade;
  className?: string;
  autoLoad?: boolean;
}

export function TradeReasoningPanel({ 
  trade, 
  className = '',
  autoLoad = true 
}: TradeReasoningPanelProps) {
  const [explanation, setExplanation] = useState<TradeExplanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [contextData, setContextData] = useState<{
    sentiment?: SentimentScore[];
    predictions?: PricePrediction[];
  }>({});

  const loadContextualData = useCallback(async () => {
    try {
      // Load sentiment data around trade time
      const sentimentPromise = apiClient.getSentimentHistory(trade.symbol, 2);
      
      // Load predictions around trade time
      const predictionsPromise = apiClient.getPredictionHistory(trade.symbol, 1);

      const [sentiment, predictions] = await Promise.all([
        sentimentPromise.catch(() => []),
        predictionsPromise.catch(() => [])
      ]);

      setContextData({ sentiment, predictions });
    } catch (err) {
      console.error('Error loading contextual data:', err);
    }
  }, [trade.symbol]);

  const loadExplanation = useCallback(async () => {
    if (!trade.explanation_id) {
      setError('No explanation ID available for this trade');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const explanationData = await apiClient.getTradeExplanation(trade.explanation_id);
      setExplanation(explanationData);

      // Load contextual data for better insights
      await loadContextualData();
    } catch (err) {
      setError('Failed to load trade explanation');
      console.error('Error loading explanation:', err);
    } finally {
      setLoading(false);
    }
  }, [trade.explanation_id, loadContextualData]);

  useEffect(() => {
    if (autoLoad && trade.explanation_id) {
      loadExplanation();
    }
  }, [autoLoad, trade.explanation_id, loadExplanation]);

  const getSignalStrengthColor = (strength: number) => {
    if (strength >= 0.8) return 'text-green-600';
    if (strength >= 0.6) return 'text-yellow-600';
    if (strength >= 0.4) return 'text-orange-600';
    return 'text-red-600';
  };

  const getSignalStrengthLabel = (strength: number) => {
    if (strength >= 0.8) return 'Very Strong';
    if (strength >= 0.6) return 'Strong';
    if (strength >= 0.4) return 'Moderate';
    return 'Weak';
  };

  return (
    <div className={`bg-white rounded-lg shadow-sm border ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-medium text-gray-900">Trade Reasoning</h3>
            <p className="text-sm text-gray-500 mt-1">
              {trade.symbol} • {trade.action.toUpperCase()} • {formatTimestamp(trade.timestamp, true)}
            </p>
          </div>
          
          {!loading && !explanation && trade.explanation_id && (
            <button
              onClick={loadExplanation}
              className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
            >
              Load Explanation
            </button>
          )}
        </div>
      </div>

      {/* Trade Summary */}
      <div className="p-4 bg-gray-50 border-b border-gray-200">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <span className="text-xs text-gray-500">Action</span>
            <div className={`text-sm font-medium ${
              trade.action === 'buy' ? 'text-green-600' : 'text-red-600'
            }`}>
              {trade.action.toUpperCase()}
            </div>
          </div>
          <div>
            <span className="text-xs text-gray-500">Quantity</span>
            <div className="text-sm font-medium text-gray-900">
              {trade.quantity.toLocaleString()}
            </div>
          </div>
          <div>
            <span className="text-xs text-gray-500">Price</span>
            <div className="text-sm font-medium text-gray-900">
              {formatCurrency(trade.price)}
            </div>
          </div>
          <div>
            <span className="text-xs text-gray-500">Signal Strength</span>
            <div className={`text-sm font-medium ${getSignalStrengthColor(trade.signal_strength)}`}>
              {getSignalStrengthLabel(trade.signal_strength)} ({(trade.signal_strength * 100).toFixed(0)}%)
            </div>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-4">
        {loading && (
          <div className="flex items-center justify-center py-8">
            <LoadingSpinner size="md" />
            <span className="ml-2 text-gray-500">Loading explanation...</span>
          </div>
        )}

        {error && (
          <div className="text-center py-8">
            <div className="text-red-600 mb-2">{error}</div>
            {trade.explanation_id && (
              <button
                onClick={loadExplanation}
                className="text-sm text-blue-600 hover:text-blue-800"
              >
                Try again
              </button>
            )}
          </div>
        )}

        {explanation && (
          <div className="space-y-4">
            <ExplanationCard 
              explanation={explanation}
              showSupportingData={true}
              compact={false}
            />

            {/* Contextual Insights */}
            {(contextData.sentiment?.length || contextData.predictions?.length) && (
              <div className="border-t border-gray-200 pt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-3">
                  Market Context at Trade Time
                </h4>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {/* Sentiment Context */}
                  {contextData.sentiment && contextData.sentiment.length > 0 && (
                    <div className="bg-blue-50 rounded-lg p-3">
                      <h5 className="text-xs font-medium text-blue-800 mb-2">
                        Sentiment Analysis
                      </h5>
                      {contextData.sentiment.slice(0, 2).map((sentiment, index) => (
                        <div key={index} className="flex justify-between items-center text-xs">
                          <span className="text-blue-700">
                            {sentiment.source} ({sentiment.volume} posts)
                          </span>
                          <span className={`font-medium ${
                            sentiment.score > 0 ? 'text-green-600' : 
                            sentiment.score < 0 ? 'text-red-600' : 'text-gray-600'
                          }`}>
                            {sentiment.score > 0 ? '+' : ''}{sentiment.score.toFixed(2)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}

                  {/* Prediction Context */}
                  {contextData.predictions && contextData.predictions.length > 0 && (
                    <div className="bg-green-50 rounded-lg p-3">
                      <h5 className="text-xs font-medium text-green-800 mb-2">
                        Price Predictions
                      </h5>
                      {contextData.predictions.slice(0, 2).map((prediction, index) => (
                        <div key={index} className="flex justify-between items-center text-xs">
                          <span className="text-green-700">
                            {prediction.horizon} ({prediction.model})
                          </span>
                          <span className="font-medium text-green-600">
                            {formatCurrency(prediction.predicted_price)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {!loading && !error && !explanation && !trade.explanation_id && (
          <div className="text-center py-8 text-gray-500">
            <div className="text-sm">No explanation available for this trade</div>
            <div className="text-xs mt-1">
              Explanations are generated for trades with sufficient signal strength
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default TradeReasoningPanel;