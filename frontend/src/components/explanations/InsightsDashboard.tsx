import React, { useState, useEffect } from 'react';
import { useAppContext } from '../../context/AppContext';
import { useApi } from '../../hooks/useApi';
import { TradeExplanation, Trade } from '../../types';
import LoadingSpinner from '../LoadingSpinner';
import ExplanationCard from './ExplanationCard';
import ConfidenceScore from './ConfidenceScore';
import { formatTimestamp } from '../../utils/formatters';

interface InsightsDashboardProps {
  className?: string;
  limit?: number;
}

export function InsightsDashboard({ className = '', limit = 10 }: InsightsDashboardProps) {
  const { state } = useAppContext();
  const { fetchExplanations } = useApi();
  const [selectedExplanation, setSelectedExplanation] = useState<TradeExplanation | null>(null);
  const [filterBy, setFilterBy] = useState<'all' | 'high' | 'medium' | 'low'>('all');
  const [sortBy, setSortBy] = useState<'timestamp' | 'confidence'>('timestamp');

  useEffect(() => {
    fetchExplanations(limit);
  }, [fetchExplanations, limit]);

  // Filter explanations by confidence level
  const filteredExplanations = state.explanations.filter(explanation => {
    switch (filterBy) {
      case 'high':
        return explanation.confidence >= 0.8;
      case 'medium':
        return explanation.confidence >= 0.6 && explanation.confidence < 0.8;
      case 'low':
        return explanation.confidence < 0.6;
      default:
        return true;
    }
  });

  // Sort explanations
  const sortedExplanations = [...filteredExplanations].sort((a, b) => {
    if (sortBy === 'confidence') {
      return b.confidence - a.confidence;
    }
    return b.timestamp - a.timestamp;
  });

  // Get trade for explanation
  const getTradeForExplanation = (explanation: TradeExplanation): Trade | undefined => {
    return state.trades.find(trade => trade.id === explanation.trade_id);
  };

  // Calculate confidence distribution
  const confidenceDistribution = {
    high: state.explanations.filter(e => e.confidence >= 0.8).length,
    medium: state.explanations.filter(e => e.confidence >= 0.6 && e.confidence < 0.8).length,
    low: state.explanations.filter(e => e.confidence < 0.6).length
  };

  const averageConfidence = state.explanations.length > 0 
    ? state.explanations.reduce((sum, e) => sum + e.confidence, 0) / state.explanations.length
    : 0;

  return (
    <div className={`bg-white rounded-lg shadow ${className}`}>
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Trading Insights</h2>
            <p className="text-sm text-gray-500 mt-1">
              AI-generated explanations and reasoning for trading decisions
            </p>
          </div>
          
          {/* Summary Stats */}
          <div className="mt-4 sm:mt-0 flex items-center space-x-4">
            <div className="text-center">
              <div className="text-lg font-semibold text-gray-900">
                {state.explanations.length}
              </div>
              <div className="text-xs text-gray-500">Total</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-semibold text-green-600">
                {(averageConfidence * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-gray-500">Avg Confidence</div>
            </div>
          </div>
        </div>

        {/* Confidence Distribution */}
        <div className="mt-4 grid grid-cols-3 gap-4">
          <div className="bg-green-50 rounded-lg p-3 text-center">
            <div className="text-lg font-semibold text-green-600">
              {confidenceDistribution.high}
            </div>
            <div className="text-xs text-green-700">High Confidence (≥80%)</div>
          </div>
          <div className="bg-yellow-50 rounded-lg p-3 text-center">
            <div className="text-lg font-semibold text-yellow-600">
              {confidenceDistribution.medium}
            </div>
            <div className="text-xs text-yellow-700">Medium Confidence (60-79%)</div>
          </div>
          <div className="bg-red-50 rounded-lg p-3 text-center">
            <div className="text-lg font-semibold text-red-600">
              {confidenceDistribution.low}
            </div>
            <div className="text-xs text-red-700">Low Confidence (&lt;60%)</div>
          </div>
        </div>

        {/* Filters */}
        <div className="mt-4 flex flex-col sm:flex-row gap-4">
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Filter by Confidence
            </label>
            <select
              value={filterBy}
              onChange={(e) => setFilterBy(e.target.value as typeof filterBy)}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Levels</option>
              <option value="high">High (≥80%)</option>
              <option value="medium">Medium (60-79%)</option>
              <option value="low">Low (&lt;60%)</option>
            </select>
          </div>
          
          <div>
            <label className="block text-xs font-medium text-gray-700 mb-1">
              Sort by
            </label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="timestamp">Most Recent</option>
              <option value="confidence">Highest Confidence</option>
            </select>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="p-6">
        {state.loading.explanations ? (
          <div className="flex items-center justify-center py-8">
            <LoadingSpinner size="lg" />
          </div>
        ) : sortedExplanations.length > 0 ? (
          <div className="space-y-4">
            {sortedExplanations.map((explanation) => {
              const trade = getTradeForExplanation(explanation);
              return (
                <div key={explanation.id} className="border border-gray-200 rounded-lg overflow-hidden">
                  {/* Explanation Header */}
                  <div className="p-4 bg-gray-50 border-b border-gray-200">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        {trade && (
                          <>
                            <span className="font-medium text-gray-900">
                              {trade.symbol}
                            </span>
                            <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                              trade.action === 'buy' 
                                ? 'bg-green-100 text-green-800' 
                                : 'bg-red-100 text-red-800'
                            }`}>
                              {trade.action.toUpperCase()}
                            </span>
                            <span className="text-sm text-gray-500">
                              {trade.quantity} shares @ ${trade.price.toFixed(2)}
                            </span>
                          </>
                        )}
                      </div>
                      
                      <div className="flex items-center space-x-3">
                        <ConfidenceScore 
                          confidence={explanation.confidence}
                          size="sm"
                          showLabel={false}
                        />
                        <span className="text-xs text-gray-500">
                          {formatTimestamp(explanation.timestamp, true)}
                        </span>
                        <button
                          onClick={() => setSelectedExplanation(
                            selectedExplanation?.id === explanation.id ? null : explanation
                          )}
                          className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                        >
                          {selectedExplanation?.id === explanation.id ? 'Hide' : 'Details'}
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Explanation Preview */}
                  <div className="p-4">
                    <p className="text-sm text-gray-700 line-clamp-2">
                      {explanation.explanation}
                    </p>
                  </div>

                  {/* Expanded Details */}
                  {selectedExplanation?.id === explanation.id && (
                    <div className="border-t border-gray-200">
                      <ExplanationCard 
                        explanation={explanation}
                        showSupportingData={true}
                        compact={false}
                        className="border-0 shadow-none"
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="text-gray-500 mb-2">
              {filterBy === 'all' 
                ? 'No explanations available' 
                : `No explanations found for ${filterBy} confidence level`
              }
            </div>
            <div className="text-sm text-gray-400">
              Explanations will appear here as trades are executed and analyzed
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default InsightsDashboard;