import React, { useEffect, useState, useMemo } from 'react';
import { useAppContext } from '../context/AppContext';
import { useApi } from '../hooks/useApi';
import LoadingSpinner from './LoadingSpinner';
import { formatCurrency, formatTimestamp, formatRelativeTime } from '../utils/formatters';
import { Trade, TradeExplanation } from '../types';
import { apiClient } from '../services/api';
import { ExplanationCard, ConfidenceScore } from './explanations';

interface TradeHistoryProps {
  className?: string;
  limit?: number;
}

interface TradeWithExplanation extends Trade {
  explanation?: TradeExplanation;
  loadingExplanation?: boolean;
}

export function TradeHistory({ className = '', limit = 100 }: TradeHistoryProps) {
  const { state } = useAppContext();
  const { fetchTrades, fetchExplanations } = useApi();
  
  // Local state for filtering and search
  const [searchTerm, setSearchTerm] = useState('');
  const [actionFilter, setActionFilter] = useState<'all' | 'buy' | 'sell'>('all');
  const [sortBy, setSortBy] = useState<'timestamp' | 'symbol' | 'price' | 'quantity'>('timestamp');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [expandedTrade, setExpandedTrade] = useState<string | null>(null);
  const [tradesWithExplanations, setTradesWithExplanations] = useState<TradeWithExplanation[]>([]);

  useEffect(() => {
    fetchTrades(limit);
    fetchExplanations(50);
  }, [fetchTrades, fetchExplanations, limit]);

  // Merge trades with explanations
  useEffect(() => {
    const merged = state.trades.map(trade => {
      const explanation = state.explanations.find(exp => exp.trade_id === trade.id);
      return { ...trade, explanation };
    });
    setTradesWithExplanations(merged);
  }, [state.trades, state.explanations]);

  // Load explanation for a specific trade
  const loadExplanation = async (tradeId: string) => {
    if (!tradeId) return;
    
    setTradesWithExplanations(prev => 
      prev.map(trade => 
        trade.id === tradeId 
          ? { ...trade, loadingExplanation: true }
          : trade
      )
    );

    try {
      const explanation = await apiClient.getTradeExplanation(tradeId);
      setTradesWithExplanations(prev => 
        prev.map(trade => 
          trade.id === tradeId 
            ? { ...trade, explanation, loadingExplanation: false }
            : trade
        )
      );
    } catch (error) {
      console.error('Failed to load explanation:', error);
      setTradesWithExplanations(prev => 
        prev.map(trade => 
          trade.id === tradeId 
            ? { ...trade, loadingExplanation: false }
            : trade
        )
      );
    }
  };

  // Filter and sort trades
  const filteredAndSortedTrades = useMemo(() => {
    let filtered = tradesWithExplanations.filter(trade => {
      const matchesSearch = trade.symbol.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesAction = actionFilter === 'all' || trade.action === actionFilter;
      return matchesSearch && matchesAction;
    });

    return filtered.sort((a, b) => {
      let aValue: number | string;
      let bValue: number | string;

      switch (sortBy) {
        case 'symbol':
          aValue = a.symbol;
          bValue = b.symbol;
          break;
        case 'price':
          aValue = a.price;
          bValue = b.price;
          break;
        case 'quantity':
          aValue = a.quantity;
          bValue = b.quantity;
          break;
        case 'timestamp':
        default:
          aValue = a.timestamp;
          bValue = b.timestamp;
          break;
      }

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortOrder === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
      }

      const numA = typeof aValue === 'number' ? aValue : 0;
      const numB = typeof bValue === 'number' ? bValue : 0;
      
      return sortOrder === 'asc' ? numA - numB : numB - numA;
    });
  }, [tradesWithExplanations, searchTerm, actionFilter, sortBy, sortOrder]);

  const handleSort = (field: typeof sortBy) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  const getSortIcon = (field: typeof sortBy) => {
    if (sortBy !== field) {
      return <span className="text-gray-400">↕</span>;
    }
    return sortOrder === 'asc' ? <span className="text-blue-600">↑</span> : <span className="text-blue-600">↓</span>;
  };

  const toggleTradeExpansion = (tradeId: string) => {
    if (expandedTrade === tradeId) {
      setExpandedTrade(null);
    } else {
      setExpandedTrade(tradeId);
      const trade = tradesWithExplanations.find(t => t.id === tradeId);
      if (trade && !trade.explanation && !trade.loadingExplanation) {
        loadExplanation(tradeId);
      }
    }
  };

  if (state.loading.trades) {
    return (
      <div className={`bg-white rounded-lg shadow p-6 ${className}`}>
        <div className="flex items-center justify-center py-8">
          <LoadingSpinner size="lg" />
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-white rounded-lg shadow ${className}`}>
      {/* Header with filters */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900 mb-4 sm:mb-0">Trade History</h2>
          <div className="text-sm text-gray-500">
            {filteredAndSortedTrades.length} of {state.trades.length} trades
          </div>
        </div>

        {/* Search and Filter Controls */}
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="flex-1">
            <input
              type="text"
              placeholder="Search by symbol..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Action Filter */}
          <div>
            <select
              value={actionFilter}
              onChange={(e) => setActionFilter(e.target.value as typeof actionFilter)}
              className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="all">All Actions</option>
              <option value="buy">Buy Only</option>
              <option value="sell">Sell Only</option>
            </select>
          </div>
        </div>
      </div>

      {/* Trades Table */}
      <div className="overflow-x-auto">
        {filteredAndSortedTrades.length > 0 ? (
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('timestamp')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Time</span>
                    {getSortIcon('timestamp')}
                  </div>
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('symbol')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Symbol</span>
                    {getSortIcon('symbol')}
                  </div>
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Action
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('quantity')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Quantity</span>
                    {getSortIcon('quantity')}
                  </div>
                </th>
                <th 
                  className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort('price')}
                >
                  <div className="flex items-center space-x-1">
                    <span>Price</span>
                    {getSortIcon('price')}
                  </div>
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Total Value
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Signal Strength
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Explanation
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {filteredAndSortedTrades.map((trade) => (
                <React.Fragment key={trade.id}>
                  <tr className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{formatTimestamp(trade.timestamp, true)}</div>
                      <div className="text-xs text-gray-500">{formatRelativeTime(trade.timestamp)}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">{trade.symbol}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                        trade.action === 'buy' 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {trade.action.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{trade.quantity.toLocaleString()}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm text-gray-900">{formatCurrency(trade.price)}</div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="text-sm font-medium text-gray-900">
                        {formatCurrency(trade.quantity * trade.price)}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <ConfidenceScore 
                        confidence={trade.signal_strength}
                        size="sm"
                        showLabel={false}
                        showPercentage={true}
                      />
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <button
                        onClick={() => toggleTradeExpansion(trade.id)}
                        className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                      >
                        {expandedTrade === trade.id ? 'Hide' : 'Show'}
                      </button>
                    </td>
                  </tr>
                  
                  {/* Expanded explanation row */}
                  {expandedTrade === trade.id && (
                    <tr>
                      <td colSpan={8} className="px-6 py-4 bg-gray-50">
                        {trade.loadingExplanation ? (
                          <div className="flex items-center justify-center space-x-2 py-4">
                            <LoadingSpinner size="sm" />
                            <span className="text-sm text-gray-500">Loading explanation...</span>
                          </div>
                        ) : trade.explanation ? (
                          <ExplanationCard 
                            explanation={trade.explanation}
                            showSupportingData={true}
                            compact={true}
                            className="border-0 shadow-none bg-transparent"
                          />
                        ) : (
                          <div className="text-center py-4">
                            <div className="text-sm text-gray-500 italic">
                              No explanation available for this trade
                            </div>
                            <div className="text-xs text-gray-400 mt-1">
                              Explanations are generated for trades with sufficient signal strength
                            </div>
                          </div>
                        )}
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="text-center text-gray-500 py-8">
            <div className="text-lg font-medium mb-2">
              {searchTerm || actionFilter !== 'all' ? 'No matching trades found' : 'No trades found'}
            </div>
            <div className="text-sm">
              {searchTerm || actionFilter !== 'all' 
                ? 'Try adjusting your search or filter criteria'
                : 'Trade history will appear here once trades are executed'
              }
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default TradeHistory;