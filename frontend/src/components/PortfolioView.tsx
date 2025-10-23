import React, { useEffect, useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { useApi } from '../hooks/useApi';
import LoadingSpinner from './LoadingSpinner';
import { formatCurrency, formatPercentage, getPnLColorClass, formatTimestamp } from '../utils/formatters';

interface PortfolioViewProps {
  className?: string;
}

export function PortfolioView({ className = '' }: PortfolioViewProps) {
  const { state } = useAppContext();
  const { fetchPortfolioData } = useApi();
  const [sortBy, setSortBy] = useState<'symbol' | 'value' | 'pnl' | 'allocation'>('value');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    fetchPortfolioData();
  }, [fetchPortfolioData]);

  const handleSort = (field: typeof sortBy) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
  };

  const sortedPositions = [...state.portfolioPositions].sort((a, b) => {
    let aValue: number | string;
    let bValue: number | string;

    switch (sortBy) {
      case 'symbol':
        aValue = a.symbol;
        bValue = b.symbol;
        break;
      case 'value':
        aValue = a.quantity * a.current_price;
        bValue = b.quantity * b.current_price;
        break;
      case 'pnl':
        aValue = a.unrealized_pnl;
        bValue = b.unrealized_pnl;
        break;
      case 'allocation':
        aValue = a.allocation_percent;
        bValue = b.allocation_percent;
        break;
      default:
        aValue = 0;
        bValue = 0;
    }

    if (typeof aValue === 'string' && typeof bValue === 'string') {
      return sortOrder === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
    }

    const numA = typeof aValue === 'number' ? aValue : 0;
    const numB = typeof bValue === 'number' ? bValue : 0;
    
    return sortOrder === 'asc' ? numA - numB : numB - numA;
  });

  const getSortIcon = (field: typeof sortBy) => {
    if (sortBy !== field) {
      return <span className="text-gray-400">↕</span>;
    }
    return sortOrder === 'asc' ? <span className="text-blue-600">↑</span> : <span className="text-blue-600">↓</span>;
  };

  if (state.loading.portfolio) {
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
      {/* Portfolio Summary Header */}
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Portfolio Overview</h2>
        
        {state.riskMetrics && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">
                {formatCurrency(state.riskMetrics.total_value)}
              </div>
              <div className="text-sm text-gray-500">Total Value</div>
            </div>
            
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className={`text-2xl font-bold ${getPnLColorClass(state.riskMetrics.total_pnl)}`}>
                {formatCurrency(state.riskMetrics.total_pnl)}
              </div>
              <div className="text-sm text-gray-500">Total P&L</div>
            </div>
            
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">
                {state.riskMetrics.sharpe_ratio?.toFixed(2) || 'N/A'}
              </div>
              <div className="text-sm text-gray-500">Sharpe Ratio</div>
            </div>
            
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <div className="text-2xl font-bold text-red-600">
                {formatPercentage(state.riskMetrics.max_drawdown)}
              </div>
              <div className="text-sm text-gray-500">Max Drawdown</div>
            </div>
          </div>
        )}
      </div>

      {/* Positions Table */}
      <div className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">Current Positions</h3>
          <div className="text-sm text-gray-500">
            {state.portfolioPositions.length} position{state.portfolioPositions.length !== 1 ? 's' : ''}
          </div>
        </div>

        {state.portfolioPositions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
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
                    Quantity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Avg Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Current Price
                  </th>
                  <th 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort('value')}
                  >
                    <div className="flex items-center space-x-1">
                      <span>Market Value</span>
                      {getSortIcon('value')}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort('pnl')}
                  >
                    <div className="flex items-center space-x-1">
                      <span>Unrealized P&L</span>
                      {getSortIcon('pnl')}
                    </div>
                  </th>
                  <th 
                    className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={() => handleSort('allocation')}
                  >
                    <div className="flex items-center space-x-1">
                      <span>Allocation</span>
                      {getSortIcon('allocation')}
                    </div>
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {sortedPositions.map((position) => {
                  const marketValue = position.quantity * position.current_price;
                  const pnlPercent = ((position.current_price - position.avg_price) / position.avg_price) * 100;
                  
                  return (
                    <tr key={position.symbol} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">{position.symbol}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{position.quantity.toLocaleString()}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{formatCurrency(position.avg_price)}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{formatCurrency(position.current_price)}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">{formatCurrency(marketValue)}</div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className={`text-sm font-medium ${getPnLColorClass(position.unrealized_pnl)}`}>
                          {formatCurrency(position.unrealized_pnl)}
                        </div>
                        <div className={`text-xs ${getPnLColorClass(pnlPercent)}`}>
                          ({pnlPercent > 0 ? '+' : ''}{pnlPercent.toFixed(2)}%)
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">{formatPercentage(position.allocation_percent / 100)}</div>
                        <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
                          <div 
                            className="bg-blue-600 h-2 rounded-full" 
                            style={{ width: `${Math.min(position.allocation_percent, 100)}%` }}
                          ></div>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center text-gray-500 py-8">
            <div className="text-lg font-medium mb-2">No positions found</div>
            <div className="text-sm">Your portfolio is currently empty</div>
          </div>
        )}
      </div>

      {/* Risk Metrics Footer */}
      {state.riskMetrics && (
        <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Volatility:</span>
              <span className="ml-2 font-medium">{formatPercentage(state.riskMetrics.volatility)}</span>
            </div>
            <div>
              <span className="text-gray-500">VaR (95%):</span>
              <span className="ml-2 font-medium text-red-600">{formatCurrency(state.riskMetrics.var_95)}</span>
            </div>
            <div>
              <span className="text-gray-500">Positions:</span>
              <span className="ml-2 font-medium">{state.riskMetrics.positions_count}</span>
            </div>
            <div>
              <span className="text-gray-500">Last Updated:</span>
              <span className="ml-2 font-medium">{formatTimestamp(state.riskMetrics.timestamp)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default PortfolioView;