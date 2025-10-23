import React, { useState, useEffect } from 'react';
import { apiClient } from '../services/api';
import { LoadingSpinner } from './LoadingSpinner';

interface SystemHealth {
  overall_status: string;
  timestamp: string;
  data_integration: any;
  services: any;
  repositories: any;
  metrics: any;
}

interface SystemMetrics {
  data_ingested: number;
  sentiment_processed: number;
  predictions_generated: number;
  trades_executed: number;
  errors: number;
  last_run: string | null;
  timestamp: string;
}

interface PipelineResult {
  status: string;
  execution_time_seconds: number;
  symbols_processed: string[];
  data_ingestion: any;
  predictions: any;
  trading_results: any;
  explanations: any;
  metrics: any;
}

export const SystemIntegration: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [pipelineResult, setPipelineResult] = useState<PipelineResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [symbols, setSymbols] = useState('AAPL,GOOGL,MSFT');
  const [includeTrading, setIncludeTrading] = useState(true);
  const [includeExplanations, setIncludeExplanations] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadSystemStatus();
  }, []);

  const loadSystemStatus = async () => {
    try {
      const [healthData, metricsData] = await Promise.all([
        apiClient.getSystemHealth(),
        apiClient.getSystemMetrics()
      ]);
      
      setHealth(healthData);
      setMetrics(metricsData);
      setError(null);
    } catch (err) {
      console.error('Error loading system status:', err);
      setError('Failed to load system status');
    }
  };

  const runPipeline = async () => {
    if (!symbols.trim()) {
      setError('Please enter at least one symbol');
      return;
    }

    setLoading(true);
    setError(null);
    setPipelineResult(null);

    try {
      const symbolList = symbols.split(',').map(s => s.trim().toUpperCase()).filter(s => s);
      
      const result = await apiClient.runCompletePipeline(
        symbolList,
        includeTrading,
        includeExplanations
      );

      setPipelineResult(result);
      
      // Refresh metrics after pipeline run
      const updatedMetrics = await apiClient.getSystemMetrics();
      setMetrics(updatedMetrics);
      
    } catch (err: any) {
      console.error('Error running pipeline:', err);
      setError(err.response?.data?.detail || 'Failed to run pipeline');
    } finally {
      setLoading(false);
    }
  };

  const runPipelineAsync = async () => {
    if (!symbols.trim()) {
      setError('Please enter at least one symbol');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const symbolList = symbols.split(',').map(s => s.trim().toUpperCase()).filter(s => s);
      
      const result = await apiClient.runPipelineAsync(
        symbolList,
        includeTrading,
        includeExplanations
      );

      setError(null);
      alert(`Pipeline submitted successfully for symbols: ${symbolList.join(', ')}`);
      
    } catch (err: any) {
      console.error('Error submitting pipeline:', err);
      setError(err.response?.data?.detail || 'Failed to submit pipeline');
    } finally {
      setLoading(false);
    }
  };

  const resetMetrics = async () => {
    try {
      await apiClient.resetSystemMetrics();
      await loadSystemStatus();
      setError(null);
    } catch (err: any) {
      console.error('Error resetting metrics:', err);
      setError('Failed to reset metrics');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'degraded': return 'text-yellow-600';
      case 'unhealthy': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatusBadge = (status: string) => {
    const baseClasses = 'px-2 py-1 rounded-full text-xs font-medium';
    switch (status) {
      case 'healthy': return `${baseClasses} bg-green-100 text-green-800`;
      case 'degraded': return `${baseClasses} bg-yellow-100 text-yellow-800`;
      case 'unhealthy': return `${baseClasses} bg-red-100 text-red-800`;
      default: return `${baseClasses} bg-gray-100 text-gray-800`;
    }
  };

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-gray-900">System Integration</h2>
        <button
          onClick={loadSystemStatus}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
        >
          Refresh Status
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* System Health */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">System Health</h3>
        {health ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="font-medium">Overall Status:</span>
              <span className={getStatusBadge(health.overall_status)}>
                {health.overall_status.toUpperCase()}
              </span>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gray-50 p-3 rounded">
                <h4 className="font-medium text-sm text-gray-700 mb-2">Data Integration</h4>
                <span className={getStatusColor(health.data_integration?.overall_status || 'unknown')}>
                  {health.data_integration?.overall_status || 'Unknown'}
                </span>
              </div>
              
              <div className="bg-gray-50 p-3 rounded">
                <h4 className="font-medium text-sm text-gray-700 mb-2">Services</h4>
                <div className="space-y-1">
                  {Object.entries(health.services || {}).map(([service, status]: [string, any]) => (
                    <div key={service} className="flex justify-between text-sm">
                      <span>{service}:</span>
                      <span className={getStatusColor(status.status)}>
                        {status.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="bg-gray-50 p-3 rounded">
                <h4 className="font-medium text-sm text-gray-700 mb-2">Repositories</h4>
                <div className="space-y-1">
                  {Object.entries(health.repositories || {}).map(([repo, status]: [string, any]) => (
                    <div key={repo} className="flex justify-between text-sm">
                      <span>{repo.replace('_repository', '')}:</span>
                      <span className={getStatusColor(status.status)}>
                        {status.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <LoadingSpinner />
        )}
      </div>

      {/* System Metrics */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">System Metrics</h3>
          <button
            onClick={resetMetrics}
            className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-700 transition-colors"
          >
            Reset Metrics
          </button>
        </div>
        
        {metrics ? (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{metrics.data_ingested}</div>
              <div className="text-sm text-gray-600">Data Ingested</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{metrics.sentiment_processed}</div>
              <div className="text-sm text-gray-600">Sentiment Processed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{metrics.predictions_generated}</div>
              <div className="text-sm text-gray-600">Predictions Generated</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">{metrics.trades_executed}</div>
              <div className="text-sm text-gray-600">Trades Executed</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-red-600">{metrics.errors}</div>
              <div className="text-sm text-gray-600">Errors</div>
            </div>
          </div>
        ) : (
          <LoadingSpinner />
        )}
      </div>

      {/* Pipeline Control */}
      <div className="bg-white rounded-lg shadow p-6">
        <h3 className="text-lg font-semibold mb-4">Run Complete Pipeline</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Stock Symbols (comma-separated)
            </label>
            <input
              type="text"
              value={symbols}
              onChange={(e) => setSymbols(e.target.value)}
              placeholder="AAPL,GOOGL,MSFT"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
          
          <div className="flex space-x-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={includeTrading}
                onChange={(e) => setIncludeTrading(e.target.checked)}
                className="mr-2"
              />
              Include Trading
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={includeExplanations}
                onChange={(e) => setIncludeExplanations(e.target.checked)}
                className="mr-2"
              />
              Include Explanations
            </label>
          </div>
          
          <div className="flex space-x-4">
            <button
              onClick={runPipeline}
              disabled={loading}
              className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Running...' : 'Run Pipeline (Sync)'}
            </button>
            <button
              onClick={runPipelineAsync}
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Run Pipeline (Async)
            </button>
          </div>
        </div>
      </div>

      {/* Pipeline Results */}
      {pipelineResult && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold mb-4">Pipeline Results</h3>
          
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="font-medium">Status:</span>
              <span className={getStatusBadge(pipelineResult.status)}>
                {pipelineResult.status.toUpperCase()}
              </span>
            </div>
            
            <div className="flex justify-between">
              <span className="font-medium">Execution Time:</span>
              <span>{pipelineResult.execution_time_seconds.toFixed(2)}s</span>
            </div>
            
            <div className="flex justify-between">
              <span className="font-medium">Symbols Processed:</span>
              <span>{pipelineResult.symbols_processed.join(', ')}</span>
            </div>
            
            {pipelineResult.data_ingestion && (
              <div>
                <h4 className="font-medium mb-2">Data Ingestion Summary:</h4>
                <div className="bg-gray-50 p-3 rounded text-sm">
                  <div>Social Media Sent: {pipelineResult.data_ingestion.streaming_results?.social_media_sent || 0}</div>
                  <div>News Sent: {pipelineResult.data_ingestion.streaming_results?.news_sent || 0}</div>
                  <div>Market Data Sent: {pipelineResult.data_ingestion.streaming_results?.market_data_sent || 0}</div>
                </div>
              </div>
            )}
            
            {pipelineResult.trading_results && Object.keys(pipelineResult.trading_results).length > 0 && (
              <div>
                <h4 className="font-medium mb-2">Trading Results:</h4>
                <div className="bg-gray-50 p-3 rounded text-sm space-y-1">
                  {Object.entries(pipelineResult.trading_results).map(([symbol, result]: [string, any]) => (
                    <div key={symbol} className="flex justify-between">
                      <span>{symbol}:</span>
                      <span>{result.action || result.status || 'No action'}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};