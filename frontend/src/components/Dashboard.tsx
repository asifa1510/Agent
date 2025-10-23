import { useAppContext } from '../context/AppContext';
import { useRealTimeData } from '../hooks/useRealTimeData';
import SymbolSelector from './SymbolSelector';
import LoadingSpinner from './LoadingSpinner';
import { SentimentChart, PredictionPanel, InteractiveChart } from './charts';
import { InsightsDashboard } from './explanations';

export function Dashboard() {
  const { state } = useAppContext();
  
  // Initialize real-time data updates
  useRealTimeData();

  return (
    <div className="space-y-6">
      {/* Symbol Selection */}
      <SymbolSelector />

      {/* Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        
        {/* Sentiment Overview Card */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Sentiment Overview</h3>
            {state.loading.sentiment && <LoadingSpinner size="sm" />}
          </div>
          
          {state.sentimentScores.length > 0 ? (
            <div className="space-y-3">
              {state.sentimentScores.slice(0, 3).map((sentiment, index) => (
                <div key={index} className="flex justify-between items-center">
                  <div>
                    <span className="font-medium text-gray-900">{sentiment.symbol}</span>
                    <span className="text-sm text-gray-500 ml-2">
                      ({sentiment.source})
                    </span>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-medium ${
                      sentiment.score > 0 ? 'text-green-600' : 
                      sentiment.score < 0 ? 'text-red-600' : 'text-gray-600'
                    }`}>
                      {sentiment.score > 0 ? '+' : ''}{sentiment.score.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500">
                      {sentiment.volume} posts
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              No sentiment data available
            </div>
          )}
        </div>

        {/* Predictions Overview Card */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Price Predictions</h3>
            {state.loading.predictions && <LoadingSpinner size="sm" />}
          </div>
          
          {state.predictions.length > 0 ? (
            <div className="space-y-3">
              {state.predictions.slice(0, 3).map((prediction, index) => (
                <div key={index} className="flex justify-between items-center">
                  <div>
                    <span className="font-medium text-gray-900">{prediction.symbol}</span>
                    <span className="text-sm text-gray-500 ml-2">
                      ({prediction.horizon})
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium text-gray-900">
                      ${prediction.predicted_price.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500">
                      {prediction.model}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              No predictions available
            </div>
          )}
        </div>

        {/* Recent Trades Card */}
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Recent Trades</h3>
            {state.loading.trades && <LoadingSpinner size="sm" />}
          </div>
          
          {state.trades.length > 0 ? (
            <div className="space-y-3">
              {state.trades.slice(0, 3).map((trade, index) => (
                <div key={index} className="flex justify-between items-center">
                  <div>
                    <span className="font-medium text-gray-900">{trade.symbol}</span>
                    <span className={`text-sm ml-2 ${
                      trade.action === 'buy' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {trade.action.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium text-gray-900">
                      {trade.quantity} @ ${trade.price.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500">
                      {new Date(trade.timestamp * 1000).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              No trades available
            </div>
          )}
        </div>

        {/* Portfolio Summary Card */}
        <div className="bg-white rounded-lg shadow p-6 lg:col-span-2 xl:col-span-3">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">Portfolio Summary</h3>
            {state.loading.portfolio && <LoadingSpinner size="sm" />}
          </div>
          
          {state.riskMetrics ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">
                  ${state.riskMetrics.total_value.toLocaleString()}
                </div>
                <div className="text-sm text-gray-500">Total Value</div>
              </div>
              <div className="text-center">
                <div className={`text-2xl font-bold ${
                  state.riskMetrics.total_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  ${state.riskMetrics.total_pnl.toLocaleString()}
                </div>
                <div className="text-sm text-gray-500">Total P&L</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">
                  {state.riskMetrics.sharpe_ratio?.toFixed(2) || 'N/A'}
                </div>
                <div className="text-sm text-gray-500">Sharpe Ratio</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">
                  {state.portfolioPositions.length}
                </div>
                <div className="text-sm text-gray-500">Positions</div>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              No portfolio data available
            </div>
          )}
        </div>
      </div>

      {/* Real-time Data Visualization */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        
        {/* Sentiment Chart */}
        <div className="bg-white rounded-lg shadow p-6">
          <SentimentChart 
            sentimentData={state.sentimentScores}
            symbol={state.selectedSymbol}
            height={300}
            showVolume={true}
          />
        </div>

        {/* Prediction Panel */}
        <div>
          <PredictionPanel 
            predictions={state.predictions}
            symbol={state.selectedSymbol}
            height={300}
          />
        </div>
      </div>

      {/* Interactive Combined Chart */}
      {state.selectedSymbol && (
        <InteractiveChart
          sentimentData={state.sentimentScores}
          predictions={state.predictions}
          trades={state.trades}
          symbol={state.selectedSymbol}
          height={400}
        />
      )}

      {/* Trading Insights Dashboard */}
      <InsightsDashboard limit={20} />
    </div>
  );
}

export default Dashboard;