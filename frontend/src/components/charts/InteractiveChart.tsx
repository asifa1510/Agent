import React, { useState, useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import { SentimentScore, PricePrediction, Trade } from '../../types';
import { formatTimestamp, formatCurrency } from '../../utils/formatters';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  Filler
);

interface InteractiveChartProps {
  sentimentData: SentimentScore[];
  predictions: PricePrediction[];
  trades: Trade[];
  symbol: string;
  height?: number;
}

type ChartView = 'sentiment' | 'predictions' | 'combined';
type TimeRange = '1h' | '6h' | '24h' | '7d' | '30d';

export function InteractiveChart({ 
  sentimentData, 
  predictions, 
  trades, 
  symbol,
  height = 500 
}: InteractiveChartProps) {
  const [activeView, setActiveView] = useState<ChartView>('combined');
  const [timeRange, setTimeRange] = useState<TimeRange>('24h');
  const [showTrades, setShowTrades] = useState(true);

  // Filter data by time range
  const getTimeRangeMs = (range: TimeRange): number => {
    const now = Date.now();
    switch (range) {
      case '1h': return now - (1 * 60 * 60 * 1000);
      case '6h': return now - (6 * 60 * 60 * 1000);
      case '24h': return now - (24 * 60 * 60 * 1000);
      case '7d': return now - (7 * 24 * 60 * 60 * 1000);
      case '30d': return now - (30 * 24 * 60 * 60 * 1000);
      default: return now - (24 * 60 * 60 * 1000);
    }
  };

  const filteredData = useMemo(() => {
    const cutoffTime = getTimeRangeMs(timeRange) / 1000; // Convert to seconds
    
    return {
      sentiment: sentimentData.filter(item => 
        item.symbol === symbol && item.timestamp >= cutoffTime
      ),
      predictions: predictions.filter(item => 
        item.symbol === symbol && item.timestamp >= cutoffTime
      ),
      trades: trades.filter(item => 
        item.symbol === symbol && item.timestamp >= cutoffTime
      ),
    };
  }, [sentimentData, predictions, trades, symbol, timeRange]);

  const chartData = useMemo(() => {
    const datasets = [];

    // Sentiment data
    if (activeView === 'sentiment' || activeView === 'combined') {
      const twitterSentiment = filteredData.sentiment
        .filter(item => item.source === 'twitter')
        .map(item => ({ x: item.timestamp * 1000, y: item.score }))
        .sort((a, b) => a.x - b.x);

      const redditSentiment = filteredData.sentiment
        .filter(item => item.source === 'reddit')
        .map(item => ({ x: item.timestamp * 1000, y: item.score }))
        .sort((a, b) => a.x - b.x);

      if (twitterSentiment.length > 0) {
        datasets.push({
          label: 'Twitter Sentiment',
          data: twitterSentiment,
          borderColor: 'rgb(29, 161, 242)',
          backgroundColor: 'rgba(29, 161, 242, 0.1)',
          tension: 0.1,
          pointRadius: 2,
          pointHoverRadius: 4,
          yAxisID: 'sentiment',
        });
      }

      if (redditSentiment.length > 0) {
        datasets.push({
          label: 'Reddit Sentiment',
          data: redditSentiment,
          borderColor: 'rgb(255, 69, 0)',
          backgroundColor: 'rgba(255, 69, 0, 0.1)',
          tension: 0.1,
          pointRadius: 2,
          pointHoverRadius: 4,
          yAxisID: 'sentiment',
        });
      }
    }

    // Prediction data
    if (activeView === 'predictions' || activeView === 'combined') {
      const lstmPredictions = filteredData.predictions
        .filter(item => item.model === 'lstm')
        .map(item => ({ x: item.timestamp * 1000, y: item.predicted_price }))
        .sort((a, b) => a.x - b.x);

      const xgboostPredictions = filteredData.predictions
        .filter(item => item.model === 'xgboost')
        .map(item => ({ x: item.timestamp * 1000, y: item.predicted_price }))
        .sort((a, b) => a.x - b.x);

      if (lstmPredictions.length > 0) {
        datasets.push({
          label: 'LSTM Predictions',
          data: lstmPredictions,
          borderColor: 'rgb(34, 197, 94)',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          tension: 0.1,
          pointRadius: 3,
          pointHoverRadius: 5,
          yAxisID: activeView === 'combined' ? 'price' : 'y',
        });
      }

      if (xgboostPredictions.length > 0) {
        datasets.push({
          label: 'XGBoost Predictions',
          data: xgboostPredictions,
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          tension: 0.1,
          pointRadius: 3,
          pointHoverRadius: 5,
          yAxisID: activeView === 'combined' ? 'price' : 'y',
        });
      }
    }

    // Trade markers
    if (showTrades && filteredData.trades.length > 0) {
      const buyTrades = filteredData.trades
        .filter(trade => trade.action === 'buy')
        .map(trade => ({ x: trade.timestamp * 1000, y: trade.price }));

      const sellTrades = filteredData.trades
        .filter(trade => trade.action === 'sell')
        .map(trade => ({ x: trade.timestamp * 1000, y: trade.price }));

      if (buyTrades.length > 0) {
        datasets.push({
          label: 'Buy Trades',
          data: buyTrades,
          borderColor: 'rgb(34, 197, 94)',
          backgroundColor: 'rgb(34, 197, 94)',
          pointStyle: 'triangle',
          pointRadius: 6,
          pointHoverRadius: 8,
          showLine: false,
          yAxisID: activeView === 'combined' ? 'price' : 'y',
        });
      }

      if (sellTrades.length > 0) {
        datasets.push({
          label: 'Sell Trades',
          data: sellTrades,
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgb(239, 68, 68)',
          pointStyle: 'rectRot',
          pointRadius: 6,
          pointHoverRadius: 8,
          showLine: false,
          yAxisID: activeView === 'combined' ? 'price' : 'y',
        });
      }
    }

    return { datasets };
  }, [filteredData, activeView, showTrades]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: `${symbol} - ${activeView.charAt(0).toUpperCase() + activeView.slice(1)} Analysis`,
        font: {
          size: 16,
          weight: 'bold' as const,
        },
      },
      tooltip: {
        callbacks: {
          title: (context: any) => {
            return formatTimestamp(context[0].parsed.x);
          },
          label: (context: any) => {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            
            if (label.includes('Sentiment')) {
              return `${label}: ${value.toFixed(3)}`;
            } else if (label.includes('Prediction') || label.includes('Trade')) {
              return `${label}: ${formatCurrency(value)}`;
            } else {
              return `${label}: ${value}`;
            }
          },
        },
      },
    },
    scales: {
      x: {
        type: 'time' as const,
        time: {
          displayFormats: {
            minute: 'HH:mm',
            hour: 'HH:mm',
            day: 'MMM dd',
          },
        },
        title: {
          display: true,
          text: 'Time',
        },
      },
      ...(activeView === 'combined' ? {
        sentiment: {
          type: 'linear' as const,
          display: true,
          position: 'left' as const,
          min: -1,
          max: 1,
          title: {
            display: true,
            text: 'Sentiment Score',
          },
          grid: {
            drawOnChartArea: false,
          },
        },
        price: {
          type: 'linear' as const,
          display: true,
          position: 'right' as const,
          title: {
            display: true,
            text: 'Price ($)',
          },
          ticks: {
            callback: (value: any) => formatCurrency(value),
          },
        },
      } : activeView === 'sentiment' ? {
        y: {
          min: -1,
          max: 1,
          title: {
            display: true,
            text: 'Sentiment Score',
          },
        },
      } : {
        y: {
          title: {
            display: true,
            text: 'Price ($)',
          },
          ticks: {
            callback: (value: any) => formatCurrency(value),
          },
        },
      }),
    },
  };

  if (chartData.datasets.length === 0 || chartData.datasets.every(d => d.data.length === 0)) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex flex-wrap items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">Interactive Analysis</h3>
        </div>
        <div 
          className="flex items-center justify-center bg-gray-50 rounded-lg border-2 border-dashed border-gray-300"
          style={{ height }}
        >
          <div className="text-center">
            <div className="text-gray-400 text-lg mb-2">📊</div>
            <p className="text-gray-500">No data available for {symbol}</p>
            <p className="text-sm text-gray-400 mt-1">in the selected time range</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between mb-6 gap-4">
        <h3 className="text-lg font-medium text-gray-900">Interactive Analysis</h3>
        
        <div className="flex flex-wrap items-center gap-4">
          {/* View Toggle */}
          <div className="flex rounded-lg border border-gray-300 overflow-hidden">
            {(['sentiment', 'predictions', 'combined'] as ChartView[]).map((view) => (
              <button
                key={view}
                onClick={() => setActiveView(view)}
                className={`px-3 py-1 text-sm font-medium ${
                  activeView === view
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                {view.charAt(0).toUpperCase() + view.slice(1)}
              </button>
            ))}
          </div>

          {/* Time Range */}
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as TimeRange)}
            className="rounded-md border border-gray-300 px-3 py-1 text-sm"
          >
            <option value="1h">1 Hour</option>
            <option value="6h">6 Hours</option>
            <option value="24h">24 Hours</option>
            <option value="7d">7 Days</option>
            <option value="30d">30 Days</option>
          </select>

          {/* Show Trades Toggle */}
          <label className="flex items-center text-sm">
            <input
              type="checkbox"
              checked={showTrades}
              onChange={(e) => setShowTrades(e.target.checked)}
              className="mr-2 rounded"
            />
            Show Trades
          </label>
        </div>
      </div>

      {/* Chart */}
      <div style={{ height }}>
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}

export default InteractiveChart;