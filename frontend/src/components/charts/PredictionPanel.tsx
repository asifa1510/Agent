import React, { useMemo } from 'react';
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
import { PricePrediction } from '../../types';
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

interface PredictionPanelProps {
  predictions: PricePrediction[];
  symbol?: string | null;
  height?: number;
  horizon?: '1d' | '3d' | '7d';
}

export function PredictionPanel({ 
  predictions, 
  symbol, 
  height = 400,
  horizon 
}: PredictionPanelProps) {
  const chartData = useMemo(() => {
    // Filter data by symbol and horizon if specified
    let filteredData = predictions;
    
    if (symbol) {
      filteredData = filteredData.filter(item => item.symbol === symbol);
    }
    
    if (horizon) {
      filteredData = filteredData.filter(item => item.horizon === horizon);
    }

    // Group by model and horizon
    const lstmData: any[] = [];
    const xgboostData: any[] = [];
    const confidenceUpper: any[] = [];
    const confidenceLower: any[] = [];

    filteredData.forEach(item => {
      const timestamp = item.timestamp * 1000;
      const point = { x: timestamp, y: item.predicted_price };
      const upperPoint = { x: timestamp, y: item.confidence_upper };
      const lowerPoint = { x: timestamp, y: item.confidence_lower };
      
      if (item.model === 'lstm') {
        lstmData.push(point);
      } else if (item.model === 'xgboost') {
        xgboostData.push(point);
      }
      
      confidenceUpper.push(upperPoint);
      confidenceLower.push(lowerPoint);
    });

    // Sort by timestamp
    lstmData.sort((a, b) => a.x - b.x);
    xgboostData.sort((a, b) => a.x - b.x);
    confidenceUpper.sort((a, b) => a.x - b.x);
    confidenceLower.sort((a, b) => a.x - b.x);

    const datasets = [];

    // Confidence band (fill between upper and lower)
    if (confidenceUpper.length > 0 && confidenceLower.length > 0) {
      datasets.push({
        label: 'Confidence Band',
        data: confidenceUpper,
        borderColor: 'rgba(156, 163, 175, 0.3)',
        backgroundColor: 'rgba(156, 163, 175, 0.1)',
        fill: '+1',
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 0,
        order: 3,
      });
      
      datasets.push({
        label: 'Confidence Lower',
        data: confidenceLower,
        borderColor: 'rgba(156, 163, 175, 0.3)',
        backgroundColor: 'rgba(156, 163, 175, 0.1)',
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 0,
        order: 4,
      });
    }

    // LSTM predictions
    if (lstmData.length > 0) {
      datasets.push({
        label: 'LSTM Model',
        data: lstmData,
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        tension: 0.1,
        pointRadius: 4,
        pointHoverRadius: 6,
        order: 1,
      });
    }

    // XGBoost predictions
    if (xgboostData.length > 0) {
      datasets.push({
        label: 'XGBoost Model',
        data: xgboostData,
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        tension: 0.1,
        pointRadius: 4,
        pointHoverRadius: 6,
        order: 2,
      });
    }

    return { datasets };
  }, [predictions, symbol, horizon]);

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
        labels: {
          filter: (legendItem: any) => {
            // Hide the confidence lower line from legend
            return legendItem.text !== 'Confidence Lower';
          },
        },
      },
      title: {
        display: true,
        text: symbol 
          ? `${symbol} Price Predictions${horizon ? ` (${horizon})` : ''}`
          : `Price Predictions${horizon ? ` (${horizon})` : ''}`,
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
            
            if (label.includes('Confidence')) {
              return `${label}: ${formatCurrency(value)}`;
            } else {
              return `${label}: ${formatCurrency(value)}`;
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
      y: {
        title: {
          display: true,
          text: 'Price ($)',
        },
        ticks: {
          callback: (value: any) => formatCurrency(value),
        },
      },
    },
  };

  // Calculate prediction summary stats
  const summaryStats = useMemo(() => {
    if (chartData.datasets.length === 0) return null;

    const allPredictions = predictions.filter(p => 
      (!symbol || p.symbol === symbol) && 
      (!horizon || p.horizon === horizon)
    );

    if (allPredictions.length === 0) return null;

    const latest = allPredictions.reduce((latest, current) => 
      current.timestamp > latest.timestamp ? current : latest
    );

    const avgPrice = allPredictions.reduce((sum, p) => sum + p.predicted_price, 0) / allPredictions.length;
    const avgConfidenceRange = allPredictions.reduce((sum, p) => 
      sum + (p.confidence_upper - p.confidence_lower), 0
    ) / allPredictions.length;

    return {
      latest,
      avgPrice,
      avgConfidenceRange,
      totalPredictions: allPredictions.length,
    };
  }, [predictions, symbol, horizon, chartData.datasets]);

  if (chartData.datasets.length === 0 || chartData.datasets.every(d => d.data.length === 0)) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div 
          className="flex items-center justify-center bg-gray-50 rounded-lg border-2 border-dashed border-gray-300"
          style={{ height }}
        >
          <div className="text-center">
            <div className="text-gray-400 text-lg mb-2">📈</div>
            <p className="text-gray-500">No prediction data available</p>
            {symbol && (
              <p className="text-sm text-gray-400 mt-1">for {symbol}</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-6">
      {/* Summary Stats */}
      {summaryStats && (
        <div className="mb-6 grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-gray-50 rounded-lg">
          <div className="text-center">
            <div className="text-lg font-semibold text-gray-900">
              {formatCurrency(summaryStats.latest.predicted_price)}
            </div>
            <div className="text-sm text-gray-500">Latest Prediction</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-gray-900">
              {formatCurrency(summaryStats.avgPrice)}
            </div>
            <div className="text-sm text-gray-500">Average Price</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-gray-900">
              ±{formatCurrency(summaryStats.avgConfidenceRange / 2)}
            </div>
            <div className="text-sm text-gray-500">Avg Confidence</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-gray-900">
              {summaryStats.totalPredictions}
            </div>
            <div className="text-sm text-gray-500">Predictions</div>
          </div>
        </div>
      )}

      {/* Chart */}
      <div style={{ height }}>
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}

export default PredictionPanel;