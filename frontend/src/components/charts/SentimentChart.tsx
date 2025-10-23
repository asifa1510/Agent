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
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import { SentimentScore, ChartDataPoint } from '../../types';
import { formatTimestamp } from '../../utils/formatters';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

interface SentimentChartProps {
  sentimentData: SentimentScore[];
  symbol?: string | null;
  height?: number;
  showVolume?: boolean;
}

export function SentimentChart({ 
  sentimentData, 
  symbol, 
  height = 300,
  showVolume = false 
}: SentimentChartProps) {
  const chartData = useMemo(() => {
    // Filter data by symbol if specified
    const filteredData = symbol 
      ? sentimentData.filter(item => item.symbol === symbol)
      : sentimentData;

    // Group by source for multiple lines
    const twitterData: ChartDataPoint[] = [];
    const redditData: ChartDataPoint[] = [];
    const volumeData: ChartDataPoint[] = [];

    filteredData.forEach(item => {
      const point = { x: item.timestamp * 1000, y: item.score };
      const volumePoint = { x: item.timestamp * 1000, y: item.volume };
      
      if (item.source === 'twitter') {
        twitterData.push(point);
      } else if (item.source === 'reddit') {
        redditData.push(point);
      }
      
      if (showVolume) {
        volumeData.push(volumePoint);
      }
    });

    // Sort by timestamp
    twitterData.sort((a, b) => a.x - b.x);
    redditData.sort((a, b) => a.x - b.x);
    volumeData.sort((a, b) => a.x - b.x);

    const datasets = [
      {
        label: 'Twitter Sentiment',
        data: twitterData,
        borderColor: 'rgb(29, 161, 242)',
        backgroundColor: 'rgba(29, 161, 242, 0.1)',
        tension: 0.1,
        pointRadius: 3,
        pointHoverRadius: 5,
        yAxisID: 'sentiment',
      },
      {
        label: 'Reddit Sentiment',
        data: redditData,
        borderColor: 'rgb(255, 69, 0)',
        backgroundColor: 'rgba(255, 69, 0, 0.1)',
        tension: 0.1,
        pointRadius: 3,
        pointHoverRadius: 5,
        yAxisID: 'sentiment',
      }
    ];

    if (showVolume && volumeData.length > 0) {
      datasets.push({
        label: 'Post Volume',
        data: volumeData,
        borderColor: 'rgb(156, 163, 175)',
        backgroundColor: 'rgba(156, 163, 175, 0.1)',
        tension: 0.1,
        pointRadius: 2,
        pointHoverRadius: 4,
        yAxisID: 'volume',
      });
    }

    return { datasets };
  }, [sentimentData, symbol, showVolume]);

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
        text: symbol ? `${symbol} Sentiment Analysis` : 'Sentiment Analysis',
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
            
            if (label.includes('Volume')) {
              return `${label}: ${value} posts`;
            } else {
              return `${label}: ${value.toFixed(3)}`;
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
          drawOnChartArea: true,
        },
        ticks: {
          callback: (value: any) => value.toFixed(1),
        },
      },
      ...(showVolume && {
        volume: {
          type: 'linear' as const,
          display: true,
          position: 'right' as const,
          title: {
            display: true,
            text: 'Post Volume',
          },
          grid: {
            drawOnChartArea: false,
          },
        },
      }),
    },
  };

  if (chartData.datasets.every(dataset => dataset.data.length === 0)) {
    return (
      <div 
        className="flex items-center justify-center bg-gray-50 rounded-lg border-2 border-dashed border-gray-300"
        style={{ height }}
      >
        <div className="text-center">
          <div className="text-gray-400 text-lg mb-2">📊</div>
          <p className="text-gray-500">No sentiment data available</p>
          {symbol && (
            <p className="text-sm text-gray-400 mt-1">for {symbol}</p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div style={{ height }}>
      <Line data={chartData} options={options} />
    </div>
  );
}

export default SentimentChart;