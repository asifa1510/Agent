import React, { useState, useEffect } from 'react';
import { LoadingSpinner } from './LoadingSpinner';
import { config } from '../config';

interface SystemMetrics {
    api_calls: number;
    ml_inferences: number;
    trades_executed: number;
    total_errors: number;
}

interface LatencyData {
    average_ms: number[];
    maximum_ms: number[];
    timestamps: string[];
    error?: string;
}

interface PerformanceMetrics {
    api_latency: LatencyData;
    ml_inference_latency: LatencyData;
    trade_execution_latency: LatencyData;
}

interface ErrorData {
    total_errors: number;
    error_timeline: Array<{
        timestamp: string;
        count: number;
    }>;
}

export const MonitoringDashboard: React.FC = () => {
    const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
    const [performance, setPerformance] = useState<PerformanceMetrics | null>(null);
    const [errors, setErrors] = useState<ErrorData | null>(null);
    const [loading, setLoading] = useState(false);
    const [timeRange, setTimeRange] = useState(24);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadMonitoringData();
    }, [timeRange]);

    const loadMonitoringData = async () => {
        setLoading(true);
        setError(null);

        try {
            const [metricsResponse, performanceResponse, errorsResponse] = await Promise.all([
                fetch(`${config.apiBaseUrl}/monitoring/metrics?hours=${timeRange}`),
                fetch(`${config.apiBaseUrl}/monitoring/performance?hours=${timeRange}`),
                fetch(`${config.apiBaseUrl}/monitoring/errors?hours=${timeRange}`)
            ]);

            if (metricsResponse.ok) {
                const metricsData = await metricsResponse.json();
                setMetrics(metricsData.data.metrics);
            }

            if (performanceResponse.ok) {
                const performanceData = await performanceResponse.json();
                setPerformance(performanceData.data.performance_metrics);
            }

            if (errorsResponse.ok) {
                const errorsData = await errorsResponse.json();
                setErrors(errorsData.data.error_data);
            }

        } catch (err: any) {
            console.error('Error loading monitoring data:', err);
            setError('Failed to load monitoring data');
        } finally {
            setLoading(false);
        }
    };

    const flushBuffers = async () => {
        try {
            const response = await fetch(`${config.apiBaseUrl}/monitoring/flush`, { method: 'POST' });
            if (response.ok) {
                alert('Monitoring buffers flushed successfully');
                await loadMonitoringData();
            } else {
                throw new Error('Failed to flush buffers');
            }
        } catch (err: any) {
            console.error('Error flushing buffers:', err);
            setError('Failed to flush monitoring buffers');
        }
    };

    const formatNumber = (num: number) => {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    };

    const getStatusColor = (value: number, thresholds: { warning: number; critical: number }) => {
        if (value >= thresholds.critical) return 'text-red-600';
        if (value >= thresholds.warning) return 'text-yellow-600';
        return 'text-green-600';
    };

    return (
        <div className="p-6 space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-gray-900">System Monitoring</h2>
                <div className="flex space-x-4">
                    <select
                        value={timeRange}
                        onChange={(e) => setTimeRange(Number(e.target.value))}
                        className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                        <option value={1}>Last 1 hour</option>
                        <option value={6}>Last 6 hours</option>
                        <option value={24}>Last 24 hours</option>
                        <option value={72}>Last 3 days</option>
                        <option value={168}>Last 7 days</option>
                    </select>
                    <button
                        onClick={loadMonitoringData}
                        disabled={loading}
                        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                    >
                        {loading ? 'Loading...' : 'Refresh'}
                    </button>
                    <button
                        onClick={flushBuffers}
                        className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
                    >
                        Flush Buffers
                    </button>
                </div>
            </div>

            {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <p className="text-red-800">{error}</p>
                </div>
            )}

            {/* System Metrics Overview */}
            <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4">System Metrics Overview</h3>
                {loading ? (
                    <LoadingSpinner />
                ) : metrics ? (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center">
                            <div className="text-3xl font-bold text-blue-600">
                                {formatNumber(metrics.api_calls)}
                            </div>
                            <div className="text-sm text-gray-600">API Calls</div>
                        </div>
                        <div className="text-center">
                            <div className="text-3xl font-bold text-purple-600">
                                {formatNumber(metrics.ml_inferences)}
                            </div>
                            <div className="text-sm text-gray-600">ML Inferences</div>
                        </div>
                        <div className="text-center">
                            <div className="text-3xl font-bold text-green-600">
                                {formatNumber(metrics.trades_executed)}
                            </div>
                            <div className="text-sm text-gray-600">Trades Executed</div>
                        </div>
                        <div className="text-center">
                            <div className={`text-3xl font-bold ${getStatusColor(metrics.total_errors, { warning: 10, critical: 50 })}`}>
                                {formatNumber(metrics.total_errors)}
                            </div>
                            <div className="text-sm text-gray-600">Total Errors</div>
                        </div>
                    </div>
                ) : (
                    <div className="text-center text-gray-500 py-8">
                        No metrics data available
                    </div>
                )}
            </div>

            {/* Performance Metrics */}
            <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
                {loading ? (
                    <LoadingSpinner />
                ) : performance ? (
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        {/* API Latency */}
                        <div className="bg-gray-50 p-4 rounded">
                            <h4 className="font-medium text-gray-700 mb-2">API Latency</h4>
                            {performance.api_latency && !performance.api_latency.error ? (
                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <span className="text-sm text-gray-600">Average:</span>
                                        <span className="text-sm font-medium">
                                            {performance.api_latency.average_ms.length > 0
                                                ? `${Math.round(performance.api_latency.average_ms.reduce((a, b) => a + b, 0) / performance.api_latency.average_ms.length)}ms`
                                                : 'N/A'
                                            }
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-sm text-gray-600">Maximum:</span>
                                        <span className="text-sm font-medium">
                                            {performance.api_latency.maximum_ms.length > 0
                                                ? `${Math.round(Math.max(...performance.api_latency.maximum_ms))}ms`
                                                : 'N/A'
                                            }
                                        </span>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-sm text-gray-500">No data available</div>
                            )}
                        </div>

                        {/* ML Inference Latency */}
                        <div className="bg-gray-50 p-4 rounded">
                            <h4 className="font-medium text-gray-700 mb-2">ML Inference Latency</h4>
                            {performance.ml_inference_latency && !performance.ml_inference_latency.error ? (
                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <span className="text-sm text-gray-600">Average:</span>
                                        <span className="text-sm font-medium">
                                            {performance.ml_inference_latency.average_ms.length > 0
                                                ? `${Math.round(performance.ml_inference_latency.average_ms.reduce((a, b) => a + b, 0) / performance.ml_inference_latency.average_ms.length)}ms`
                                                : 'N/A'
                                            }
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-sm text-gray-600">Maximum:</span>
                                        <span className="text-sm font-medium">
                                            {performance.ml_inference_latency.maximum_ms.length > 0
                                                ? `${Math.round(Math.max(...performance.ml_inference_latency.maximum_ms))}ms`
                                                : 'N/A'
                                            }
                                        </span>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-sm text-gray-500">No data available</div>
                            )}
                        </div>

                        {/* Trade Execution Latency */}
                        <div className="bg-gray-50 p-4 rounded">
                            <h4 className="font-medium text-gray-700 mb-2">Trade Execution Latency</h4>
                            {performance.trade_execution_latency && !performance.trade_execution_latency.error ? (
                                <div className="space-y-2">
                                    <div className="flex justify-between">
                                        <span className="text-sm text-gray-600">Average:</span>
                                        <span className="text-sm font-medium">
                                            {performance.trade_execution_latency.average_ms.length > 0
                                                ? `${Math.round(performance.trade_execution_latency.average_ms.reduce((a, b) => a + b, 0) / performance.trade_execution_latency.average_ms.length)}ms`
                                                : 'N/A'
                                            }
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-sm text-gray-600">Maximum:</span>
                                        <span className="text-sm font-medium">
                                            {performance.trade_execution_latency.maximum_ms.length > 0
                                                ? `${Math.round(Math.max(...performance.trade_execution_latency.maximum_ms))}ms`
                                                : 'N/A'
                                            }
                                        </span>
                                    </div>
                                </div>
                            ) : (
                                <div className="text-sm text-gray-500">No data available</div>
                            )}
                        </div>
                    </div>
                ) : (
                    <div className="text-center text-gray-500 py-8">
                        No performance data available
                    </div>
                )}
            </div>

            {/* Error Tracking */}
            <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4">Error Tracking</h3>
                {loading ? (
                    <LoadingSpinner />
                ) : errors ? (
                    <div className="space-y-4">
                        <div className="flex justify-between items-center">
                            <span className="font-medium">Total Errors in Period:</span>
                            <span className={`text-lg font-bold ${getStatusColor(errors.total_errors, { warning: 10, critical: 50 })}`}>
                                {errors.total_errors}
                            </span>
                        </div>

                        {errors.error_timeline && errors.error_timeline.length > 0 && (
                            <div>
                                <h4 className="font-medium text-gray-700 mb-2">Error Timeline</h4>
                                <div className="space-y-2 max-h-40 overflow-y-auto">
                                    {errors.error_timeline.map((entry, index) => (
                                        <div key={index} className="flex justify-between text-sm">
                                            <span className="text-gray-600">
                                                {new Date(entry.timestamp).toLocaleString()}
                                            </span>
                                            <span className="font-medium text-red-600">
                                                {entry.count} errors
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="text-center text-gray-500 py-8">
                        No error data available
                    </div>
                )}
            </div>

            {/* System Health Indicators */}
            <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-lg font-semibold mb-4">System Health Indicators</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="text-center">
                        <div className="w-4 h-4 bg-green-500 rounded-full mx-auto mb-2"></div>
                        <div className="text-sm font-medium">API Services</div>
                        <div className="text-xs text-gray-600">Operational</div>
                    </div>
                    <div className="text-center">
                        <div className="w-4 h-4 bg-green-500 rounded-full mx-auto mb-2"></div>
                        <div className="text-sm font-medium">ML Models</div>
                        <div className="text-xs text-gray-600">Operational</div>
                    </div>
                    <div className="text-center">
                        <div className="w-4 h-4 bg-green-500 rounded-full mx-auto mb-2"></div>
                        <div className="text-sm font-medium">Data Pipeline</div>
                        <div className="text-xs text-gray-600">Operational</div>
                    </div>
                    <div className="text-center">
                        <div className="w-4 h-4 bg-green-500 rounded-full mx-auto mb-2"></div>
                        <div className="text-sm font-medium">Trading Engine</div>
                        <div className="text-xs text-gray-600">Operational</div>
                    </div>
                </div>
            </div>
        </div>
    );
};