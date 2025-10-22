/**
 * Frontend configuration settings
 */

// Declare process for TypeScript
declare const process: {
    env: {
        [key: string]: string | undefined;
    };
};

export const config = {
    // API Configuration
    apiBaseUrl: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',

    // Chart Configuration
    chartUpdateInterval: 30000, // 30 seconds
    maxDataPoints: 100,

    // WebSocket Configuration (for future real-time updates)
    wsUrl: process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws',

    // UI Configuration
    theme: {
        colors: {
            primary: '#3B82F6',
            secondary: '#10B981',
            danger: '#EF4444',
            warning: '#F59E0B',
            success: '#10B981',
        }
    },

    // Data refresh intervals
    refreshIntervals: {
        sentiment: 30000, // 30 seconds
        predictions: 60000, // 1 minute
        portfolio: 30000, // 30 seconds
        trades: 15000, // 15 seconds
    }
};

export default config;