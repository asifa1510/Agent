import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { config } from '../config';
import {
  SentimentScore,
  PricePrediction,
  Trade,
  TradeExplanation,
  PortfolioPosition,
  RiskMetrics,
  ApiResponse
} from '../types';

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: config.apiBaseUrl,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Request interceptor for logging
    this.client.interceptors.request.use(
      (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('API Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Response Error:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  // Sentiment API endpoints
  async getSentimentScores(symbol?: string): Promise<SentimentScore[]> {
    const params = symbol ? { symbol } : {};
    const response: AxiosResponse<ApiResponse<SentimentScore[]>> = await this.client.get('/sentiment', { params });
    return response.data.data;
  }

  async getSentimentHistory(symbol: string, hours: number = 24): Promise<SentimentScore[]> {
    const response: AxiosResponse<ApiResponse<SentimentScore[]>> = await this.client.get(
      `/sentiment/${symbol}/history`,
      { params: { hours } }
    );
    return response.data.data;
  }

  // Predictions API endpoints
  async getPredictions(symbol?: string): Promise<PricePrediction[]> {
    const params = symbol ? { symbol } : {};
    const response: AxiosResponse<ApiResponse<PricePrediction[]>> = await this.client.get('/predictions', { params });
    return response.data.data;
  }

  async getPredictionHistory(symbol: string, days: number = 7): Promise<PricePrediction[]> {
    const response: AxiosResponse<ApiResponse<PricePrediction[]>> = await this.client.get(
      `/predictions/${symbol}/history`,
      { params: { days } }
    );
    return response.data.data;
  }

  // Trades API endpoints
  async getTrades(limit: number = 100): Promise<Trade[]> {
    const response: AxiosResponse<ApiResponse<Trade[]>> = await this.client.get('/trades', { params: { limit } });
    return response.data.data;
  }

  async getTradeHistory(symbol?: string, limit: number = 100): Promise<Trade[]> {
    const params = { limit, ...(symbol && { symbol }) };
    const response: AxiosResponse<ApiResponse<Trade[]>> = await this.client.get('/trades/history', { params });
    return response.data.data;
  }

  // Explanations API endpoints
  async getTradeExplanation(tradeId: string): Promise<TradeExplanation> {
    const response: AxiosResponse<ApiResponse<TradeExplanation>> = await this.client.get(`/explanations/trade/${tradeId}`);
    return response.data.data;
  }

  async getExplanations(limit: number = 50): Promise<TradeExplanation[]> {
    const response: AxiosResponse<ApiResponse<TradeExplanation[]>> = await this.client.get('/explanations', { params: { limit } });
    return response.data.data;
  }

  async generateTradeExplanation(trade: any, includeContext: boolean = true): Promise<TradeExplanation> {
    const response: AxiosResponse<ApiResponse<TradeExplanation>> = await this.client.post('/explanations/trade', {
      trade,
      include_context: includeContext
    });
    return response.data.data;
  }

  // Portfolio API endpoints
  async getPortfolioPositions(): Promise<PortfolioPosition[]> {
    const response: AxiosResponse<ApiResponse<PortfolioPosition[]>> = await this.client.get('/portfolio/positions');
    return response.data.data;
  }

  async getRiskMetrics(): Promise<RiskMetrics> {
    const response: AxiosResponse<ApiResponse<RiskMetrics>> = await this.client.get('/portfolio/risk');
    return response.data.data;
  }

  async getPortfolioPerformance(days: number = 30): Promise<any> {
    const response: AxiosResponse<ApiResponse<any>> = await this.client.get('/portfolio/performance', { params: { days } });
    return response.data.data;
  }

  // Simulation API endpoints
  async runSimulation(params: any): Promise<any> {
    const response: AxiosResponse<ApiResponse<any>> = await this.client.post('/portfolio/simulate', params);
    return response.data.data;
  }

  async getBenchmarkComparison(benchmark: string = 'SPY', days: number = 30): Promise<any> {
    const response: AxiosResponse<ApiResponse<any>> = await this.client.get('/portfolio/benchmark', { 
      params: { benchmark, days } 
    });
    return response.data.data;
  }

  // Integration API endpoints
  async runCompletePipeline(
    symbols: string[], 
    includeTrading: boolean = true, 
    includeExplanations: boolean = true
  ): Promise<any> {
    const response: AxiosResponse<ApiResponse<any>> = await this.client.post('/integration/run-pipeline', {
      symbols,
      include_trading: includeTrading,
      include_explanations: includeExplanations
    });
    return response.data.data;
  }

  async runPipelineAsync(
    symbols: string[], 
    includeTrading: boolean = true, 
    includeExplanations: boolean = true
  ): Promise<any> {
    const response: AxiosResponse<ApiResponse<any>> = await this.client.post('/integration/run-pipeline-async', {
      symbols,
      include_trading: includeTrading,
      include_explanations: includeExplanations
    });
    return response.data.data;
  }

  async getSystemHealth(): Promise<any> {
    const response: AxiosResponse<ApiResponse<any>> = await this.client.get('/integration/health');
    return response.data.data;
  }

  async getSystemMetrics(): Promise<any> {
    const response: AxiosResponse<ApiResponse<any>> = await this.client.get('/integration/metrics');
    return response.data.data;
  }

  async resetSystemMetrics(): Promise<any> {
    const response: AxiosResponse<ApiResponse<any>> = await this.client.post('/integration/metrics/reset');
    return response.data.data;
  }

  async ingestDataOnly(
    symbols: string[], 
    includeSocial: boolean = true, 
    includeNews: boolean = true, 
    includeMarket: boolean = true
  ): Promise<any> {
    const response: AxiosResponse<ApiResponse<any>> = await this.client.post('/integration/data/ingest', {
      symbols,
      include_social: includeSocial,
      include_news: includeNews,
      include_market: includeMarket
    });
    return response.data.data;
  }

  async getIntegrationStatus(): Promise<any> {
    const response: AxiosResponse<ApiResponse<any>> = await this.client.get('/integration/status');
    return response.data.data;
  }
}

// Create and export a singleton instance
export const apiClient = new ApiClient();
export default apiClient;