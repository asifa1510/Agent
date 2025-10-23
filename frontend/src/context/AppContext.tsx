import React, { createContext, useContext, useReducer, ReactNode } from 'react';
import {
  SentimentScore,
  PricePrediction,
  Trade,
  TradeExplanation,
  PortfolioPosition,
  RiskMetrics
} from '../types';

// State interface
interface AppState {
  // Data states
  sentimentScores: SentimentScore[];
  predictions: PricePrediction[];
  trades: Trade[];
  explanations: TradeExplanation[];
  portfolioPositions: PortfolioPosition[];
  riskMetrics: RiskMetrics | null;
  
  // UI states
  selectedSymbol: string | null;
  loading: {
    sentiment: boolean;
    predictions: boolean;
    trades: boolean;
    explanations: boolean;
    portfolio: boolean;
  };
  error: string | null;
}

// Action types
type AppAction =
  | { type: 'SET_SENTIMENT_SCORES'; payload: SentimentScore[] }
  | { type: 'SET_PREDICTIONS'; payload: PricePrediction[] }
  | { type: 'SET_TRADES'; payload: Trade[] }
  | { type: 'SET_EXPLANATIONS'; payload: TradeExplanation[] }
  | { type: 'SET_PORTFOLIO_POSITIONS'; payload: PortfolioPosition[] }
  | { type: 'SET_RISK_METRICS'; payload: RiskMetrics }
  | { type: 'SET_SELECTED_SYMBOL'; payload: string | null }
  | { type: 'SET_LOADING'; payload: { key: keyof AppState['loading']; value: boolean } }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'CLEAR_ERROR' };

// Initial state
const initialState: AppState = {
  sentimentScores: [],
  predictions: [],
  trades: [],
  explanations: [],
  portfolioPositions: [],
  riskMetrics: null,
  selectedSymbol: null,
  loading: {
    sentiment: false,
    predictions: false,
    trades: false,
    explanations: false,
    portfolio: false,
  },
  error: null,
};

// Reducer
function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_SENTIMENT_SCORES':
      return { ...state, sentimentScores: action.payload };
    case 'SET_PREDICTIONS':
      return { ...state, predictions: action.payload };
    case 'SET_TRADES':
      return { ...state, trades: action.payload };
    case 'SET_EXPLANATIONS':
      return { ...state, explanations: action.payload };
    case 'SET_PORTFOLIO_POSITIONS':
      return { ...state, portfolioPositions: action.payload };
    case 'SET_RISK_METRICS':
      return { ...state, riskMetrics: action.payload };
    case 'SET_SELECTED_SYMBOL':
      return { ...state, selectedSymbol: action.payload };
    case 'SET_LOADING':
      return {
        ...state,
        loading: { ...state.loading, [action.payload.key]: action.payload.value }
      };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'CLEAR_ERROR':
      return { ...state, error: null };
    default:
      return state;
  }
}

// Context
interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

// Provider component
interface AppProviderProps {
  children: ReactNode;
}

export function AppProvider({ children }: AppProviderProps) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

// Custom hook to use the context
export function useAppContext() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
}

export default AppContext;