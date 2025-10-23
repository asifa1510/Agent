import React, { useState } from 'react';
import { useAppContext } from '../context/AppContext';
import { useApi } from '../hooks/useApi';

const POPULAR_SYMBOLS = [
  'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 
  'META', 'NVDA', 'NFLX', 'SPY', 'QQQ'
];

export function SymbolSelector() {
  const { state } = useAppContext();
  const { setSelectedSymbol } = useApi();
  const [customSymbol, setCustomSymbol] = useState('');
  const [showCustomInput, setShowCustomInput] = useState(false);

  const handleSymbolSelect = (symbol: string) => {
    setSelectedSymbol(symbol);
    setShowCustomInput(false);
    setCustomSymbol('');
  };

  const handleCustomSymbolSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (customSymbol.trim()) {
      handleSymbolSelect(customSymbol.trim().toUpperCase());
    }
  };

  const handleClearSelection = () => {
    setSelectedSymbol(null);
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Symbol Selection</h3>
      
      {/* Current selection */}
      {state.selectedSymbol && (
        <div className="mb-4 p-3 bg-blue-50 rounded-lg flex justify-between items-center">
          <span className="text-blue-800 font-medium">
            Selected: {state.selectedSymbol}
          </span>
          <button
            onClick={handleClearSelection}
            className="text-blue-600 hover:text-blue-800 text-sm"
          >
            Clear
          </button>
        </div>
      )}

      {/* Popular symbols */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">Popular Symbols</h4>
        <div className="grid grid-cols-5 gap-2">
          {POPULAR_SYMBOLS.map((symbol) => (
            <button
              key={symbol}
              onClick={() => handleSymbolSelect(symbol)}
              className={`px-3 py-2 text-sm rounded-md border transition-colors ${
                state.selectedSymbol === symbol
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-gray-700 border-gray-300 hover:bg-gray-50'
              }`}
            >
              {symbol}
            </button>
          ))}
        </div>
      </div>

      {/* Custom symbol input */}
      <div>
        {!showCustomInput ? (
          <button
            onClick={() => setShowCustomInput(true)}
            className="text-blue-600 hover:text-blue-800 text-sm font-medium"
          >
            + Add Custom Symbol
          </button>
        ) : (
          <form onSubmit={handleCustomSymbolSubmit} className="flex space-x-2">
            <input
              type="text"
              value={customSymbol}
              onChange={(e) => setCustomSymbol(e.target.value)}
              placeholder="Enter symbol (e.g., AAPL)"
              className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              autoFocus
            />
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              Add
            </button>
            <button
              type="button"
              onClick={() => {
                setShowCustomInput(false);
                setCustomSymbol('');
              }}
              className="px-4 py-2 bg-gray-300 text-gray-700 text-sm rounded-md hover:bg-gray-400 focus:outline-none focus:ring-2 focus:ring-gray-500"
            >
              Cancel
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

export default SymbolSelector;