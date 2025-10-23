import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import Layout from './components/Layout';
import Navigation from './components/Navigation';
import Dashboard from './components/Dashboard';
import SentimentPage from './pages/SentimentPage';
import PredictionsPage from './pages/PredictionsPage';
import TradesPage from './pages/TradesPage';
import PortfolioPage from './pages/PortfolioPage';

function App() {
  return (
    <AppProvider>
      <Router>
        <Layout>
          <Navigation />
          <div className="mt-6">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/sentiment" element={<SentimentPage />} />
              <Route path="/predictions" element={<PredictionsPage />} />
              <Route path="/trades" element={<TradesPage />} />
              <Route path="/portfolio" element={<PortfolioPage />} />
            </Routes>
          </div>
        </Layout>
      </Router>
    </AppProvider>
  );
}

export default App;