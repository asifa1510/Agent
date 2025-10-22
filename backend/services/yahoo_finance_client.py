"""
Yahoo Finance API client for market data collection
"""
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import asyncio

import yfinance as yf
import pandas as pd
from ..config import settings
from .base_api_client import BaseAPIClient, APIError

logger = logging.getLogger(__name__)


class YahooFinanceClient(BaseAPIClient):
    """Yahoo Finance client for market data with retry logic and circuit breaker"""
    
    def __init__(self):
        # Yahoo Finance doesn't require API key but has rate limits
        super().__init__(
            base_url="https://query1.finance.yahoo.com",
            api_key=None,
            rate_limit_per_minute=60  # Conservative rate limiting
        )
        
    def _get_auth_headers(self) -> Dict[str, str]:
        """Yahoo Finance doesn't require authentication headers"""
        return {}
        
    async def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive stock information
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Dictionary containing stock information
        """
        try:
            # Run yfinance operations in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            info = await loop.run_in_executor(None, lambda: ticker.info)
            
            # Process and standardize the information
            processed_info = self._process_stock_info(info, symbol)
            logger.info(f"Retrieved stock info for {symbol}")
            return processed_info
            
        except Exception as e:
            logger.error(f"Error getting stock info for {symbol}: {e}")
            raise APIError(f"Failed to get stock info for {symbol}: {e}")
            
    async def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1mo",
        interval: str = "1d",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical price data
        
        Args:
            symbol: Stock symbol
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date: Start date for custom range
            end_date: End date for custom range
            
        Returns:
            List of historical price data dictionaries
        """
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            
            # Get historical data
            if start_date and end_date:
                hist_data = await loop.run_in_executor(
                    None, 
                    lambda: ticker.history(start=start_date, end=end_date, interval=interval)
                )
            else:
                hist_data = await loop.run_in_executor(
                    None, 
                    lambda: ticker.history(period=period, interval=interval)
                )
                
            # Convert to list of dictionaries
            historical_data = []
            for date, row in hist_data.iterrows():
                historical_data.append({
                    'symbol': symbol,
                    'date': date.isoformat(),
                    'timestamp': int(date.timestamp()),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']),
                    'adj_close': float(row.get('Adj Close', row['Close']))
                })
                
            logger.info(f"Retrieved {len(historical_data)} historical data points for {symbol}")
            return historical_data
            
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            raise APIError(f"Failed to get historical data for {symbol}: {e}")
            
    async def get_real_time_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time price data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing current price information
        """
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            
            # Get current data (last trading day)
            hist_data = await loop.run_in_executor(
                None, 
                lambda: ticker.history(period="1d", interval="1m")
            )
            
            if hist_data.empty:
                raise APIError(f"No real-time data available for {symbol}")
                
            # Get the most recent data point
            latest = hist_data.iloc[-1]
            latest_time = hist_data.index[-1]
            
            # Get additional info
            info = await loop.run_in_executor(None, lambda: ticker.info)
            
            price_data = {
                'symbol': symbol,
                'timestamp': int(latest_time.timestamp()),
                'datetime': latest_time.isoformat(),
                'price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': int(latest['Volume']),
                'previous_close': float(info.get('previousClose', latest['Close'])),
                'change': float(latest['Close'] - info.get('previousClose', latest['Close'])),
                'change_percent': float((latest['Close'] - info.get('previousClose', latest['Close'])) / info.get('previousClose', latest['Close']) * 100) if info.get('previousClose') else 0.0,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'volume_avg': info.get('averageVolume'),
                'day_high': info.get('dayHigh'),
                'day_low': info.get('dayLow'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow')
            }
            
            logger.info(f"Retrieved real-time price for {symbol}: ${price_data['price']}")
            return price_data
            
        except Exception as e:
            logger.error(f"Error getting real-time price for {symbol}: {e}")
            raise APIError(f"Failed to get real-time price for {symbol}: {e}")
            
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get real-time prices for multiple symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to price data
        """
        results = {}
        
        # Process symbols in batches to avoid rate limiting
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [self.get_real_time_price(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error getting price for {symbol}: {result}")
                    results[symbol] = None
                else:
                    results[symbol] = result
                    
            # Rate limiting between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(1)
                
        logger.info(f"Retrieved prices for {len([r for r in results.values() if r is not None])}/{len(symbols)} symbols")
        return results
        
    async def get_financial_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get financial statements and ratios
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing financial data
        """
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(None, yf.Ticker, symbol)
            
            # Get financial data
            financials = await loop.run_in_executor(None, lambda: ticker.financials)
            balance_sheet = await loop.run_in_executor(None, lambda: ticker.balance_sheet)
            cashflow = await loop.run_in_executor(None, lambda: ticker.cashflow)
            info = await loop.run_in_executor(None, lambda: ticker.info)
            
            financial_data = {
                'symbol': symbol,
                'market_cap': info.get('marketCap'),
                'enterprise_value': info.get('enterpriseValue'),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'peg_ratio': info.get('pegRatio'),
                'price_to_book': info.get('priceToBook'),
                'price_to_sales': info.get('priceToSalesTrailing12Months'),
                'debt_to_equity': info.get('debtToEquity'),
                'return_on_equity': info.get('returnOnEquity'),
                'return_on_assets': info.get('returnOnAssets'),
                'profit_margin': info.get('profitMargins'),
                'operating_margin': info.get('operatingMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'earnings_growth': info.get('earningsGrowth'),
                'dividend_yield': info.get('dividendYield'),
                'payout_ratio': info.get('payoutRatio'),
                'beta': info.get('beta'),
                'shares_outstanding': info.get('sharesOutstanding'),
                'float_shares': info.get('floatShares'),
                'insider_ownership': info.get('heldPercentInsiders'),
                'institutional_ownership': info.get('heldPercentInstitutions')
            }
            
            logger.info(f"Retrieved financial data for {symbol}")
            return financial_data
            
        except Exception as e:
            logger.error(f"Error getting financial data for {symbol}: {e}")
            raise APIError(f"Failed to get financial data for {symbol}: {e}")
            
    def _process_stock_info(self, info: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Process raw stock info into standardized format"""
        return {
            'symbol': symbol,
            'company_name': info.get('longName', info.get('shortName', symbol)),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'country': info.get('country'),
            'website': info.get('website'),
            'business_summary': info.get('longBusinessSummary'),
            'employees': info.get('fullTimeEmployees'),
            'market_cap': info.get('marketCap'),
            'enterprise_value': info.get('enterpriseValue'),
            'current_price': info.get('currentPrice'),
            'previous_close': info.get('previousClose'),
            'open': info.get('open'),
            'day_high': info.get('dayHigh'),
            'day_low': info.get('dayLow'),
            'volume': info.get('volume'),
            'average_volume': info.get('averageVolume'),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'price_to_book': info.get('priceToBook'),
            'dividend_yield': info.get('dividendYield'),
            'beta': info.get('beta'),
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
            'recommendation': info.get('recommendationKey'),
            'target_price': info.get('targetMeanPrice'),
            'earnings_date': info.get('earningsDate'),
            'ex_dividend_date': info.get('exDividendDate')
        }
        
    async def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for stock symbols by company name or symbol
        
        Args:
            query: Search query (company name or symbol)
            
        Returns:
            List of matching symbols with basic info
        """
        try:
            # This is a simplified implementation
            # In a production system, you might want to use a dedicated symbol search API
            
            # Try to get info for the query as a symbol
            try:
                info = await self.get_stock_info(query.upper())
                return [info]
            except:
                pass
                
            # For now, return empty list if direct symbol lookup fails
            # In production, implement proper symbol search functionality
            logger.warning(f"Symbol search not fully implemented for query: {query}")
            return []
            
        except Exception as e:
            logger.error(f"Error searching symbols for query '{query}': {e}")
            return []