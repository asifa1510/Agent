"""
Base API client with retry logic and circuit breaker patterns
"""
import asyncio
import logging
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from circuitbreaker import circuit

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base exception for API errors"""
    pass


class RateLimitError(APIError):
    """Exception for rate limit errors"""
    pass


class BaseAPIClient(ABC):
    """Base class for external API clients with common functionality"""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, rate_limit_per_minute: int = 60):
        self.base_url = base_url
        self.api_key = api_key
        self.rate_limit_per_minute = rate_limit_per_minute
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = asyncio.Semaphore(rate_limit_per_minute)
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self._create_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()
        
    async def _create_session(self):
        """Create aiohttp session"""
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            
    async def _close_session(self):
        """Close aiohttp session"""
        if self._session:
            await self._session.close()
            self._session = None
            
    @property
    def session(self) -> aiohttp.ClientSession:
        """Get the current session"""
        if not self._session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        return self._session
        
    def get_headers(self) -> Dict[str, str]:
        """Get default headers for requests"""
        headers = {
            'User-Agent': 'SentimentTradingAgent/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        if self.api_key:
            headers.update(self._get_auth_headers())
            
        return headers
        
    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers - implemented by subclasses"""
        pass
        
    @circuit(failure_threshold=5, recovery_timeout=60, expected_exception=APIError)
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic and circuit breaker"""
        
        # Rate limiting
        async with self._rate_limiter:
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            headers = self.get_headers()
            
            try:
                async with self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=data
                ) as response:
                    
                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"Rate limited. Waiting {retry_after} seconds")
                        await asyncio.sleep(retry_after)
                        raise RateLimitError(f"Rate limited for {retry_after} seconds")
                    
                    # Handle other HTTP errors
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(f"API error {response.status}: {error_text}")
                        raise APIError(f"HTTP {response.status}: {error_text}")
                    
                    return await response.json()
                    
            except aiohttp.ClientError as e:
                logger.error(f"Client error for {url}: {e}")
                raise APIError(f"Client error: {e}")
            except asyncio.TimeoutError as e:
                logger.error(f"Timeout error for {url}: {e}")
                raise APIError(f"Timeout error: {e}")
            
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make GET request"""
        return await self._make_request('GET', endpoint, params=params)
        
    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make POST request"""
        return await self._make_request('POST', endpoint, data=data)