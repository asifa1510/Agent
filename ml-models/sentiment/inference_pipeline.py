"""
Inference pipeline for real-time sentiment analysis processing.
Integrates with SageMaker endpoint for social media text processing.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import aiohttp
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from .sagemaker_deployment import SentimentInferencePipeline

logger = logging.getLogger(__name__)


class RealTimeSentimentProcessor:
    """
    Real-time sentiment processing pipeline for social media data.
    Handles batch processing, confidence filtering, and aggregation.
    """
    
    def __init__(self, 
                 endpoint_name: str,
                 confidence_threshold: float = 0.7,
                 batch_size: int = 10,
                 max_workers: int = 4):
        """
        Initialize real-time sentiment processor.
        
        Args:
            endpoint_name: SageMaker endpoint name
            confidence_threshold: Minimum confidence for predictions
            batch_size: Batch size for processing
            max_workers: Maximum worker threads
        """
        self.endpoint_name = endpoint_name
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Initialize inference pipeline
        self.inference_pipeline = SentimentInferencePipeline(endpoint_name)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Processing metrics
        self.processed_count = 0
        self.filtered_count = 0
        self.error_count = 0
        
    def process_single_post(self, post_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process single social media post for sentiment.
        
        Args:
            post_data: Social media post data
            
        Returns:
            Processed sentiment data or None if filtered
        """
        try:
            result = self.inference_pipeline.process_social_media_post(post_data)
            
            if result is None:
                self.filtered_count += 1
                return None
            
            self.processed_count += 1
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Failed to process post: {str(e)}")
            return None
    
    def process_batch(self, posts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process batch of social media posts.
        
        Args:
            posts: List of social media post data
            
        Returns:
            List of processed sentiment data
        """
        results = []
        
        # Extract texts for batch prediction
        texts = [post.get("text", "") for post in posts]
        valid_indices = [i for i, text in enumerate(texts) if text.strip()]
        
        if not valid_indices:
            return results
        
        try:
            # Get batch predictions
            valid_texts = [texts[i] for i in valid_indices]
            batch_response = self.inference_pipeline.batch_predict_sentiment(valid_texts)
            
            predictions = batch_response.get("predictions", [])
            
            # Process each prediction
            for pred_idx, prediction in enumerate(predictions):
                original_idx = valid_indices[pred_idx]
                post_data = posts[original_idx]
                
                # Apply confidence filtering
                if prediction["confidence"] < self.confidence_threshold:
                    self.filtered_count += 1
                    continue
                
                # Create result with metadata
                result = {
                    "symbol": post_data.get("symbol"),
                    "timestamp": post_data.get("timestamp", int(time.time())),
                    "text": post_data.get("text"),
                    "sentiment_score": prediction["sentiment_score"],
                    "confidence": prediction["confidence"],
                    "classification": prediction["classification"],
                    "source": post_data.get("source", "unknown"),
                    "post_id": post_data.get("post_id")
                }
                
                results.append(result)
                self.processed_count += 1
                
        except Exception as e:
            self.error_count += len(valid_indices)
            logger.error(f"Batch processing failed: {str(e)}")
        
        return results
    
    async def process_stream(self, posts_stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process stream of social media posts asynchronously.
        
        Args:
            posts_stream: Stream of social media posts
            
        Returns:
            List of processed sentiment data
        """
        all_results = []
        
        # Process in batches
        for i in range(0, len(posts_stream), self.batch_size):
            batch = posts_stream[i:i + self.batch_size]
            
            # Process batch in thread pool
            loop = asyncio.get_event_loop()
            batch_results = await loop.run_in_executor(
                self.executor, 
                self.process_batch, 
                batch
            )
            
            all_results.extend(batch_results)
            
            # Log progress
            if i % (self.batch_size * 10) == 0:
                logger.info(f"Processed {i + len(batch)} posts")
        
        return all_results
    
    def aggregate_sentiment_by_symbol(self, 
                                    sentiment_data: List[Dict[str, Any]],
                                    time_window_minutes: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate sentiment scores by symbol within time windows.
        
        Args:
            sentiment_data: List of sentiment analysis results
            time_window_minutes: Time window for aggregation in minutes
            
        Returns:
            Aggregated sentiment data by symbol
        """
        symbol_aggregates = {}
        current_time = int(time.time())
        window_seconds = time_window_minutes * 60
        
        for data in sentiment_data:
            symbol = data.get("symbol")
            if not symbol:
                continue
            
            timestamp = data.get("timestamp", current_time)
            
            # Check if within time window
            if current_time - timestamp > window_seconds:
                continue
            
            if symbol not in symbol_aggregates:
                symbol_aggregates[symbol] = {
                    "symbol": symbol,
                    "sentiment_scores": [],
                    "confidences": [],
                    "post_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "latest_timestamp": timestamp
                }
            
            agg = symbol_aggregates[symbol]
            sentiment_score = data["sentiment_score"]
            confidence = data["confidence"]
            classification = data["classification"]
            
            # Update aggregates
            agg["sentiment_scores"].append(sentiment_score)
            agg["confidences"].append(confidence)
            agg["post_count"] += 1
            agg["latest_timestamp"] = max(agg["latest_timestamp"], timestamp)
            
            # Count classifications
            if classification == "positive":
                agg["positive_count"] += 1
            elif classification == "negative":
                agg["negative_count"] += 1
            else:
                agg["neutral_count"] += 1
        
        # Calculate final aggregated metrics
        for symbol, agg in symbol_aggregates.items():
            if agg["sentiment_scores"]:
                # Weighted average by confidence
                scores = agg["sentiment_scores"]
                confidences = agg["confidences"]
                
                weighted_sum = sum(s * c for s, c in zip(scores, confidences))
                confidence_sum = sum(confidences)
                
                agg["avg_sentiment"] = weighted_sum / confidence_sum if confidence_sum > 0 else 0.0
                agg["avg_confidence"] = sum(confidences) / len(confidences)
                agg["sentiment_volatility"] = float(np.std(scores)) if len(scores) > 1 else 0.0
            else:
                agg["avg_sentiment"] = 0.0
                agg["avg_confidence"] = 0.0
                agg["sentiment_volatility"] = 0.0
            
            # Clean up raw data
            del agg["sentiment_scores"]
            del agg["confidences"]
        
        return symbol_aggregates
    
    def get_processing_metrics(self) -> Dict[str, int]:
        """
        Get processing metrics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "processed_count": self.processed_count,
            "filtered_count": self.filtered_count,
            "error_count": self.error_count,
            "total_count": self.processed_count + self.filtered_count + self.error_count
        }
    
    def reset_metrics(self) -> None:
        """Reset processing metrics."""
        self.processed_count = 0
        self.filtered_count = 0
        self.error_count = 0


class SentimentAnalysisService:
    """
    High-level service for sentiment analysis operations.
    Provides interface for real-time and batch sentiment processing.
    """
    
    def __init__(self, endpoint_name: str):
        """
        Initialize sentiment analysis service.
        
        Args:
            endpoint_name: SageMaker endpoint name
        """
        self.endpoint_name = endpoint_name
        self.processor = RealTimeSentimentProcessor(endpoint_name)
        
    async def analyze_social_media_stream(self, 
                                        posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment for social media stream.
        
        Args:
            posts: List of social media posts
            
        Returns:
            Analysis results with aggregated sentiment data
        """
        try:
            # Process posts
            sentiment_results = await self.processor.process_stream(posts)
            
            # Aggregate by symbol
            symbol_aggregates = self.processor.aggregate_sentiment_by_symbol(sentiment_results)
            
            # Get processing metrics
            metrics = self.processor.get_processing_metrics()
            
            return {
                "sentiment_results": sentiment_results,
                "symbol_aggregates": symbol_aggregates,
                "processing_metrics": metrics,
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Stream analysis failed: {str(e)}")
            raise
    
    def analyze_single_post(self, post_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Analyze sentiment for single post.
        
        Args:
            post_data: Social media post data
            
        Returns:
            Sentiment analysis result or None if filtered
        """
        return self.processor.process_single_post(post_data)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on sentiment analysis service.
        
        Returns:
            Health check results
        """
        try:
            # Test with sample text
            test_post = {
                "text": "This is a test post for sentiment analysis",
                "symbol": "TEST",
                "timestamp": int(time.time())
            }
            
            result = self.analyze_single_post(test_post)
            
            return {
                "status": "healthy",
                "endpoint": self.endpoint_name,
                "test_result": result is not None,
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "endpoint": self.endpoint_name,
                "error": str(e),
                "timestamp": int(time.time())
            }