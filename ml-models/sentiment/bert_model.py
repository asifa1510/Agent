"""
BERT-based sentiment analysis model for social media text processing.
Implements sentiment classification with confidence scoring.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

logger = logging.getLogger(__name__)


class BERTSentimentModel:
    """
    BERT model for sentiment analysis with confidence scoring.
    Supports real-time inference for social media text processing.
    """
    
    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        Initialize BERT sentiment model.
        
        Args:
            model_name: Pre-trained BERT model name from HuggingFace
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self) -> None:
        """Load the pre-trained BERT model and tokenizer."""
        try:
            logger.info(f"Loading BERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("BERT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {str(e)}")
            raise
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for BERT inference.
        
        Args:
            text: Input text to process
            
        Returns:
            Tokenized inputs for BERT model
        """
        # Clean and truncate text
        text = text.strip()[:512]  # BERT max length
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs
    
    def predict_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Predict sentiment score and confidence for input text.
        
        Args:
            text: Input text for sentiment analysis
            
        Returns:
            Tuple of (sentiment_score, confidence)
            sentiment_score: -1 (negative) to 1 (positive)
            confidence: 0 to 1
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess text
            inputs = self.preprocess_text(text)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Convert to probabilities
            probabilities = torch.softmax(logits, dim=-1)
            probs = probabilities.cpu().numpy()[0]
            
            # Convert 5-class sentiment to -1 to 1 scale
            # Assuming model outputs: [very_negative, negative, neutral, positive, very_positive]
            if len(probs) == 5:
                # Weighted average: -1, -0.5, 0, 0.5, 1
                weights = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
                sentiment_score = np.sum(probs * weights)
                confidence = np.max(probs)
            elif len(probs) == 3:
                # 3-class: [negative, neutral, positive]
                weights = np.array([-1.0, 0.0, 1.0])
                sentiment_score = np.sum(probs * weights)
                confidence = np.max(probs)
            else:
                # Binary classification: [negative, positive]
                sentiment_score = probs[1] * 2 - 1  # Convert 0-1 to -1-1
                confidence = np.max(probs)
            
            return float(sentiment_score), float(confidence)
            
        except Exception as e:
            logger.error(f"Sentiment prediction failed: {str(e)}")
            raise
    
    def batch_predict(self, texts: List[str]) -> List[Tuple[float, float]]:
        """
        Predict sentiment for multiple texts in batch.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of (sentiment_score, confidence) tuples
        """
        results = []
        for text in texts:
            try:
                sentiment, confidence = self.predict_sentiment(text)
                results.append((sentiment, confidence))
            except Exception as e:
                logger.warning(f"Failed to process text: {str(e)}")
                results.append((0.0, 0.0))  # Neutral with zero confidence
        
        return results


def model_fn(model_dir: str) -> BERTSentimentModel:
    """
    SageMaker model loading function.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Loaded BERT sentiment model
    """
    model = BERTSentimentModel()
    model.load_model()
    return model


def input_fn(request_body: str, content_type: str = "application/json") -> Dict:
    """
    SageMaker input processing function.
    
    Args:
        request_body: Raw request body
        content_type: Content type of request
        
    Returns:
        Parsed input data
    """
    if content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data: Dict, model: BERTSentimentModel) -> Dict:
    """
    SageMaker prediction function.
    
    Args:
        input_data: Parsed input data
        model: Loaded BERT model
        
    Returns:
        Prediction results
    """
    try:
        if "text" in input_data:
            # Single text prediction
            text = input_data["text"]
            sentiment, confidence = model.predict_sentiment(text)
            
            return {
                "sentiment_score": sentiment,
                "confidence": confidence,
                "classification": "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
            }
        
        elif "texts" in input_data:
            # Batch prediction
            texts = input_data["texts"]
            results = model.batch_predict(texts)
            
            predictions = []
            for i, (sentiment, confidence) in enumerate(results):
                predictions.append({
                    "text_index": i,
                    "sentiment_score": sentiment,
                    "confidence": confidence,
                    "classification": "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
                })
            
            return {"predictions": predictions}
        
        else:
            raise ValueError("Input must contain 'text' or 'texts' field")
            
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {"error": str(e)}


def output_fn(prediction: Dict, accept: str = "application/json") -> str:
    """
    SageMaker output processing function.
    
    Args:
        prediction: Model prediction results
        accept: Accepted response format
        
    Returns:
        Formatted response
    """
    if accept == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")