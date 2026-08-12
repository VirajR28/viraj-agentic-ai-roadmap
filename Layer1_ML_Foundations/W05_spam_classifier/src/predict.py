import logging
from pathlib import Path
from typing import Tuple
import numpy as np

logger = logging.getLogger(__name__)


def predict_message(model, text: str) -> dict:
    """
    Classify a single message as spam or not spam.
    
    Args:
        model: Trained model (Pipeline)
        text: Message text to classify
        
    Returns:
        Dictionary with:
        - 'label': 'Spam' or 'Not Spam'
        - 'probability': Confidence score (0-1)
        - 'raw_prediction': 0 or 1
    """
    # Predict
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    
    # Get the probability of the predicted class
    confidence = probabilities[prediction]
    
    label_map = {0: 'Not Spam', 1: 'Spam'}
    
    return {
        'label': label_map[prediction],
        'probability': float(confidence),
        'raw_prediction': int(prediction)
    }


def predict_batch(model, texts: list) -> list:
    """
    Classify multiple messages at once.
    
    Args:
        model: Trained model (Pipeline)
        texts: List of message texts
        
    Returns:
        List of prediction dictionaries
    """
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)
    
    label_map = {0: 'Not Spam', 1: 'Spam'}
    
    results = []
    for pred, probs in zip(predictions, probabilities):
        confidence = probs[pred]
        results.append({
            'label': label_map[pred],
            'probability': float(confidence),
            'raw_prediction': int(pred)
        })
    
    return results


def analyze_prediction(model, text: str, preprocessor=None) -> dict:
    """
    Analyze a prediction in detail.
    
    Returns both the prediction and relevant features from TF-IDF.
    
    Args:
        model: Trained model (Pipeline)
        text: Message text
        preprocessor: Optional preprocessing function to clean text first
        
    Returns:
        Dictionary with prediction and feature analysis
    """
    # Preprocess if function provided
    if preprocessor:
        text = preprocessor(text)
    
    prediction = predict_message(model, text)
    
    # Get the TF-IDF vectorizer
    tfidf = model.named_steps['tfidf']
    
    # Transform the text
    tfidf_vector = tfidf.transform([text])
    
    # Get feature names and values
    feature_names = tfidf.get_feature_names_out()
    feature_values = tfidf_vector.toarray()[0]
    
    # Get non-zero features (the words/bigrams that appeared in the text)
    non_zero_indices = tfidf_vector.nonzero()[1]
    
    top_features = []
    for idx in non_zero_indices:
        if feature_values[idx] > 0:
            top_features.append({
                'feature': feature_names[idx],
                'tfidf_score': float(feature_values[idx])
            })
    
    # Sort by TF-IDF score
    top_features = sorted(top_features, key=lambda x: x['tfidf_score'], reverse=True)[:10]
    
    return {
        'text': text,
        'prediction': prediction,
        'top_features': top_features
    }
