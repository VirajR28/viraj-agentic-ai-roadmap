import logging
from pathlib import Path
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

logger = logging.getLogger(__name__)


def train_model(X_train, y_train, model_save_path: str = None):
    """
    Train a spam classification model using TF-IDF and Logistic Regression.
    
    Creates a pipeline that:
    1. Converts text to TF-IDF features
    2. Trains Logistic Regression on the features
    
    Args:
        X_train: Training text messages
        y_train: Training labels
        model_save_path: Path to save the trained model
        
    Returns:
        Trained pipeline model
    """
    logger.info("Training model...")
    
    # Create a pipeline
    # This ensures TF-IDF is fit only on training data (prevents data leakage)
    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=5000,
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,  # Allow single occurrence terms for small datasets
            max_df=0.95
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        ))
    ])
    
    model.fit(X_train, y_train)
    
    logger.info("Model training complete")
    
    if model_save_path:
        save_model(model, model_save_path)
    
    return model


def save_model(model, path: str):
    """
    Save the trained model to disk using joblib.
    
    Args:
        model: Trained model (Pipeline)
        path: Path to save the model
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


def load_model(path: str):
    """
    Load a trained model from disk.
    
    Args:
        path: Path to the saved model file
        
    Returns:
        Loaded model
    """
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        X_test: Test text messages
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    
    return metrics


def get_feature_importance(model, top_n: int = 10):
    """
    Get the most important features from the TF-IDF vectorizer.
    
    Args:
        model: Trained pipeline model
        top_n: Number of top features to return
        
    Returns:
        DataFrame with features and their importance scores
    """
    # Get the TF-IDF vectorizer and classifier
    tfidf = model.named_steps['tfidf']
    classifier = model.named_steps['classifier']
    
    # Get feature names
    feature_names = tfidf.get_feature_names_out()
    
    # Get coefficients from Logistic Regression
    coefficients = classifier.coef_[0]
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': abs(coefficients)
    })
    
    # Sort by absolute coefficient
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
    
    return feature_importance.head(top_n)
