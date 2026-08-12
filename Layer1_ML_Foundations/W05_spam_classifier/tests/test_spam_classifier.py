"""
Tests for the spam classifier project.

Tests cover:
- Data loading and preprocessing
- Text cleaning
- Model training
- Predictions
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_preprocessing import (
    load_dataset, preprocess_data, clean_text, get_class_distribution
)
from train import train_model, save_model, load_model, evaluate_model
from predict import predict_message


class TestDataPreprocessing:
    """Test data preprocessing functions."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        # Original text with various issues
        original = "Hello WORLD!!!  http://example.com"
        cleaned = clean_text(original)
        
        # Should be lowercase
        assert cleaned.islower()
        # Should not have extra spaces
        assert '  ' not in cleaned
        # Should have removed URLs
        assert 'http' not in cleaned
    
    def test_clean_text_empty(self):
        """Test cleaning empty string."""
        result = clean_text("")
        assert result == ""
    
    def test_clean_text_non_string(self):
        """Test cleaning with non-string input."""
        result = clean_text(None)
        assert result == ""
    
    def test_get_class_distribution(self):
        """Test class distribution calculation."""
        labels = np.array([0, 0, 0, 1, 1])
        dist = get_class_distribution(labels)
        
        assert 'Not Spam' in dist
        assert 'Spam' in dist
        assert dist['Not Spam']['count'] == 3
        assert dist['Spam']['count'] == 2
        assert dist['Not Spam']['percentage'] == 60.0


class TestDataLoading:
    """Test dataset loading."""
    
    def test_load_dataset_structure(self):
        """Test that loaded dataset has correct structure."""
        # Use existing local data file if it exists
        import os
        from pathlib import Path
        
        # Navigate to data file location
        current_dir = Path(__file__).parent.parent
        data_path = current_dir / 'data' / 'spam.csv'
        
        if not data_path.exists():
            pytest.skip("Data file not available for testing")
        
        df = load_dataset(str(data_path))
        
        # Should have label and text columns
        assert 'label' in df.columns
        assert 'text' in df.columns
        
        # Should have data
        assert len(df) > 0
        
        # Labels should be either string (ham/spam) or numeric (0/1) 
        # The raw load doesn't preprocess, so may have strings
        assert all(label in ['ham', 'spam', 0, 1] for label in df['label'])
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Create sample data
        df = pd.DataFrame({
            'label': [0, 1, 0, 1],
            'text': ['Hello', 'FREE PRIZE!!', 'Hi there', 'CLICK NOW!!!']
        })
        
        df_clean = preprocess_data(df)
        
        # Should have label and text
        assert 'label' in df_clean.columns
        assert 'text' in df_clean.columns
        
        # Text should be cleaned (lowercase)
        assert df_clean['text'].str.islower().all()


class TestModel:
    """Test model training and prediction."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        X_train = [
            'win free prize',
            'click now',
            'hello friend',
            'meeting tomorrow'
        ]
        y_train = np.array([1, 1, 0, 0])
        return X_train, y_train
    
    def test_train_model(self, sample_data):
        """Test model training."""
        X_train, y_train = sample_data
        
        model = train_model(X_train, y_train)
        
        # Model should exist
        assert model is not None
        
        # Should be able to predict
        predictions = model.predict(X_train)
        assert len(predictions) == len(y_train)
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_message(self, sample_data):
        """Test single message prediction."""
        X_train, y_train = sample_data
        
        model = train_model(X_train, y_train)
        
        result = predict_message(model, "win free prize")
        
        # Should return dict with required keys
        assert 'label' in result
        assert 'probability' in result
        assert 'raw_prediction' in result
        
        # Probability should be between 0 and 1
        assert 0 <= result['probability'] <= 1
        
        # Label should be spam or not spam
        assert result['label'] in ['Spam', 'Not Spam']
    
    def test_model_save_load(self, sample_data, tmp_path):
        """Test saving and loading model."""
        X_train, y_train = sample_data
        
        model = train_model(X_train, y_train)
        
        # Save model
        model_path = tmp_path / 'test_model.joblib'
        save_model(model, str(model_path))
        
        # File should exist
        assert model_path.exists()
        
        # Load model
        loaded_model = load_model(str(model_path))
        
        # Should make same predictions
        original_pred = model.predict(['hello world'])
        loaded_pred = loaded_model.predict(['hello world'])
        
        assert original_pred[0] == loaded_pred[0]


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_load_train_evaluate_cycle(self):
        """Test complete cycle: load -> train -> evaluate."""
        import pytest
        from pathlib import Path
        
        # Use existing local data file
        current_dir = Path(__file__).parent.parent
        data_path = current_dir / 'data' / 'spam.csv'
        
        if not data_path.exists():
            pytest.skip("Data file not available for testing")
        
        # Load data
        df = load_dataset(str(data_path))
        
        # Preprocess
        df_clean = preprocess_data(df)
        
        # Split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            df_clean['text'], df_clean['label'],
            test_size=0.1, random_state=42, stratify=df_clean['label']
        )
        
        # Train
        model = train_model(X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        
        # Check metrics exist and are reasonable
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Accuracy should be reasonable (> 50% for binary classification)
        assert metrics['accuracy'] > 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
