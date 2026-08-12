import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_dataset(data_path: str = None) -> pd.DataFrame:
    """
    Load the spam dataset from a local CSV file.
    
    If the file doesn't exist, download it from the UCI repository.
    
    Args:
        data_path: Path to the dataset CSV file. If None, looks for data/spam.csv
        
    Returns:
        DataFrame with columns 'label' and 'text'
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "spam.csv"
    
    data_path = Path(data_path)
    
    # If file doesn't exist, download it
    if not data_path.exists():
        logger.info(f"Dataset not found at {data_path}. Downloading...")
        download_dataset(data_path)
    
    # Load the dataset
    df = pd.read_csv(data_path, encoding='latin-1')
    
    # Handle different column names from the source
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']].copy()
        df.columns = ['label', 'text']
    elif 'label' not in df.columns or 'text' not in df.columns:
        raise ValueError(f"Dataset must have 'label' and 'text' columns. Found: {df.columns.tolist()}")
    
    logger.info(f"Dataset loaded: {len(df)} messages")
    return df


def download_dataset(output_path):
    """
    Download the UCI SMS Spam Collection dataset.
    
    Args:
        output_path: Path where to save the downloaded file
    """
    import urllib.request
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection/SMSSpamCollection"
    
    logger.info(f"Downloading from {url}")
    
    try:
        # Download the file
        urllib.request.urlretrieve(url, output_path)
        
        # The downloaded file doesn't have headers, add them
        df = pd.read_csv(output_path, sep='\t', header=None, encoding='utf-8')
        df.columns = ['v1', 'v2']
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Dataset downloaded successfully to {output_path}")
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def clean_text(text: str) -> str:
    """
    Clean a single text message.
    
    - Convert to lowercase
    - Remove special characters (keep alphanumeric and spaces)
    - Remove extra whitespace
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs (basic)
    text = text.replace('http', '').replace('www', '')
    
    # Keep only alphanumeric characters and spaces
    text = ''.join(c if c.isalnum() or c.isspace() else '' for c in text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset.
    
    - Handle missing values
    - Remove duplicates
    - Clean text
    - Encode labels (ham -> 0, spam -> 1)
    
    Args:
        df: Raw dataset
        
    Returns:
        Preprocessed dataset
    """
    df = df.copy()
    
    # Check for missing values
    initial_rows = len(df)
    df = df.dropna(subset=['label', 'text'])
    if len(df) < initial_rows:
        logger.info(f"Dropped {initial_rows - len(df)} rows with missing values")
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['text'])
    if len(df) < initial_rows:
        logger.info(f"Dropped {initial_rows - len(df)} duplicate messages")
    
    # Clean text
    df['text'] = df['text'].apply(clean_text)
    
    # Encode labels: ham -> 0, spam -> 1
    # Handle both string labels and numeric labels
    label_mapping = {'ham': 0, 'spam': 1}
    
    if df['label'].dtype == 'object':
        # String labels - map them
        df['label'] = df['label'].str.lower().map(label_mapping)
        
        # Handle any label values we didn't map
        if df['label'].isna().any():
            logger.warning("Found unmapped label values, dropping those rows")
            df = df.dropna(subset=['label'])
    else:
        # Numeric labels - ensure they're int
        try:
            df['label'] = df['label'].astype(int)
        except (ValueError, TypeError):
            # If conversion fails, try mapping as strings
            df['label'] = df['label'].astype(str).str.lower().map(label_mapping)
            if df['label'].isna().any():
                logger.warning("Found unmapped label values, dropping those rows")
                df = df.dropna(subset=['label'])
    
    df['label'] = df['label'].astype(int)
    
    df = df.reset_index(drop=True)
    
    logger.info(f"Preprocessing complete: {len(df)} messages")
    return df


def get_class_distribution(labels) -> dict:
    """
    Get the distribution of classes in the labels.
    
    Args:
        labels: Array or Series of labels
        
    Returns:
        Dictionary with class counts and percentages
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    distribution = {}
    class_names = {0: 'Not Spam', 1: 'Spam'}
    
    for label, count in zip(unique, counts):
        percentage = (count / total) * 100
        distribution[class_names.get(label, label)] = {
            'count': int(count),
            'percentage': round(percentage, 2)
        }
    
    return distribution
