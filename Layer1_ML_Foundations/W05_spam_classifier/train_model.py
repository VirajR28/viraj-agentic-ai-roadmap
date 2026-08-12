"""
Training script to run the complete spam classifier pipeline.
This script runs independently of Jupyter.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))

from data_preprocessing import load_dataset, preprocess_data, get_class_distribution
from train import train_model, save_model, evaluate_model
from sklearn.model_selection import train_test_split

def main():
    print("=" * 80)
    print("SPAM CLASSIFIER TRAINING PIPELINE")
    print("=" * 80)
    
    # 1. Load data
    print("\n[1/5] Loading dataset...")
    df = load_dataset(str(project_root / 'data' / 'spam.csv'))
    print(f"      Loaded: {len(df)} messages")
    
    # 2. Preprocess
    print("\n[2/5] Preprocessing data...")
    df_clean = preprocess_data(df)
    print(f"      Cleaned: {len(df_clean)} messages")
    
    # Show class distribution
    dist = get_class_distribution(df_clean['label'])
    print(f"      Distribution: {dist['Not Spam']['count']} legitimate, {dist['Spam']['count']} spam")
    
    # 3. Train/test split
    print("\n[3/5] Splitting data (80/20 with stratification)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean['text'],
        df_clean['label'],
        test_size=0.2,
        random_state=42,
        stratify=df_clean['label']
    )
    print(f"      Training: {len(X_train)} messages")
    print(f"      Test: {len(X_test)} messages")
    
    # 4. Train model
    print("\n[4/5] Training Logistic Regression + TF-IDF...")
    model = train_model(X_train, y_train)
    print(f"      Model trained successfully")
    
    # 5. Evaluate
    print("\n[5/5] Evaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nAccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    
    # 6. Save model
    print("\n[6/6] Saving model...")
    model_path = project_root / 'models' / 'spam_classifier.joblib'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(model_path))
    print(f"      Model saved: {model_path}")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Run tests: pytest tests/ -v")
    print("2. Start Streamlit: streamlit run app/app.py")
    print("3. View notebook: jupyter notebook notebooks/spam_classifier.ipynb")
    
    return metrics

if __name__ == '__main__':
    metrics = main()
