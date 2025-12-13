"""
TIER 1: Isolation Forest for Anomaly Detection
Trains on BENIGN traffic only to detect anomalies
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
from datetime import datetime
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import IoTDataPreprocessor


class Tier1AnomalyDetector:
    """Tier 1: Fast anomaly detection using Isolation Forest"""
    
    def __init__(self, contamination=0.1, n_estimators=100, max_samples=256):
        """
        Args:
            contamination: Expected proportion of outliers (0.1 = 10%)
            n_estimators: Number of isolation trees
            max_samples: Samples per tree (smaller = faster)
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        self.preprocessor = None
        self.training_date = None
    
    def train(self, benign_data: pd.DataFrame):
        """Train on BENIGN traffic only"""
        print(f"[TIER 1] Training Isolation Forest on {len(benign_data):,} benign samples")
        
        # Clean data
        benign_data = benign_data.replace([np.inf, -np.inf], np.nan)
        benign_data = benign_data.fillna(benign_data.median())
        
        # Fit the model
        self.model.fit(benign_data)
        self.training_date = datetime.now()
        
        print("[TIER 1] Training complete!")
        
        # Evaluate on training data
        predictions = self.model.predict(benign_data)
        anomalies = (predictions == -1).sum()
        normal = (predictions == 1).sum()
        
        print(f"\n[TIER 1] Training Set Evaluation:")
        print(f"  Normal: {normal:,} ({normal/len(benign_data)*100:.2f}%)")
        print(f"  Anomalies: {anomalies:,} ({anomalies/len(benign_data)*100:.2f}%)")
        
        return self
    
    def predict(self, X):
        """Predict: 1 = normal, -1 = anomaly"""
        return self.model.predict(X)
    
    def decision_function(self, X):
        """Return anomaly scores (lower = more anomalous)"""
        return self.model.decision_function(X)
    
    def evaluate(self, X_test, y_test_binary):
        """
        Evaluate on test set
        
        Args:
            X_test: Feature matrix
            y_test_binary: 0 = benign, 1 = attack
        """
        predictions = self.predict(X_test)
        
        # Convert: 1 (normal) -> 0 (benign), -1 (anomaly) -> 1 (attack)
        y_pred_binary = np.where(predictions == 1, 0, 1)
        
        print("\n[TIER 1] Evaluation Results:")
        print("\nClassification Report:")
        print(classification_report(
            y_test_binary, 
            y_pred_binary,
            target_names=['Benign', 'Attack']
        ))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test_binary, y_pred_binary)
        print(cm)
        print(f"\nTrue Negatives (Benign as Benign): {cm[0,0]}")
        print(f"False Positives (Benign as Attack): {cm[0,1]}")
        print(f"False Negatives (Attack as Benign): {cm[1,0]}")
        print(f"True Positives (Attack as Attack): {cm[1,1]}")
        
        # Calculate metrics
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nPrecision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        return {
            'predictions': y_pred_binary,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm
        }
    
    def save(self, filepath):
        """Save the model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'training_date': self.training_date,
                'params': self.model.get_params()
            }, f)
        print(f"\n[TIER 1] Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.training_date = data['training_date']
        print(f"[TIER 1] Model loaded from {filepath}")


def main():
    """Train Tier 1: Isolation Forest"""
    
    print("="*60)
    print("IOTGUARD - TIER 1: ISOLATION FOREST TRAINING")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = IoTDataPreprocessor(dataset_path="dataset/CSV/MERGED_CSV")
    
    # Load data
    print("\nLoading dataset...")
    df = preprocessor.load_data(sample_size=50000)  # Sample for faster training
    
    # Get benign data only for training
    print("\nExtracting BENIGN traffic...")
    benign_df = preprocessor.get_benign_only(df)
    benign_df = preprocessor.clean_data(benign_df)
    
    print(f"Benign samples: {len(benign_df):,}")
    
    # Scale features
    benign_scaled = preprocessor.scaler.fit_transform(benign_df)
    
    # Train Tier 1
    tier1 = Tier1AnomalyDetector(
        contamination=0.05,  # Expect 5% anomalies in benign traffic
        n_estimators=100,
        max_samples=256
    )
    tier1.train(benign_scaled)
    
    # Prepare test set (benign + attacks)
    print("\nPreparing test set...")
    df_test = preprocessor.load_data(sample_size=20000)
    df_test = preprocessor.clean_data(df_test)
    
    X_test = df_test.drop('Label', axis=1)
    y_test_binary = (df_test['Label'] != 'BENIGN').astype(int)
    
    X_test_scaled = preprocessor.scaler.transform(X_test)
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = tier1.evaluate(X_test_scaled, y_test_binary)
    
    # Save model
    tier1.save('models/tier1_isolation_forest.pkl')
    preprocessor.save_preprocessor('models/tier1_preprocessor.pkl')
    
    print("\n[TIER 1] Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
