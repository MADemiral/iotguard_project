"""
TIER 2: LightGBM for 7-Class Attack Category Classification
Classifies attacks into: DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
from datetime import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import IoTDataPreprocessor


class Tier2CategoryClassifier:
    """Tier 2: Multi-class classification for attack categories"""
    
    def __init__(self, use_class_weights=True):
        """
        Args:
            use_class_weights: Whether to use class weights for imbalance
        """
        self.model = None
        self.use_class_weights = use_class_weights
        self.class_weights = None
        self.training_date = None
        self.label_encoder = None
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train LightGBM for 7-class classification"""
        
        print(f"\n[TIER 2] Training LightGBM on {len(X_train):,} samples")
        print(f"[TIER 2] Validation set: {len(X_val):,} samples")
        
        # Calculate class weights
        if self.use_class_weights:
            unique_classes = np.unique(y_train)
            class_weights = compute_class_weight(
                'balanced',
                classes=unique_classes,
                y=y_train
            )
            self.class_weights = dict(zip(unique_classes, class_weights))
            print(f"\n[TIER 2] Class weights: {self.class_weights}")
            
            # Create sample weights
            sample_weights = np.array([self.class_weights[label] for label in y_train])
        else:
            sample_weights = None
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            weight=sample_weights
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data
        )
        
        # LightGBM parameters optimized for multi-class
        params = {
            'objective': 'multiclass',
            'num_class': len(np.unique(y_train)),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'min_data_in_leaf': 20,
            'verbose': 1,
            'n_jobs': -1
        }
        
        # Train with early stopping
        print("\n[TIER 2] Starting training...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )
        
        self.training_date = datetime.now()
        
        print("\n[TIER 2] Training complete!")
        print(f"[TIER 2] Best iteration: {self.model.best_iteration}")
        print(f"[TIER 2] Best score: {self.model.best_score}")
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def evaluate(self, X_test, y_test, label_encoder=None):
        """Evaluate on test set"""
        
        print("\n[TIER 2] Evaluating on test set...")
        
        y_pred = self.predict(X_test)
        
        # Get class names
        if label_encoder:
            class_names = label_encoder.classes_
        else:
            class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n[TIER 2] Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\n[TIER 2] Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n[TIER 2] Confusion Matrix:")
        print(cm)
        
        # Feature importance
        importance = self.model.feature_importance(importance_type='gain')
        top_features = np.argsort(importance)[-20:]  # Top 20
        
        print("\n[TIER 2] Top 20 Important Features:")
        for idx in reversed(top_features):
            print(f"  Feature {idx}: {importance[idx]:.2f}")
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm,
            'feature_importance': importance
        }
    
    def plot_confusion_matrix(self, cm, class_names, save_path='results/tier2_confusion_matrix.png'):
        """Plot confusion matrix"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('TIER 2: Attack Category Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"\n[TIER 2] Confusion matrix saved to {save_path}")
        plt.close()
    
    def save(self, filepath):
        """Save the model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)
        
        # Save metadata
        metadata_path = filepath.replace('.txt', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'training_date': self.training_date,
                'class_weights': self.class_weights,
                'best_iteration': self.model.best_iteration
            }, f)
        
        print(f"\n[TIER 2] Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model"""
        self.model = lgb.Booster(model_file=filepath)
        
        # Load metadata
        metadata_path = filepath.replace('.txt', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.training_date = data['training_date']
                self.class_weights = data['class_weights']
        
        print(f"[TIER 2] Model loaded from {filepath}")


def main():
    """Train Tier 2: 7-Class Category Classifier"""
    
    print("="*60)
    print("IOTGUARD - TIER 2: 7-CLASS CATEGORY CLASSIFICATION")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = IoTDataPreprocessor(dataset_path="dataset/CSV/MERGED_CSV")
    
    # Load data (attacks only, no benign - benign filtered by Tier 1)
    print("\nLoading dataset...")
    df = preprocessor.load_data(sample_size=100000)  # Larger sample for better training
    
    # Remove benign (Tier 1 already handles it)
    df_attacks = df[df['Label'] != 'BENIGN'].copy()
    print(f"Attack samples: {len(df_attacks):,}")
    
    # Clean data
    df_attacks = preprocessor.clean_data(df_attacks)
    
    # Prepare for 7-class training
    print("\nPreparing 7-class training data...")
    data = preprocessor.prepare_for_training(df_attacks, label_type='7class', test_size=0.2)
    
    # Split validation from training
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Use 20% of training for validation
    val_size = int(len(X_train) * 0.2)
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Print class distribution
    print("\n7-Class Distribution in Training:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        class_name = data['label_encoder'].inverse_transform([cls])[0]
        print(f"  {class_name}: {count:,}")
    
    # Train Tier 2
    tier2 = Tier2CategoryClassifier(use_class_weights=True)
    tier2.train(X_train, y_train, X_val, y_val)
    tier2.label_encoder = data['label_encoder']
    
    # Evaluate
    results = tier2.evaluate(X_test, y_test, label_encoder=data['label_encoder'])
    
    # Plot confusion matrix
    tier2.plot_confusion_matrix(
        results['confusion_matrix'],
        data['label_encoder'].classes_,
        save_path='results/tier2_confusion_matrix.png'
    )
    
    # Save model
    tier2.save('models/tier2_7class_lgb.txt')
    preprocessor.save_preprocessor('models/tier2_preprocessor.pkl')
    
    print("\n[TIER 2] Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
