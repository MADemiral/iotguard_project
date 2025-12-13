"""
TIER 3: Specific Attack Type Classification
Specialized models for DDoS variants and other attack categories
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

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import IoTDataPreprocessor


class Tier3SpecificClassifier:
    """Tier 3: Specific attack type within a category"""
    
    def __init__(self, attack_category: str, use_class_weights=True):
        """
        Args:
            attack_category: DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce
            use_class_weights: Whether to use class weights
        """
        self.attack_category = attack_category
        self.model = None
        self.use_class_weights = use_class_weights
        self.class_weights = None
        self.training_date = None
        self.label_encoder = None
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train LightGBM for specific attack classification"""
        
        num_classes = len(np.unique(y_train))
        
        print(f"\n[TIER 3-{self.attack_category}] Training on {len(X_train):,} samples")
        print(f"[TIER 3-{self.attack_category}] Classes: {num_classes}")
        
        # Calculate class weights
        if self.use_class_weights:
            unique_classes = np.unique(y_train)
            class_weights = compute_class_weight(
                'balanced',
                classes=unique_classes,
                y=y_train
            )
            self.class_weights = dict(zip(unique_classes, class_weights))
            sample_weights = np.array([self.class_weights[label] for label in y_train])
        else:
            sample_weights = None
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Parameters
        params = {
            'objective': 'multiclass',
            'num_class': num_classes,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 1,
            'n_jobs': -1
        }
        
        # Train
        print(f"\n[TIER 3-{self.attack_category}] Starting training...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=True),
                lgb.log_evaluation(period=50)
            ]
        )
        
        self.training_date = datetime.now()
        print(f"\n[TIER 3-{self.attack_category}] Training complete!")
        
        return self
    
    def predict(self, X):
        """Predict class labels"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        y_pred_proba = self.model.predict(X, num_iteration=self.model.best_iteration)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred
    
    def evaluate(self, X_test, y_test, label_encoder=None):
        """Evaluate on test set"""
        
        y_pred = self.predict(X_test)
        
        if label_encoder:
            class_names = label_encoder.classes_
        else:
            class_names = [f"Class {i}" for i in range(len(np.unique(y_test)))]
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n[TIER 3-{self.attack_category}] Accuracy: {accuracy:.4f}")
        
        print(f"\n[TIER 3-{self.attack_category}] Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': cm
        }
    
    def save(self, filepath):
        """Save the model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)
        
        metadata_path = filepath.replace('.txt', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'attack_category': self.attack_category,
                'training_date': self.training_date,
                'class_weights': self.class_weights
            }, f)
        
        print(f"\n[TIER 3-{self.attack_category}] Model saved to {filepath}")
    
    def load(self, filepath):
        """Load the model"""
        self.model = lgb.Booster(model_file=filepath)
        
        metadata_path = filepath.replace('.txt', '_metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.attack_category = data['attack_category']
                self.training_date = data['training_date']
                self.class_weights = data['class_weights']
        
        print(f"[TIER 3-{self.attack_category}] Model loaded from {filepath}")


def train_ddos_classifier():
    """Train Tier 3 DDoS-specific classifier"""
    
    print("="*60)
    print("IOTGUARD - TIER 3: DDoS SPECIFIC CLASSIFICATION")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = IoTDataPreprocessor(dataset_path="dataset/CSV/MERGED_CSV")
    
    # Load data
    print("\nLoading dataset...")
    df = preprocessor.load_data(sample_size=100000)
    
    # Get only DDoS attacks
    ddos_attacks = preprocessor.class_mapping['DDoS']
    df_ddos = df[df['Label'].isin(ddos_attacks)].copy()
    
    print(f"DDoS samples: {len(df_ddos):,}")
    print(f"DDoS attack types: {df_ddos['Label'].nunique()}")
    
    # Clean and prepare
    df_ddos = preprocessor.clean_data(df_ddos)
    data = preprocessor.prepare_for_training(df_ddos, label_type='34class', test_size=0.2)
    
    # Split validation
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']
    
    val_size = int(len(X_train) * 0.2)
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Print distribution
    print("\nDDoS Attack Distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        attack_name = data['label_encoder'].inverse_transform([cls])[0]
        print(f"  {attack_name}: {count:,}")
    
    # Train
    tier3_ddos = Tier3SpecificClassifier('DDoS', use_class_weights=True)
    tier3_ddos.train(X_train, y_train, X_val, y_val)
    tier3_ddos.label_encoder = data['label_encoder']
    
    # Evaluate
    results = tier3_ddos.evaluate(X_test, y_test, label_encoder=data['label_encoder'])
    
    # Save
    tier3_ddos.save('models/tier3_ddos_specific.txt')
    
    print("\n[TIER 3-DDoS] Training complete!")
    print("="*60)
    
    return tier3_ddos


def train_web_attack_classifier():
    """Train Tier 3 Web attack classifier (binary: attack present or not)"""
    
    print("="*60)
    print("IOTGUARD - TIER 3: WEB ATTACK DETECTION")
    print("="*60)
    
    # Initialize preprocessor
    preprocessor = IoTDataPreprocessor(dataset_path="dataset/CSV/MERGED_CSV")
    
    # Load data
    print("\nLoading dataset...")
    df = preprocessor.load_data(sample_size=100000)
    
    # Get web attacks
    web_attacks = preprocessor.class_mapping['Web']
    df_web = df[df['Label'].isin(web_attacks)].copy()
    
    print(f"Web attack samples: {len(df_web):,}")
    print(f"Web attack types: {df_web['Label'].nunique()}")
    
    # For web attacks (very rare), we might want to keep all specific types
    if len(df_web) > 100:
        df_web = preprocessor.clean_data(df_web)
        data = preprocessor.prepare_for_training(df_web, label_type='34class', test_size=0.2)
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        val_size = int(len(X_train) * 0.2)
        X_val = X_train[:val_size]
        y_val = y_train[:val_size]
        X_train = X_train[val_size:]
        y_train = y_train[val_size:]
        
        print(f"\nTraining set: {X_train.shape}")
        
        # Train
        tier3_web = Tier3SpecificClassifier('Web', use_class_weights=True)
        tier3_web.train(X_train, y_train, X_val, y_val)
        tier3_web.label_encoder = data['label_encoder']
        
        # Evaluate
        tier3_web.evaluate(X_test, y_test, label_encoder=data['label_encoder'])
        
        # Save
        tier3_web.save('models/tier3_web_specific.txt')
        
        print("\n[TIER 3-Web] Training complete!")
    else:
        print("\n[TIER 3-Web] Not enough samples for training, using rule-based detection")
    
    print("="*60)


def main():
    """Train all Tier 3 classifiers"""
    
    # Train DDoS-specific (most important)
    train_ddos_classifier()
    
    # Train Web attack classifier
    train_web_attack_classifier()
    
    # Add more specific classifiers as needed
    # train_dos_classifier()
    # train_mirai_classifier()
    # etc.


if __name__ == "__main__":
    main()
