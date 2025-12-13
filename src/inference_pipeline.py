"""
IoTGuard Inference Pipeline
Integrates all 3 tiers for complete attack detection and classification
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from typing import Dict, Tuple

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import IoTDataPreprocessor
from train_tier1_isolation_forest import Tier1AnomalyDetector
from train_tier2_category_classifier import Tier2CategoryClassifier
from train_tier3_specific_attacks import Tier3SpecificClassifier


class IoTGuardPipeline:
    """
    3-Tier Cascaded Detection System
    
    Tier 1: Isolation Forest (Benign vs Anomaly)
    Tier 2: LightGBM (7-class attack category)
    Tier 3: LightGBM (Specific attack type)
    """
    
    def __init__(self):
        self.tier1 = None
        self.tier2 = None
        self.tier3_ddos = None
        self.tier3_web = None
        self.preprocessor = None
        
        self.tier2_label_encoder = None
        self.tier3_ddos_label_encoder = None
        self.tier3_web_label_encoder = None
    
    def load_models(self, model_dir='models'):
        """Load all trained models"""
        
        print("Loading IoTGuard 3-Tier System...")
        
        # Load Tier 1
        print("\n[TIER 1] Loading Isolation Forest...")
        self.tier1 = Tier1AnomalyDetector()
        tier1_path = os.path.join(model_dir, 'tier1_isolation_forest.pkl')
        if os.path.exists(tier1_path):
            self.tier1.load(tier1_path)
        else:
            print(f"Warning: {tier1_path} not found!")
        
        # Load Tier 1 preprocessor
        preprocessor_path = os.path.join(model_dir, 'tier1_preprocessor.pkl')
        if os.path.exists(preprocessor_path):
            self.preprocessor = IoTDataPreprocessor()
            self.preprocessor.load_preprocessor(preprocessor_path)
        
        # Load Tier 2
        print("\n[TIER 2] Loading 7-Class Classifier...")
        self.tier2 = Tier2CategoryClassifier()
        tier2_path = os.path.join(model_dir, 'tier2_7class_lgb.txt')
        if os.path.exists(tier2_path):
            self.tier2.load(tier2_path)
            
            # Load label encoder
            tier2_preprocessor_path = os.path.join(model_dir, 'tier2_preprocessor.pkl')
            if os.path.exists(tier2_preprocessor_path):
                with open(tier2_preprocessor_path, 'rb') as f:
                    data = pickle.load(f)
                    self.tier2_label_encoder = data.get('label_encoder')
        else:
            print(f"Warning: {tier2_path} not found!")
        
        # Load Tier 3 DDoS
        print("\n[TIER 3] Loading DDoS-Specific Classifier...")
        tier3_ddos_path = os.path.join(model_dir, 'tier3_ddos_specific.txt')
        if os.path.exists(tier3_ddos_path):
            self.tier3_ddos = Tier3SpecificClassifier('DDoS')
            self.tier3_ddos.load(tier3_ddos_path)
        else:
            print(f"Note: {tier3_ddos_path} not found (optional)")
        
        # Load Tier 3 Web
        tier3_web_path = os.path.join(model_dir, 'tier3_web_specific.txt')
        if os.path.exists(tier3_web_path):
            self.tier3_web = Tier3SpecificClassifier('Web')
            self.tier3_web.load(tier3_web_path)
        else:
            print(f"Note: {tier3_web_path} not found (optional)")
        
        print("\nIoTGuard System Loaded Successfully!")
    
    def predict_single(self, X: np.ndarray) -> Dict:
        """
        Predict on a single sample through the cascade
        
        Returns:
            dict with tier1_result, tier2_result, tier3_result, final_label
        """
        result = {
            'tier1_result': None,
            'tier1_score': None,
            'tier2_result': None,
            'tier2_confidence': None,
            'tier3_result': None,
            'final_label': 'BENIGN',
            'confidence': 1.0
        }
        
        # TIER 1: Anomaly Detection
        tier1_pred = self.tier1.predict(X.reshape(1, -1))[0]
        tier1_score = self.tier1.decision_function(X.reshape(1, -1))[0]
        
        result['tier1_result'] = 'NORMAL' if tier1_pred == 1 else 'ANOMALY'
        result['tier1_score'] = tier1_score
        
        if tier1_pred == 1:  # Normal traffic
            result['final_label'] = 'BENIGN'
            return result
        
        # TIER 2: Attack Category Classification
        tier2_proba = self.tier2.predict_proba(X.reshape(1, -1))[0]
        tier2_class = np.argmax(tier2_proba)
        tier2_confidence = tier2_proba[tier2_class]
        
        if self.tier2_label_encoder:
            tier2_category = self.tier2_label_encoder.inverse_transform([tier2_class])[0]
        else:
            tier2_category = f"Category_{tier2_class}"
        
        result['tier2_result'] = tier2_category
        result['tier2_confidence'] = tier2_confidence
        
        # TIER 3: Specific Attack Type (if available)
        if tier2_category == 'DDoS' and self.tier3_ddos:
            tier3_pred = self.tier3_ddos.predict(X.reshape(1, -1))[0]
            if self.tier3_ddos.label_encoder:
                specific_attack = self.tier3_ddos.label_encoder.inverse_transform([tier3_pred])[0]
                result['tier3_result'] = specific_attack
                result['final_label'] = specific_attack
            else:
                result['final_label'] = tier2_category
        
        elif tier2_category == 'Web' and self.tier3_web:
            tier3_pred = self.tier3_web.predict(X.reshape(1, -1))[0]
            if self.tier3_web.label_encoder:
                specific_attack = self.tier3_web.label_encoder.inverse_transform([tier3_pred])[0]
                result['tier3_result'] = specific_attack
                result['final_label'] = specific_attack
            else:
                result['final_label'] = tier2_category
        else:
            # No Tier 3 for this category, use Tier 2 result
            result['final_label'] = tier2_category
        
        result['confidence'] = tier2_confidence
        
        return result
    
    def predict_batch(self, X: np.ndarray) -> pd.DataFrame:
        """Predict on batch of samples"""
        
        results = []
        for i in range(len(X)):
            result = self.predict_single(X[i])
            results.append(result)
        
        return pd.DataFrame(results)
    
    def evaluate_pipeline(self, X_test, y_test_true):
        """Evaluate the full pipeline"""
        
        print("\n" + "="*60)
        print("IOTGUARD PIPELINE EVALUATION")
        print("="*60)
        
        predictions = self.predict_batch(X_test)
        
        # Calculate accuracy
        correct = (predictions['final_label'] == y_test_true).sum()
        accuracy = correct / len(y_test_true)
        
        print(f"\nOverall Accuracy: {accuracy:.4f} ({correct}/{len(y_test_true)})")
        
        # Tier 1 stats
        tier1_anomalies = (predictions['tier1_result'] == 'ANOMALY').sum()
        print(f"\nTIER 1 Statistics:")
        print(f"  Detected Anomalies: {tier1_anomalies} ({tier1_anomalies/len(predictions)*100:.2f}%)")
        
        # Tier 2 stats
        print(f"\nTIER 2 Category Distribution:")
        print(predictions['tier2_result'].value_counts())
        
        # Tier 3 stats
        if predictions['tier3_result'].notna().any():
            print(f"\nTIER 3 Specific Attack Distribution:")
            print(predictions['tier3_result'].value_counts())
        
        return predictions


def demo_inference():
    """Demo: Load and run inference on test data"""
    
    print("="*60)
    print("IOTGUARD - INFERENCE DEMO")
    print("="*60)
    
    # Initialize pipeline
    pipeline = IoTGuardPipeline()
    pipeline.load_models('models')
    
    # Load some test data
    print("\nLoading test data...")
    preprocessor = IoTDataPreprocessor(dataset_path="dataset/CSV/MERGED_CSV")
    df_test = preprocessor.load_data(sample_size=1000)
    df_test = preprocessor.clean_data(df_test)
    
    X_test = df_test.drop('Label', axis=1)
    y_test = df_test['Label']
    
    # Scale features
    X_test_scaled = pipeline.preprocessor.scaler.transform(X_test)
    
    # Run pipeline
    print("\nRunning inference on 1000 samples...")
    results = pipeline.evaluate_pipeline(X_test_scaled, y_test)
    
    # Show some examples
    print("\nSample Predictions:")
    print(results[['tier1_result', 'tier2_result', 'tier3_result', 'final_label', 'confidence']].head(10))
    
    print("\n" + "="*60)


if __name__ == "__main__":
    demo_inference()
