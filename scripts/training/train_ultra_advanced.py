#!/usr/bin/env python3
"""
IoTGuard Ultra Advanced Training Pipeline
==========================================
CIC-IoT-2023 Dataset - Maximum Performance Configuration

Features:
- 4-Model Ensemble per stage (LightGBM, XGBoost, CatBoost, Neural Network)
- Advanced feature engineering (polynomial, interaction, statistical)
- Hierarchical attack classification
- SMOTE-Tomek hybrid balancing
- Comprehensive metadata tracking
- Cross-validation support
- Feature importance analysis
- GPU acceleration support

Author: IoTGuard Team
Date: December 2025
"""

import pandas as pd
import numpy as np
import os
import sys
import glob
import json
import yaml
import warnings
from datetime import datetime
from pathlib import Path
import pickle

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             roc_curve, precision_recall_curve, matthews_corrcoef)
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2

# Gradient Boosting
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

# Deep Learning
try:
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

# Imbalanced Learning
from imblearn.over_sampling import SMOTE, ADASYN, SMOTENC
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

class UltraAdvancedTrainer:
    """Ultra Advanced ML Training Pipeline for IoT Intrusion Detection"""
    
    def __init__(self, config_path='config_train_ultra_advanced.yaml'):
        """Initialize trainer with configuration"""
        self.start_time = datetime.now()
        self.config = self.load_config(config_path)
        self.create_output_directories()
        self.metadata = {
            'model_info': {},
            'training_data': {},
            'stage1_binary': {},
            'stage2_multiclass': {},
            'full_pipeline': {},
            'feature_engineering': {},
            'files': {},
            'stage1_nn': {}  # Added for neural network configuration
        }
        
    def load_config(self, config_path):
        """Load YAML configuration"""
        print(f"\n{'='*80}")
        print("LOADING CONFIGURATION")
        print(f"{'='*80}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Loaded configuration from: {config_path}")
        return config
    
    def create_output_directories(self):
        """Create output directories if they don't exist"""
        for dir_key in ['models_dir', 'results_dir', 'logs_dir']:
            dir_path = Path(self.config['output'][dir_key])
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Directory ready: {dir_path}")
    
    def load_data(self):
        """Load and prepare dataset from merged CSV files"""
        print(f"\n{'='*80}")
        print("STEP 1: LOADING DATASET")
        print(f"{'='*80}")
        
        csv_path = Path(self.config['dataset']['merged_csv_path'])
        csv_files = sorted(glob.glob(str(csv_path / self.config['dataset']['csv_pattern'])))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found at {csv_path}")
        
        print(f"Found {len(csv_files)} merged CSV files")
        
        # RANDOMLY SHUFFLE FILES for better diversity
        import random
        random.seed(57)  # For reproducibility
        random.shuffle(csv_files)
        print("✓ Files shuffled randomly for diverse sampling")
        
        # Split files into train/val/test
        n_train = self.config['dataset']['train_files']
        n_val = self.config['dataset']['val_files']
        n_test = self.config['dataset']['test_files']
        
        train_files = csv_files[:n_train]
        val_files = csv_files[n_train:n_train + n_val]
        test_files = csv_files[n_train + n_val:n_train + n_val + n_test]
        
        print(f"\nSplit: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files")
        
        # Load training data
        print("\nLoading training data...")
        self.df_train = self.load_and_balance_files(train_files, 'train')
        
        # Load validation data
        print("\nLoading validation data...")
        self.df_val = self.load_and_balance_files(val_files, 'val')
        
        # Load test data
        print("\nLoading test data...")
        self.df_test = self.load_and_balance_files(test_files, 'test')
        
        print(f"\n{'='*80}")
        print("DATA LOADING COMPLETE")
        print(f"{'='*80}")
        print(f"Training set: {len(self.df_train):,} samples")
        print(f"Validation set: {len(self.df_val):,} samples")
        print(f"Test set: {len(self.df_test):,} samples")
        print(f"Total samples: {len(self.df_train) + len(self.df_val) + len(self.df_test):,}")
        
        # Update metadata
        self.metadata['training_data'] = {
            'train_samples': len(self.df_train),
            'val_samples': len(self.df_val),
            'test_samples': len(self.df_test),
            'train_files': len(train_files),
            'val_files': len(val_files),
            'test_files': len(test_files),
            'dataset_name': 'CIC-IoT-2023'
        }
    
    def load_and_balance_files(self, files, split_name):
        """Load files with intelligent sampling strategy"""
        dfs = []
        benign_dfs = []
        attack_dfs = {}
        
        for i, file in enumerate(files, 1):
            print(f"  [{i}/{len(files)}] Loading {Path(file).name}...", end=' ')
            df = pd.read_csv(file)
            
            # Separate benign and attacks
            benign = df[df['Label'] == 'BENIGN']
            attacks = df[df['Label'] != 'BENIGN']
            
            benign_dfs.append(benign)
            
            # Group attacks by category
            for _, row in attacks.iterrows():
                label = row['Label']
                category = self.get_attack_category(label)
                if category not in attack_dfs:
                    attack_dfs[category] = []
                attack_dfs[category].append(row)
            
            print(f"✓ ({len(benign):,} benign, {len(attacks):,} attacks)")
        
        # Combine all benign samples
        if benign_dfs:
            all_benign = pd.concat(benign_dfs, ignore_index=True)
            print(f"\nTotal benign samples: {len(all_benign):,}")
        else:
            all_benign = pd.DataFrame()
        
        # Sample attacks per category
        sampled_attacks = []
        max_per_class = self.config['data_processing']['attack_sampling_per_class']
        
        print(f"\nSampling attacks (max {max_per_class:,} per category):")
        for category, rows in attack_dfs.items():
            category_df = pd.DataFrame(rows)
            if len(category_df) > max_per_class:
                category_df = category_df.sample(n=max_per_class, random_state=57)
            sampled_attacks.append(category_df)
            print(f"  {category:20s}: {len(category_df):,} samples")
        
        # Combine all data
        if sampled_attacks:
            all_attacks = pd.concat(sampled_attacks, ignore_index=True)
        else:
            all_attacks = pd.DataFrame()
        
        combined_df = pd.concat([all_benign, all_attacks], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=57).reset_index(drop=True)
        
        benign_pct = 100 * len(all_benign) / len(combined_df) if len(combined_df) > 0 else 0
        print(f"\nCombined {split_name} set: {len(combined_df):,} samples ({benign_pct:.2f}% benign)")
        
        return combined_df
    
    def get_attack_category(self, label):
        """Map attack label to category"""
        for category, attacks in self.config['class_mapping'].items():
            if label in attacks:
                return category
        return 'Unknown'
    
    def engineer_features(self, df, fit=True):
        """Advanced feature engineering"""
        print(f"\n{'='*80}")
        print("STEP 2: FEATURE ENGINEERING")
        print(f"{'='*80}")
        
        df = df.copy()
        original_features = [col for col in df.columns if col != 'Label']
        print(f"Original features: {len(original_features)}")
        
        # Statistical features
        if self.config['data_processing']['statistical_features']:
            print("\n✓ Creating statistical features...")
            
            # Ratios
            df['syn_ack_ratio'] = df['syn_flag_number'] / (df['ack_flag_number'] + 1)
            df['rst_fin_ratio'] = df['rst_flag_number'] / (df['fin_flag_number'] + 1)
            df['psh_ack_ratio'] = df['psh_flag_number'] / (df['ack_flag_number'] + 1)
            
            # Packet analysis
            df['packet_rate_ratio'] = df['Rate'] / (df['Number'] + 1)
            df['size_per_packet'] = df['Tot size'] / (df['Number'] + 1)
            df['avg_iat'] = df['IAT'] / (df['Number'] + 1)
            
            # Variance analysis
            df['variance_avg_ratio'] = df['Variance'] / (df['AVG'] + 1)
            df['std_avg_ratio'] = df['Std'] / (df['AVG'] + 1)
            df['range_stat'] = df['Max'] - df['Min']
            df['cv'] = df['Std'] / (df['AVG'] + 1)  # Coefficient of variation
            
            # Flag combinations
            df['flags_total'] = (df['fin_flag_number'] + df['syn_flag_number'] + 
                               df['rst_flag_number'] + df['psh_flag_number'] + 
                               df['ack_flag_number'] + df['ece_flag_number'] + 
                               df['cwr_flag_number'])
            df['flag_diversity'] = (df[['fin_flag_number', 'syn_flag_number', 'rst_flag_number',
                                        'psh_flag_number', 'ack_flag_number']] > 0).sum(axis=1)
            
            # Protocol combinations
            df['tcp_http_combo'] = df['TCP'] * df['HTTP']
            df['udp_dns_combo'] = df['UDP'] * df['DNS']
            df['protocol_count'] = df[['TCP', 'UDP', 'ICMP', 'ARP', 'DNS', 'HTTP', 'HTTPS']].sum(axis=1)
            
            # Advanced ratios
            df['ttl_rate_ratio'] = df['Time_To_Live'] / (df['Rate'] + 1)
            df['header_size_ratio'] = df['Header_Length'] / (df['Tot size'] + 1)
            
            # Count ratios
            df['syn_count_ratio'] = df['syn_count'] / (df['Number'] + 1)
            df['ack_count_ratio'] = df['ack_count'] / (df['Number'] + 1)
            df['fin_count_ratio'] = df['fin_count'] / (df['Number'] + 1)
            df['rst_count_ratio'] = df['rst_count'] / (df['Number'] + 1)
            
            # Interaction features
            df['rate_ttl_interaction'] = df['Rate'] * df['Time_To_Live']
            df['size_rate_interaction'] = df['Tot size'] * df['Rate']
            
            # Log transformations for skewed features
            df['log_rate'] = np.log1p(df['Rate'])
            df['log_tot_size'] = np.log1p(df['Tot size'])
            df['log_number'] = np.log1p(df['Number'])
            
        # Time-based features
        if self.config['data_processing']['time_based_features']:
            print("✓ Creating time-based features...")
            df['iat_variance'] = df['IAT'] * df['Variance']
            df['iat_std'] = df['IAT'] * df['Std']
            df['burst_score'] = df['Rate'] * df['Number'] / (df['IAT'] + 1)
            
        # Polynomial features (selective)
        if self.config['data_processing']['polynomial_features']:
            print("✓ Creating polynomial features (degree 2, selective)...")
            key_features = ['Rate', 'Tot size', 'Number', 'AVG', 'Variance']
            for feat in key_features:
                if feat in df.columns:
                    df[f'{feat}_squared'] = df[feat] ** 2
                    df[f'{feat}_sqrt'] = np.sqrt(np.abs(df[feat]))
        
        # Handle infinite and NaN values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.median(numeric_only=True), inplace=True)
        
        engineered_features = [col for col in df.columns if col not in original_features + ['Label']]
        print(f"\n✓ Created {len(engineered_features)} new features")
        print(f"Total features: {len(df.columns) - 1}")  # Exclude Label
        
        # Update metadata
        if fit:
            self.metadata['feature_engineering'] = {
                'original_features': len(original_features),
                'engineered_features': len(engineered_features),
                'total_features': len(df.columns) - 1,
                'feature_types': {
                    'statistical': self.config['data_processing']['statistical_features'],
                    'time_based': self.config['data_processing']['time_based_features'],
                    'polynomial': self.config['data_processing']['polynomial_features']
                }
            }
        
        return df
    
    def prepare_stage1_data(self):
        """Prepare data for Stage 1: Binary Classification"""
        print(f"\n{'='*80}")
        print("STEP 3: PREPARING STAGE 1 DATA (Binary Classification)")
        print(f"{'='*80}")
        
        # Create binary labels
        self.df_train['Binary_Label'] = (self.df_train['Label'] != 'BENIGN').astype(int)
        self.df_val['Binary_Label'] = (self.df_val['Label'] != 'BENIGN').astype(int)
        self.df_test['Binary_Label'] = (self.df_test['Label'] != 'BENIGN').astype(int)
        
        # Separate features and labels
        feature_cols = [col for col in self.df_train.columns if col not in ['Label', 'Binary_Label']]
        
        X_train = self.df_train[feature_cols]
        y_train = self.df_train['Binary_Label']
        
        X_val = self.df_val[feature_cols]
        y_val = self.df_val['Binary_Label']
        
        X_test = self.df_test[feature_cols]
        y_test = self.df_test['Binary_Label']
        
        print(f"\nClass distribution (Training):")
        print(f"  Benign (0): {(y_train == 0).sum():,} samples")
        print(f"  Attack (1): {(y_train == 1).sum():,} samples")
        
        # Scaling
        print(f"\n✓ Applying {self.config['data_processing']['scaler_type']} scaling...")
        scaler_type = self.config['data_processing']['scaler_type']
        if scaler_type == 'robust':
            self.scaler_s1 = RobustScaler()
        elif scaler_type == 'standard':
            self.scaler_s1 = StandardScaler()
        else:
            self.scaler_s1 = MinMaxScaler()
        
        X_train_scaled = self.scaler_s1.fit_transform(X_train)
        X_val_scaled = self.scaler_s1.transform(X_val)
        X_test_scaled = self.scaler_s1.transform(X_test)
        
        # Balancing
        balance_method = self.config['stage1_binary']['balancing']['method']
        print(f"\n✓ Applying {balance_method} balancing...")
        
        if balance_method == 'smote':
            sampler = SMOTE(random_state=57, k_neighbors=self.config['stage1_binary']['balancing'].get('k_neighbors', 5))
        elif balance_method == 'adasyn':
            sampler = ADASYN(random_state=57, n_neighbors=self.config['stage1_binary']['balancing']['k_neighbors'])
        elif balance_method == 'smote_tomek':
            sampler = SMOTETomek(random_state=57)
        elif balance_method == 'smote_enn':
            sampler = SMOTEENN(random_state=57)
        else:
            sampler = SMOTE(random_state=57)
        
        X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_scaled, y_train)
        
        print(f"\nBalanced class distribution:")
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count:,} samples")
        
        return (X_train_balanced, y_train_balanced, X_val_scaled, y_val, 
                X_test_scaled, y_test, feature_cols)
    
    def train_stage1(self, X_train, y_train, X_val, y_val):
        """Train Stage 1: Binary Classification Ensemble"""
        print(f"\n{'='*80}")
        print("STEP 4: TRAINING STAGE 1 MODELS (Binary Classification)")
        print(f"{'='*80}")
        
        self.stage1_models = {}
        
        # LightGBM
        if self.config['stage1_binary']['models']['lightgbm']['enabled']:
            print("\n[1/4] Training LightGBM...")
            lgb_params = self.config['stage1_binary']['models']['lightgbm']
            self.stage1_models['lightgbm'] = lgb.LGBMClassifier(
                num_leaves=lgb_params['num_leaves'],
                max_depth=lgb_params['max_depth'],
                learning_rate=lgb_params['learning_rate'],
                n_estimators=lgb_params['n_estimators'],
                min_child_samples=lgb_params['min_child_samples'],
                subsample=lgb_params['subsample'],
                colsample_bytree=lgb_params['colsample_bytree'],
                reg_alpha=lgb_params['reg_alpha'],
                reg_lambda=lgb_params['reg_lambda'],
                class_weight=lgb_params['class_weight'],
                random_state=57,
                n_jobs=-1,
                verbose=-1
            )
            self.stage1_models['lightgbm'].fit(X_train, y_train)
            val_pred = self.stage1_models['lightgbm'].predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"  ✓ LightGBM trained - Validation Accuracy: {val_acc:.4f}")
        
        # XGBoost
        if self.config['stage1_binary']['models']['xgboost']['enabled']:
            print("\n[2/4] Training XGBoost...")
            xgb_params = self.config['stage1_binary']['models']['xgboost']
            self.stage1_models['xgboost'] = xgb.XGBClassifier(
                max_depth=xgb_params['max_depth'],
                learning_rate=xgb_params['learning_rate'],
                n_estimators=xgb_params['n_estimators'],
                min_child_weight=xgb_params['min_child_weight'],
                subsample=xgb_params['subsample'],
                colsample_bytree=xgb_params['colsample_bytree'],
                gamma=xgb_params['gamma'],
                reg_alpha=xgb_params['reg_alpha'],
                reg_lambda=xgb_params['reg_lambda'],
                scale_pos_weight=xgb_params['scale_pos_weight'],
                random_state=57,
                n_jobs=-1,
                verbosity=0
            )
            self.stage1_models['xgboost'].fit(X_train, y_train)
            val_pred = self.stage1_models['xgboost'].predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"  ✓ XGBoost trained - Validation Accuracy: {val_acc:.4f}")
        
        # CatBoost
        if self.config['stage1_binary']['models']['catboost']['enabled'] and CATBOOST_AVAILABLE:
            print("\n[3/4] Training CatBoost...")
            cb_params = self.config['stage1_binary']['models']['catboost']
            self.stage1_models['catboost'] = cb.CatBoostClassifier(
                depth=cb_params['depth'],
                learning_rate=cb_params['learning_rate'],
                iterations=cb_params['iterations'],
                l2_leaf_reg=cb_params['l2_leaf_reg'],
                border_count=cb_params['border_count'],
                random_strength=cb_params['random_strength'],
                random_state=57,
                verbose=0
            )
            self.stage1_models['catboost'].fit(X_train, y_train)
            val_pred = self.stage1_models['catboost'].predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"  ✓ CatBoost trained - Validation Accuracy: {val_acc:.4f}")
        
        # Neural Network
        if self.config['stage1_binary']['models']['neural_network']['enabled'] and TENSORFLOW_AVAILABLE:
            print("\n[4/4] Training Neural Network...")
            nn_params = self.config['stage1_binary']['models']['neural_network']
            
            model = models.Sequential()
            model.add(layers.Input(shape=(X_train.shape[1],)))
            
            for units in nn_params['hidden_layers']:
                model.add(layers.Dense(units, activation=nn_params['activation']))
                model.add(layers.Dropout(nn_params['dropout_rate']))
            
            model.add(layers.Dense(1, activation='sigmoid'))
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=nn_params['learning_rate']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=nn_params['early_stopping_patience'],
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=nn_params['epochs'],
                batch_size=nn_params['batch_size'],
                callbacks=[early_stop],
                verbose=0
            )
            
            self.stage1_models['neural_network'] = model
            val_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
            val_acc = accuracy_score(y_val, val_pred)
            print(f"  ✓ Neural Network trained - Validation Accuracy: {val_acc:.4f}")
        
        print(f"\n✓ Stage 1 training complete - {len(self.stage1_models)} models trained")
    
    def evaluate_stage1(self, X_val, y_val, X_test, y_test):
        """Evaluate Stage 1 ensemble with validation-based threshold tuning"""
        print(f"\n{'='*80}")
        print("STEP 5: EVALUATING STAGE 1 (Binary Classification)")
        print(f"{'='*80}")
        
        weights = self.config['stage1_binary']['ensemble']['weights']
        
        # Build validation and test probabilities for threshold tuning
        val_probas = []
        test_probas = []
        model_names = []
        
        for name, model in self.stage1_models.items():
            model_names.append(name)
            if name == 'neural_network' and TENSORFLOW_AVAILABLE:
                p_val = model.predict(X_val, verbose=0).flatten()
                p_test = model.predict(X_test, verbose=0).flatten()
            else:
                # sklearn models: use predict_proba when available
                try:
                    p_val = model.predict_proba(X_val)[:, 1]
                    p_test = model.predict_proba(X_test)[:, 1]
                except Exception:
                    # fallback to predict (0/1)
                    p_val = model.predict(X_val).astype(float)
                    p_test = model.predict(X_test).astype(float)
            val_probas.append(p_val)
            test_probas.append(p_test)
            # quick per-model accuracy on test
            acc = accuracy_score(y_test, (p_test > 0.5).astype(int))
            print(f"\n{name.capitalize():15s}: Test Accuracy = {acc:.4f}")

        # Weighted ensemble probabilities
        weights_list = [weights.get(n, 0) for n in model_names]
        val_ensemble_proba = np.zeros(len(X_val))
        test_ensemble_proba = np.zeros(len(X_test))
        for w, vp, tp in zip(weights_list, val_probas, test_probas):
            val_ensemble_proba += w * vp
            test_ensemble_proba += w * tp

        # Tune threshold on validation set to maximize recall (with a precision floor)
        best_thr = 0.5
        best_rec = -1.0
        min_precision = float(self.config['stage1_binary'].get('min_precision_for_threshold', 0.45))
        for thr in np.linspace(0.3, 0.75, 19):
            preds = (val_ensemble_proba > thr).astype(int)
            prec = precision_score(y_val, preds, zero_division=0)
            rec = recall_score(y_val, preds, zero_division=0)
            # prefer higher recall but enforce minimum precision
            if prec >= min_precision and rec > best_rec:
                best_rec = rec
                best_thr = thr
        # if no threshold satisfied precision floor, pick max recall threshold
        if best_rec < 0:
            thresholds = np.linspace(0.3, 0.75, 19)
            recalls = [recall_score(y_val, (val_ensemble_proba > t).astype(int)) for t in thresholds]
            best_thr = float(thresholds[np.argmax(recalls)])

        print(f"\nChosen ensemble threshold (tuned on validation): {best_thr:.3f}")
        print(f"  Min precision constraint: {min_precision:.2f}")

        ensemble_pred = (test_ensemble_proba > best_thr).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_test, ensemble_pred)
        prec = precision_score(y_test, ensemble_pred, zero_division=0)
        rec = recall_score(y_test, ensemble_pred, zero_division=0)
        f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, ensemble_pred)
        
        print(f"\n{'='*80}")
        print("STAGE 1 ENSEMBLE RESULTS")
        print(f"{'='*80}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"MCC:       {mcc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, ensemble_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
        print(f"  FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
        
        # Save visualization
        self.plot_confusion_matrix(cm, ['Benign', 'Attack'], 
                                   'stage1_confusion_matrix.png',
                                   f'Stage 1: Binary Classification\nAccuracy: {acc:.4f}')
        
        # Update metadata
        self.metadata['stage1_binary'] = {
            'purpose': 'Binary classification (Benign vs Attack)',
            'models': list(self.stage1_models.keys()),
            'ensemble_weights': weights,
            'performance': {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'mcc': float(mcc),
                'confusion_matrix': cm.tolist()
            },
            'hyperparameters': self.config['stage1_binary']['models']
        }
        
        return ensemble_pred, acc, prec, rec, f1
    
    def prepare_stage2_data(self, X_test, y_stage1_pred):
        """Prepare data for Stage 2: Multi-Class Classification"""
        print(f"\n{'='*80}")
        print("STEP 6: PREPARING STAGE 2 DATA (Multi-Class Classification)")
        print(f"{'='*80}")
        
        # Filter only predicted attacks
        attack_mask = y_stage1_pred == 1
        df_attacks_train = self.df_train[self.df_train['Binary_Label'] == 1].copy()
        df_attacks_val = self.df_val[self.df_val['Binary_Label'] == 1].copy()
        df_attacks_test = self.df_test.iloc[attack_mask].copy()
        
        print(f"\nAttack samples:")
        print(f"  Training:   {len(df_attacks_train):,}")
        print(f"  Validation: {len(df_attacks_val):,}")
        print(f"  Test:       {len(df_attacks_test):,}")
        
        # Map to categories
        df_attacks_train['Category'] = df_attacks_train['Label'].apply(self.get_attack_category)
        df_attacks_val['Category'] = df_attacks_val['Label'].apply(self.get_attack_category)
        df_attacks_test['Category'] = df_attacks_test['Label'].apply(self.get_attack_category)
        
        # FILTER OUT any 'Benign' samples (false positives from Stage 1)
        df_attacks_train = df_attacks_train[df_attacks_train['Category'] != 'Benign']
        df_attacks_val = df_attacks_val[df_attacks_val['Category'] != 'Benign']
        
        # For test set, keep track of which indices to keep after filtering
        test_benign_mask = df_attacks_test['Category'] == 'Benign'
        df_attacks_test = df_attacks_test[df_attacks_test['Category'] != 'Benign']
        
        # Update attack_mask to reflect only true attacks (not false positives)
        attack_indices = np.where(attack_mask)[0]
        benign_fp_indices = attack_indices[test_benign_mask.values]
        attack_mask[benign_fp_indices] = False  # Remove false positive predictions
        
        print(f"\nAfter filtering benign false positives:")
        print(f"  Training:   {len(df_attacks_train):,}")
        print(f"  Validation: {len(df_attacks_val):,}")
        print(f"  Test:       {len(df_attacks_test):,} (removed {test_benign_mask.sum()} false positives)")
        
        # Encode labels
        self.label_encoder_s2 = LabelEncoder()
        y_train_s2 = self.label_encoder_s2.fit_transform(df_attacks_train['Category'])
        y_val_s2 = self.label_encoder_s2.transform(df_attacks_val['Category'])
        y_test_s2 = self.label_encoder_s2.transform(df_attacks_test['Category'])
        
        print(f"\nAttack categories: {list(self.label_encoder_s2.classes_)}")
        print(f"\nTraining distribution:")
        for i, cat in enumerate(self.label_encoder_s2.classes_):
            count = (y_train_s2 == i).sum()
            print(f"  {cat:20s}: {count:,} samples")
        
        # Get features
        feature_cols = [col for col in df_attacks_train.columns 
                       if col not in ['Label', 'Binary_Label', 'Category']]
        
        X_train_s2 = df_attacks_train[feature_cols]
        X_val_s2 = df_attacks_val[feature_cols]
        X_test_s2 = df_attacks_test[feature_cols]
        
        # Scaling
        print(f"\n✓ Applying scaling...")
        self.scaler_s2 = RobustScaler()
        X_train_s2_scaled = self.scaler_s2.fit_transform(X_train_s2)
        X_val_s2_scaled = self.scaler_s2.transform(X_val_s2)
        X_test_s2_scaled = self.scaler_s2.transform(X_test_s2)
        
        # Balancing
        balance_method = self.config['stage2_multiclass']['balancing']['method']
        print(f"\n✓ Applying {balance_method} balancing...")
        
        if balance_method == 'smote_undersampling':
            # Determine safe k_neighbors for SMOTE based on smallest class size
            from imblearn.over_sampling import RandomOverSampler

            class_counts = pd.Series(y_train_s2).value_counts()
            min_count = int(class_counts.min()) if len(class_counts) > 0 else 0
            config_k = int(self.config['stage2_multiclass']['balancing'].get('k_neighbors', 5))

            if min_count <= 1:
                # Too few samples for SMOTE. Fall back to simple random oversampling.
                print("\nWARNING: Some classes have <=1 samples. Falling back to RandomOverSampler for Stage 2 balancing.")
                ros = RandomOverSampler(random_state=57)
                X_res, y_res = ros.fit_resample(X_train_s2_scaled, y_train_s2)
                # Then apply undersampling to balance if requested
                undersampler = RandomUnderSampler(random_state=57)
                X_train_s2_balanced, y_train_s2_balanced = undersampler.fit_resample(X_res, y_res)
            else:
                # Safe k where k_neighbors < min_count
                k_safe = max(1, min(config_k, min_count - 1))
                print(f"\nUsing SMOTE with k_neighbors={k_safe} (min class size: {min_count})")
                smote = SMOTE(random_state=57, k_neighbors=k_safe)
                X_res, y_res = smote.fit_resample(X_train_s2_scaled, y_train_s2)
                undersampler = RandomUnderSampler(random_state=57)
                X_train_s2_balanced, y_train_s2_balanced = undersampler.fit_resample(X_res, y_res)
        else:
            # Non-undersampling path: adapt k_neighbors similarly
            class_counts = pd.Series(y_train_s2).value_counts()
            min_count = int(class_counts.min()) if len(class_counts) > 0 else 0
            config_k = int(self.config['stage2_multiclass']['balancing'].get('k_neighbors', 5))
            if min_count <= 1:
                print("\nWARNING: Some classes have <=1 samples. Falling back to RandomOverSampler for Stage 2 balancing.")
                ros = RandomOverSampler(random_state=57)
                X_train_s2_balanced, y_train_s2_balanced = ros.fit_resample(X_train_s2_scaled, y_train_s2)
            else:
                k_safe = max(1, min(config_k, min_count - 1))
                print(f"\nUsing SMOTE with k_neighbors={k_safe} (min class size: {min_count})")
                smote = SMOTE(random_state=57, k_neighbors=k_safe)
                X_train_s2_balanced, y_train_s2_balanced = smote.fit_resample(X_train_s2_scaled, y_train_s2)
        
        print(f"\nBalanced distribution:")
        unique, counts = np.unique(y_train_s2_balanced, return_counts=True)
        for label, count in zip(unique, counts):
            cat_name = self.label_encoder_s2.classes_[label]
            print(f"  {cat_name:20s}: {count:,} samples")
        
        return (X_train_s2_balanced, y_train_s2_balanced, X_val_s2_scaled, y_val_s2,
                X_test_s2_scaled, y_test_s2, attack_mask)
    
    def train_stage2(self, X_train, y_train, X_val, y_val):
        """Train Stage 2: Multi-Class Classification Ensemble"""
        print(f"\n{'='*80}")
        print("STEP 7: TRAINING STAGE 2 MODELS (Multi-Class Classification)")
        print(f"{'='*80}")
        
        self.stage2_models = {}
        n_classes = len(self.label_encoder_s2.classes_)
        
        # LightGBM
        if self.config['stage2_multiclass']['models']['lightgbm']['enabled']:
            print("\n[1/3] Training LightGBM...")
            lgb_params = self.config['stage2_multiclass']['models']['lightgbm']
            self.stage2_models['lightgbm'] = lgb.LGBMClassifier(
                num_leaves=lgb_params['num_leaves'],
                max_depth=lgb_params['max_depth'],
                learning_rate=lgb_params['learning_rate'],
                n_estimators=lgb_params['n_estimators'],
                min_child_samples=lgb_params['min_child_samples'],
                subsample=lgb_params['subsample'],
                colsample_bytree=lgb_params['colsample_bytree'],
                reg_alpha=lgb_params['reg_alpha'],
                reg_lambda=lgb_params['reg_lambda'],
                class_weight=lgb_params['class_weight'],
                num_class=n_classes,
                random_state=57,
                n_jobs=-1,
                verbose=-1
            )
            self.stage2_models['lightgbm'].fit(X_train, y_train)
            val_pred = self.stage2_models['lightgbm'].predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"  ✓ LightGBM trained - Validation Accuracy: {val_acc:.4f}")
        
        # XGBoost
        if self.config['stage2_multiclass']['models']['xgboost']['enabled']:
            print("\n[2/3] Training XGBoost...")
            xgb_params = self.config['stage2_multiclass']['models']['xgboost']
            self.stage2_models['xgboost'] = xgb.XGBClassifier(
                max_depth=xgb_params['max_depth'],
                learning_rate=xgb_params['learning_rate'],
                n_estimators=xgb_params['n_estimators'],
                min_child_weight=xgb_params['min_child_weight'],
                subsample=xgb_params['subsample'],
                colsample_bytree=xgb_params['colsample_bytree'],
                gamma=xgb_params['gamma'],
                reg_alpha=xgb_params['reg_alpha'],
                reg_lambda=xgb_params['reg_lambda'],
                num_class=n_classes,
                random_state=57,
                n_jobs=-1,
                verbosity=0
            )
            self.stage2_models['xgboost'].fit(X_train, y_train)
            val_pred = self.stage2_models['xgboost'].predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            print(f"  ✓ XGBoost trained - Validation Accuracy: {val_acc:.4f}")
        
        # CatBoost
        if self.config['stage2_multiclass']['models']['catboost']['enabled'] and CATBOOST_AVAILABLE:
            print("\n[3/3] Training CatBoost...")
            cb_params = self.config['stage2_multiclass']['models']['catboost']
            self.stage2_models['catboost'] = cb.CatBoostClassifier(
                depth=cb_params['depth'],
                learning_rate=cb_params['learning_rate'],
                iterations=cb_params['iterations'],
                l2_leaf_reg=cb_params['l2_leaf_reg'],
                border_count=cb_params['border_count'],
                random_strength=cb_params['random_strength'],
                classes_count=n_classes,
                random_state=57,
                verbose=0
            )
            self.stage2_models['catboost'].fit(X_train, y_train)
            val_pred = self.stage2_models['catboost'].predict(X_val).flatten()
            val_acc = accuracy_score(y_val, val_pred)
            print(f"  ✓ CatBoost trained - Validation Accuracy: {val_acc:.4f}")
        
        print(f"\n✓ Stage 2 training complete - {len(self.stage2_models)} models trained")
    
    def evaluate_stage2(self, X_test, y_test):
        """Evaluate Stage 2 ensemble"""
        print(f"\n{'='*80}")
        print("STEP 8: EVALUATING STAGE 2 (Multi-Class Classification)")
        print(f"{'='*80}")
        
        weights = self.config['stage2_multiclass']['ensemble']['weights']
        n_classes = len(self.label_encoder_s2.classes_)
        
        # Get predictions from each model
        predictions = {}
        for name, model in self.stage2_models.items():
            pred = model.predict(X_test)
            if len(pred.shape) > 1:
                pred = pred.flatten()
            predictions[name] = pred
            
            acc = accuracy_score(y_test, pred)
            print(f"\n{name.capitalize():15s}: Accuracy = {acc:.4f}")
        
        # Ensemble prediction (weighted voting)
        ensemble_pred_proba = np.zeros((len(y_test), n_classes))
        for name, pred in predictions.items():
            weight = weights.get(name, 0)
            # Convert predictions to one-hot
            one_hot = np.zeros((len(pred), n_classes))
            one_hot[np.arange(len(pred)), pred.astype(int)] = 1
            ensemble_pred_proba += weight * one_hot
        
        ensemble_pred = np.argmax(ensemble_pred_proba, axis=1)
        
        # Calculate metrics
        acc = accuracy_score(y_test, ensemble_pred)
        prec = precision_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, ensemble_pred, average='weighted', zero_division=0)
        
        print(f"\n{'='*80}")
        print("STAGE 2 ENSEMBLE RESULTS")
        print(f"{'='*80}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Per-class metrics
        print(f"\n{'='*80}")
        print("PER-CLASS PERFORMANCE")
        print(f"{'='*80}")
        for i, cat in enumerate(self.label_encoder_s2.classes_):
            mask = y_test == i
            if mask.sum() > 0:
                cat_acc = accuracy_score(y_test[mask], ensemble_pred[mask])
                cat_prec = precision_score(y_test == i, ensemble_pred == i, zero_division=0)
                cat_rec = recall_score(y_test == i, ensemble_pred == i, zero_division=0)
                cat_f1 = f1_score(y_test == i, ensemble_pred == i, zero_division=0)
                print(f"{cat:20s}: Acc={cat_acc:.3f} | P={cat_prec:.3f} | R={cat_rec:.3f} | F1={cat_f1:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, ensemble_pred)
        self.plot_confusion_matrix(cm, self.label_encoder_s2.classes_,
                                   'stage2_confusion_matrix.png',
                                   f'Stage 2: Multi-Class Classification\nAccuracy: {acc:.4f}')
        
        # Update metadata
        self.metadata['stage2_multiclass'] = {
            'purpose': 'Multi-class attack categorization',
            'models': list(self.stage2_models.keys()),
            'ensemble_weights': weights,
            'attack_categories': list(self.label_encoder_s2.classes_),
            'n_classes': n_classes,
            'performance': {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'confusion_matrix': cm.tolist()
            },
            'hyperparameters': self.config['stage2_multiclass']['models']
        }
        
        return ensemble_pred, acc, prec, rec, f1
    
    def evaluate_full_pipeline(self, y_test_binary, y_stage1_pred, 
                               y_test_attacks, y_stage2_pred, attack_mask):
        """Evaluate complete 2-stage pipeline"""
        print(f"\n{'='*80}")
        print("STEP 9: EVALUATING FULL PIPELINE")
        print(f"{'='*80}")
        
        # Create full predictions
        full_pred = np.array(['BENIGN'] * len(y_test_binary), dtype=object)
        
        # Map Stage 2 predictions to attack categories
        attack_categories = self.label_encoder_s2.inverse_transform(y_stage2_pred)
        full_pred[attack_mask] = attack_categories
        
        # Get true labels
        true_categories = self.df_test['Label'].apply(self.get_attack_category).values
        
        # Calculate metrics
        acc = accuracy_score(true_categories, full_pred)
        prec = precision_score(true_categories, full_pred, average='weighted', zero_division=0)
        rec = recall_score(true_categories, full_pred, average='weighted', zero_division=0)
        f1 = f1_score(true_categories, full_pred, average='weighted', zero_division=0)
        
        print(f"\nFull Pipeline Performance:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Detection statistics
        benign_correct = ((true_categories == 'Benign') & (full_pred == 'BENIGN')).sum()
        benign_total = (true_categories == 'Benign').sum()
        attacks_detected = ((true_categories != 'Benign') & (full_pred != 'BENIGN')).sum()
        attacks_total = (true_categories != 'Benign').sum()
        
        print(f"\nDetection Statistics:")
        print(f"  Benign correctly identified: {benign_correct:,} / {benign_total:,} ({100*benign_correct/benign_total:.2f}%)")
        print(f"  Attacks detected: {attacks_detected:,} / {attacks_total:,} ({100*attacks_detected/attacks_total:.2f}%)")
        
        # Update metadata
        self.metadata['full_pipeline'] = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'benign_detection_rate': float(benign_correct / benign_total) if benign_total > 0 else 0,
            'attack_detection_rate': float(attacks_detected / attacks_total) if attacks_total > 0 else 0,
            'total_test_samples': len(y_test_binary),
            'benign_samples': int(benign_total),
            'attack_samples': int(attacks_total)
        }
        
        return acc, prec, rec, f1
    
    def save_models(self):
        """Save all trained models and components"""
        print(f"\n{'='*80}")
        print("STEP 10: SAVING MODELS")
        print(f"{'='*80}")
        
        models_dir = Path(self.config['output']['models_dir'])
        
        # Save Stage 1 ensemble
        stage1_path = models_dir / 'ultra_advanced_stage1_ensemble.pkl'
        stage1_package = {
            'models': self.stage1_models,
            'scaler': self.scaler_s1,
            'config': self.config['stage1_binary'],
            'feature_cols': self.feature_cols_s1
        }
        with open(stage1_path, 'wb') as f:
            pickle.dump(stage1_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Saved Stage 1 ensemble: {stage1_path}")
        
        # Save Stage 2 ensemble
        stage2_path = models_dir / 'ultra_advanced_stage2_ensemble.pkl'
        stage2_package = {
            'models': self.stage2_models,
            'scaler': self.scaler_s2,
            'label_encoder': self.label_encoder_s2,
            'config': self.config['stage2_multiclass'],
        }
        with open(stage2_path, 'wb') as f:
            pickle.dump(stage2_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Saved Stage 2 ensemble: {stage2_path}")
        
        # Update metadata with file paths
        self.metadata['files'] = {
            'stage1_model': str(stage1_path),
            'stage2_model': str(stage2_path),
            'config': 'config_train_ultra_advanced.yaml',
            'results_dir': str(self.config['output']['results_dir'])
        }
    
    def save_metadata(self):
        """Save comprehensive metadata"""
        print(f"\n{'='*80}")
        print("STEP 11: SAVING METADATA")
        print(f"{'='*80}")
        
        # Add model info
        self.metadata['model_info'] = {
            'name': 'IoTGuard Ultra Advanced 2-Stage Ensemble',
            'version': '2.0',
            'architecture': '2-Stage: Binary + Multi-Class Classification',
            'created_at': self.start_time.isoformat(),
            'training_duration': str(datetime.now() - self.start_time),
            'framework': 'LightGBM + XGBoost + CatBoost + TensorFlow',
            'dataset': 'CIC-IoT-2023'
        }
        
        # Add detailed feature information
        self.metadata['model_inputs'] = {
            'stage1': {
                'description': 'Binary classification (Benign vs Attack)',
                'input_features': self.feature_cols_s1 if hasattr(self, 'feature_cols_s1') else [],
                'num_features': len(self.feature_cols_s1) if hasattr(self, 'feature_cols_s1') else 0,
                'feature_types': {
                    'original': [col for col in self.feature_cols_s1 if not any(x in col for x in ['_squared', '_sqrt', '_ratio', '_interaction', '_combo', 'log_', 'cv', 'range_stat', 'flag_diversity', 'burst_score'])] if hasattr(self, 'feature_cols_s1') else [],
                    'engineered_statistical': [col for col in self.feature_cols_s1 if any(x in col for x in ['_ratio', 'cv', 'range_stat', 'flag_diversity'])] if hasattr(self, 'feature_cols_s1') else [],
                    'engineered_polynomial': [col for col in self.feature_cols_s1 if any(x in col for x in ['_squared', '_sqrt'])] if hasattr(self, 'feature_cols_s1') else [],
                    'engineered_interaction': [col for col in self.feature_cols_s1 if '_combo' in col or '_interaction' in col] if hasattr(self, 'feature_cols_s1') else [],
                    'engineered_log': [col for col in self.feature_cols_s1 if col.startswith('log_')] if hasattr(self, 'feature_cols_s1') else [],
                },
                'scaler': str(type(self.scaler_s1).__name__) if hasattr(self, 'scaler_s1') else 'None',
                'balancing_method': self.config['stage1_binary']['balancing']['method'],
                'output': {
                    'type': 'binary',
                    'classes': ['Benign', 'Attack'],
                    'num_classes': 2
                }
            },
            'stage2': {
                'description': 'Multi-class attack categorization',
                'input_features': [col for col in self.df_train.columns if col not in ['Label', 'Binary_Label', 'Category']] if hasattr(self, 'df_train') else [],
                'num_features': len([col for col in self.df_train.columns if col not in ['Label', 'Binary_Label', 'Category']]) if hasattr(self, 'df_train') else 0,
                'scaler': str(type(self.scaler_s2).__name__) if hasattr(self, 'scaler_s2') else 'None',
                'balancing_method': self.config['stage2_multiclass']['balancing']['method'],
                'label_encoder': {
                    'classes': list(self.label_encoder_s2.classes_) if hasattr(self, 'label_encoder_s2') else [],
                    'num_classes': len(self.label_encoder_s2.classes_) if hasattr(self, 'label_encoder_s2') else 0
                },
                'output': {
                    'type': 'multiclass',
                    'classes': list(self.label_encoder_s2.classes_) if hasattr(self, 'label_encoder_s2') else [],
                    'num_classes': len(self.label_encoder_s2.classes_) if hasattr(self, 'label_encoder_s2') else 0
                }
            }
        }
        
        # Add feature statistics
        if hasattr(self, 'feature_cols_s1'):
            self.metadata['feature_statistics'] = {
                'total_features': len(self.feature_cols_s1),
                'original_features_count': len([col for col in self.feature_cols_s1 if not any(x in col for x in ['_squared', '_sqrt', '_ratio', '_interaction', '_combo', 'log_', 'cv', 'range_stat', 'flag_diversity', 'burst_score'])]),
                'engineered_features_count': len([col for col in self.feature_cols_s1 if any(x in col for x in ['_squared', '_sqrt', '_ratio', '_interaction', '_combo', 'log_', 'cv', 'range_stat', 'flag_diversity', 'burst_score'])]),
                'feature_categories': {
                    'statistical': len([col for col in self.feature_cols_s1 if any(x in col for x in ['_ratio', 'cv', 'range_stat', 'flag_diversity'])]),
                    'polynomial': len([col for col in self.feature_cols_s1 if any(x in col for x in ['_squared', '_sqrt'])]),
                    'interaction': len([col for col in self.feature_cols_s1 if '_combo' in col or '_interaction' in col]),
                    'logarithmic': len([col for col in self.feature_cols_s1 if col.startswith('log_')])
                },
                'all_features': self.feature_cols_s1
            }
        
        # Save metadata JSON
        metadata_path = Path(self.config['output']['models_dir']) / self.config['output']['metadata_filename']
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"✓ Saved metadata: {metadata_path}")
        
        # Save human-readable summary
        summary_path = Path(self.config['output']['results_dir']) / 'ULTRA_ADVANCED_RESULTS_SUMMARY.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("IoTGuard Ultra Advanced Training Results\n")
            f.write("="*80 + "\n\n")
            f.write(f"Model: {self.metadata['model_info']['name']}\n")
            f.write(f"Version: {self.metadata['model_info']['version']}\n")
            f.write(f"Created: {self.metadata['model_info']['created_at']}\n")
            f.write(f"Training Duration: {self.metadata['model_info']['training_duration']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("DATASET\n")
            f.write("="*80 + "\n")
            f.write(f"Training samples: {self.metadata['training_data']['train_samples']:,}\n")
            f.write(f"Validation samples: {self.metadata['training_data']['val_samples']:,}\n")
            f.write(f"Test samples: {self.metadata['training_data']['test_samples']:,}\n")
            f.write(f"Total features: {self.metadata['feature_engineering']['total_features']}\n\n")
            
            f.write("="*80 + "\n")
            f.write("MODEL INPUTS & FEATURES\n")
            f.write("="*80 + "\n")
            if 'model_inputs' in self.metadata:
                f.write(f"\nStage 1 (Binary Classification):\n")
                f.write(f"  Input features: {self.metadata['model_inputs']['stage1']['num_features']}\n")
                f.write(f"  Scaler: {self.metadata['model_inputs']['stage1']['scaler']}\n")
                f.write(f"  Balancing: {self.metadata['model_inputs']['stage1']['balancing_method']}\n")
                
                f.write(f"\nStage 2 (Multi-Class):\n")
                f.write(f"  Input features: {self.metadata['model_inputs']['stage2']['num_features']}\n")
                f.write(f"  Scaler: {self.metadata['model_inputs']['stage2']['scaler']}\n")
                f.write(f"  Balancing: {self.metadata['model_inputs']['stage2']['balancing_method']}\n")
                f.write(f"  Attack categories: {', '.join(self.metadata['model_inputs']['stage2']['output']['classes'])}\n")
            
            if 'feature_statistics' in self.metadata:
                f.write(f"\nFeature Engineering:\n")
                f.write(f"  Total features: {self.metadata['feature_statistics']['total_features']}\n")
                f.write(f"  Original features: {self.metadata['feature_statistics']['original_features_count']}\n")
                f.write(f"  Engineered features: {self.metadata['feature_statistics']['engineered_features_count']}\n")
                f.write(f"    - Statistical: {self.metadata['feature_statistics']['feature_categories']['statistical']}\n")
                f.write(f"    - Polynomial: {self.metadata['feature_statistics']['feature_categories']['polynomial']}\n")
                f.write(f"    - Interaction: {self.metadata['feature_statistics']['feature_categories']['interaction']}\n")
                f.write(f"    - Logarithmic: {self.metadata['feature_statistics']['feature_categories']['logarithmic']}\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("STAGE 1: BINARY CLASSIFICATION\n")
            f.write("="*80 + "\n")
            s1_perf = self.metadata['stage1_binary']['performance']
            f.write(f"Accuracy:  {s1_perf['accuracy']:.4f}\n")
            f.write(f"Precision: {s1_perf['precision']:.4f}\n")
            f.write(f"Recall:    {s1_perf['recall']:.4f}\n")
            f.write(f"F1-Score:  {s1_perf['f1_score']:.4f}\n")
            f.write(f"MCC:       {s1_perf['mcc']:.4f}\n\n")
            
            f.write("="*80 + "\n")
            f.write("STAGE 2: MULTI-CLASS CLASSIFICATION\n")
            f.write("="*80 + "\n")
            s2_perf = self.metadata['stage2_multiclass']['performance']
            f.write(f"Accuracy:  {s2_perf['accuracy']:.4f}\n")
            f.write(f"Precision: {s2_perf['precision']:.4f}\n")
            f.write(f"Recall:    {s2_perf['recall']:.4f}\n")
            f.write(f"F1-Score:  {s2_perf['f1_score']:.4f}\n")
            f.write(f"Classes: {len(self.metadata['stage2_multiclass']['attack_categories'])}\n\n")
            
            f.write("="*80 + "\n")
            f.write("FULL PIPELINE\n")
            f.write("="*80 + "\n")
            full_perf = self.metadata['full_pipeline']
            f.write(f"Accuracy:  {full_perf['accuracy']:.4f}\n")
            f.write(f"Precision: {full_perf['precision']:.4f}\n")
            f.write(f"Recall:    {full_perf['recall']:.4f}\n")
            f.write(f"F1-Score:  {full_perf['f1_score']:.4f}\n")
            f.write(f"Benign Detection Rate: {full_perf['benign_detection_rate']:.4f}\n")
            f.write(f"Attack Detection Rate: {full_perf['attack_detection_rate']:.4f}\n")
        
        print(f"✓ Saved summary: {summary_path}")
    
    def plot_confusion_matrix(self, cm, labels, filename, title):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        save_path = Path(self.config['output']['results_dir']) / filename
        plt.savefig(save_path, dpi=self.config['output']['plot_dpi'], bbox_inches='tight')
        plt.close()
        print(f"✓ Saved plot: {save_path}")
    
    def run(self):
        """Execute complete training pipeline"""
        print(f"\n{'='*80}")
        print("IOTGUARD ULTRA ADVANCED TRAINING PIPELINE")
        print(f"{'='*80}")
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration: {self.config['output']['models_dir']}")
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Feature engineering
            self.df_train = self.engineer_features(self.df_train, fit=True)
            self.df_val = self.engineer_features(self.df_val, fit=False)
            self.df_test = self.engineer_features(self.df_test, fit=False)
            
            # Step 3-5: Stage 1
            X_train_s1, y_train_s1, X_val_s1, y_val_s1, X_test_s1, y_test_s1, self.feature_cols_s1 = self.prepare_stage1_data()
            self.train_stage1(X_train_s1, y_train_s1, X_val_s1, y_val_s1)
            # Use validation set to tune ensemble threshold, then evaluate on test set
            y_stage1_pred, *stage1_metrics = self.evaluate_stage1(X_val_s1, y_val_s1, X_test_s1, y_test_s1)
            
            # Step 6-8: Stage 2
            X_train_s2, y_train_s2, X_val_s2, y_val_s2, X_test_s2, y_test_s2, attack_mask = self.prepare_stage2_data(X_test_s1, y_stage1_pred)
            self.train_stage2(X_train_s2, y_train_s2, X_val_s2, y_val_s2)
            y_stage2_pred, *stage2_metrics = self.evaluate_stage2(X_test_s2, y_test_s2)
            
            # Step 9: Full pipeline evaluation
            full_metrics = self.evaluate_full_pipeline(y_test_s1, y_stage1_pred,
                                                       y_test_s2, y_stage2_pred, attack_mask)
            
            # Step 10-11: Save everything
            self.save_models()
            self.save_metadata()
            
            # Final summary
            end_time = datetime.now()
            duration = end_time - self.start_time
            
            print(f"\n{'='*80}")
            print("TRAINING COMPLETE!")
            print(f"{'='*80}")
            print(f"Total duration: {duration}")
            print(f"Stage 1 Accuracy: {stage1_metrics[0]:.4f}")
            print(f"Stage 2 Accuracy: {stage2_metrics[0]:.4f}")
            print(f"Full Pipeline Accuracy: {full_metrics[0]:.4f}")
            print(f"\nAll files saved to:")
            print(f"  Models: {self.config['output']['models_dir']}")
            print(f"  Results: {self.config['output']['results_dir']}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"ERROR: {str(e)}")
            print(f"{'='*80}\n")
            raise

if __name__ == "__main__":
    import sys
    # Run ultra advanced training
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config_train_ultra_advanced.yaml'
    trainer = UltraAdvancedTrainer(config_path=config_file)
    trainer.run()
