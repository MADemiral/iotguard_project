"""
IoTGuard - 2-STAGE APPROACH: Binary First, Then Multi-Class
Stage 1: Binary (Benign vs Attack)
Stage 2: Multi-Class (Which type of attack)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from collections import Counter

print("\n" + "="*70)
print("IOTGUARD - 2-STAGE APPROACH")
print("STAGE 1: Binary Classification (Benign vs Attack)")
print("STAGE 2: Multi-Class Classification (Which Attack)")
print("="*70)

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1/6] Loading CSV files...")

csv_files = sorted(glob.glob('dataset/CSV/MERGED_CSV/*.csv'))
print(f"Found {len(csv_files)} CSV files")

n_train = int(len(csv_files) * 0.7)
train_files = csv_files[:n_train]
test_files = csv_files[n_train:]

print(f"Training files: {len(train_files)}")
print(f"Testing files: {len(test_files)}")

# Load training data
print("\nLoading training data...")
train_dfs = []
for f in train_files[:20]:
    df = pd.read_csv(f, nrows=10000)
    train_dfs.append(df)
    print(f"  Loaded {len(df)} from {os.path.basename(f)}")

df_train = pd.concat(train_dfs, ignore_index=True)
print(f"\nTotal training samples: {len(df_train):,}")

# Load testing data
print("\nLoading testing data...")
test_dfs = []
for f in test_files[:8]:
    df = pd.read_csv(f, nrows=5000)
    test_dfs.append(df)
    print(f"  Loaded {len(df)} from {os.path.basename(f)}")

df_test = pd.concat(test_dfs, ignore_index=True)
print(f"\nTotal testing samples: {len(df_test):,}")

# ============================================================================
# PREPARE DATA
# ============================================================================
print("\n[2/6] Preparing data...")

# Define 7-class mapping
class_mapping = {
    'DDoS': ['DDOS-RSTFINFLOOD', 'DDOS-PSHACK_FLOOD', 'DDOS-SYN_FLOOD',
             'DDOS-UDP_FLOOD', 'DDOS-TCP_FLOOD', 'DDOS-ICMP_FLOOD',
             'DDOS-SYNONYMOUSIP_FLOOD', 'DDOS-ACK_FRAGMENTATION',
             'DDOS-UDP_FRAGMENTATION', 'DDOS-ICMP_FRAGMENTATION',
             'DDOS-SLOWLORIS', 'DDOS-HTTP_FLOOD'],
    'DoS': ['DOS-UDP_FLOOD', 'DOS-SYN_FLOOD', 'DOS-TCP_FLOOD', 'DOS-HTTP_FLOOD'],
    'Mirai': ['MIRAI-GREETH_FLOOD', 'MIRAI-GREIP_FLOOD', 'MIRAI-UDPPLAIN'],
    'Recon': ['RECON-PINGSWEEP', 'RECON-OSSCAN', 'RECON-PORTSCAN',
              'VULNERABILITYSCAN', 'RECON-HOSTDISCOVERY'],
    'Spoofing': ['DNS_SPOOFING', 'MITM-ARPSPOOFING'],
    'Web': ['BROWSERHIJACKING', 'BACKDOOR_MALWARE', 'XSS',
            'UPLOADING_ATTACK', 'SQLINJECTION', 'COMMANDINJECTION'],
    'BruteForce': ['DICTIONARYBRUTEFORCE'],
    'Benign': ['BENIGN']
}

label_to_group = {}
for group, labels in class_mapping.items():
    for label in labels:
        label_to_group[label] = group

df_train['Category'] = df_train['Label'].map(label_to_group)
df_test['Category'] = df_test['Label'].map(label_to_group)

# Create binary labels
df_train['Binary'] = (df_train['Label'] != 'BENIGN').astype(int)
df_test['Binary'] = (df_test['Label'] != 'BENIGN').astype(int)

print("\nTraining data - Binary distribution:")
print(df_train['Binary'].value_counts())
print(f"  0 (Benign): {(df_train['Binary']==0).sum():,}")
print(f"  1 (Attack): {(df_train['Binary']==1).sum():,}")

print("\nTraining data - Category distribution:")
print(df_train['Category'].value_counts())

# Clean data
def clean_data(df):
    df = df.replace([np.inf, -np.inf], np.nan)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    
    # DON'T clip outliers - it might remove all benign samples!
    # Outlier clipping removed because benign traffic has different patterns
    
    return df

print("\nCleaning data (without removing benign outliers)...")
df_train = clean_data(df_train)
df_test = clean_data(df_test)

print(f"After cleaning - Benign: {(df_train['Label']=='BENIGN').sum()}, Attack: {(df_train['Label']!='BENIGN').sum()}")

# Add engineered features
def add_features(df):
    df = df.copy()
    df['syn_ack_ratio'] = df['syn_flag_number'] / (df['ack_flag_number'] + 1)
    df['packet_rate_ratio'] = df['Rate'] / (df['Number'] + 1)
    df['flags_total'] = (df['syn_flag_number'] + df['ack_flag_number'] + 
                         df['rst_flag_number'] + df['fin_flag_number'] + 
                         df['psh_flag_number'])
    df['size_per_packet'] = df['Tot size'] / (df['Number'] + 1)
    df['variance_avg_ratio'] = df['Variance'] / (df['AVG'] + 1)
    return df

df_train = add_features(df_train)
df_test = add_features(df_test)

# Check we still have benign samples
print(f"\nAfter feature engineering:")
print(f"  Benign samples: {(df_train['Label']=='BENIGN').sum()}")
print(f"  Attack samples: {(df_train['Label']!='BENIGN').sum()}")

print("\n✓ Data prepared!")

# ============================================================================
# STAGE 1: BINARY CLASSIFIER (Benign vs Attack)
# ============================================================================
print("\n[3/6] Training STAGE 1: Binary Classifier (Benign vs Attack)...")

# Prepare data for Stage 1
X_train_stage1 = df_train.drop(['Label', 'Category', 'Binary'], axis=1)
y_train_stage1 = df_train['Binary']

X_test_stage1 = df_test.drop(['Label', 'Category', 'Binary'], axis=1)
y_test_stage1 = df_test['Binary']

print(f"\nStage 1 data shapes:")
print(f"  X_train: {X_train_stage1.shape}")
print(f"  y_train unique values: {np.unique(y_train_stage1)}")
print(f"  y_train counts: Benign={(y_train_stage1==0).sum()}, Attack={(y_train_stage1==1).sum()}")

# Scale features
scaler_stage1 = RobustScaler()
X_train_scaled_s1 = scaler_stage1.fit_transform(X_train_stage1)
X_test_scaled_s1 = scaler_stage1.transform(X_test_stage1)

# Balance classes with SMOTE
print("\nBalancing binary classes with SMOTE...")
print(f"Before SMOTE: Benign={(y_train_stage1==0).sum()}, Attack={(y_train_stage1==1).sum()}")

# Check if we have both classes
if len(np.unique(y_train_stage1)) < 2:
    print("ERROR: Only one class found in training data!")
    print(f"Unique values: {np.unique(y_train_stage1)}")
    raise ValueError("Training data must have both Benign and Attack samples")

smote_stage1 = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=min(5, (y_train_stage1==0).sum()-1))
X_train_balanced_s1, y_train_balanced_s1 = smote_stage1.fit_resample(X_train_scaled_s1, y_train_stage1)

print(f"After SMOTE: Benign={(y_train_balanced_s1==0).sum()}, Attack={(y_train_balanced_s1==1).sum()}")

# Train LightGBM for binary classification
print("\nTraining LightGBM Binary Classifier...")

train_data_s1 = lgb.Dataset(X_train_balanced_s1, label=y_train_balanced_s1)

params_stage1 = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.03,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'max_depth': 10,
    'min_data_in_leaf': 10,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'verbose': -1,
    'n_jobs': -1
}

stage1_model = lgb.train(
    params_stage1,
    train_data_s1,
    num_boost_round=300,
    valid_sets=[train_data_s1],
    callbacks=[lgb.log_evaluation(period=100)]
)

# Evaluate Stage 1
y_pred_proba_s1 = stage1_model.predict(X_test_scaled_s1)
y_pred_s1 = (y_pred_proba_s1 > 0.5).astype(int)

print("\n" + "="*70)
print("STAGE 1 RESULTS: Binary Classification")
print("="*70)
print("\nClassification Report:")
print(classification_report(y_test_stage1, y_pred_s1, 
                          target_names=['Benign', 'Attack']))

cm_stage1 = confusion_matrix(y_test_stage1, y_pred_s1)
print("\nConfusion Matrix:")
print(f"              Predicted Benign  Predicted Attack")
print(f"Actual Benign      {cm_stage1[0,0]:6d}           {cm_stage1[0,1]:6d}")
print(f"Actual Attack      {cm_stage1[1,0]:6d}           {cm_stage1[1,1]:6d}")

# Calculate metrics
tn, fp, fn, tp = cm_stage1.ravel()
print(f"\nTrue Negatives (Benign correctly): {tn}")
print(f"False Positives (Benign as Attack): {fp}")
print(f"False Negatives (Attack as Benign): {fn}")
print(f"True Positives (Attack correctly): {tp}")
print(f"\nAccuracy: {accuracy_score(y_test_stage1, y_pred_s1):.4f}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm_stage1, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Attack'],
            yticklabels=['Benign', 'Attack'])
plt.title('STAGE 1: Binary Classification (Benign vs Attack)', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/stage1_binary_confusion_matrix.png', dpi=300)
print("\n✓ Saved: results/stage1_binary_confusion_matrix.png")
plt.close()

# ============================================================================
# STAGE 2: MULTI-CLASS CLASSIFIER (Attack Types)
# ============================================================================
print("\n[4/6] Training STAGE 2: Multi-Class Attack Classifier...")

# Prepare data for Stage 2 (only attacks)
attack_train = df_train[df_train['Binary'] == 1].copy()
attack_test = df_test[df_test['Binary'] == 1].copy()

print(f"\nAttack training samples: {len(attack_train):,}")
print(f"Attack testing samples: {len(attack_test):,}")

X_train_stage2 = attack_train.drop(['Label', 'Category', 'Binary'], axis=1)
y_train_stage2 = attack_train['Category']

X_test_stage2 = attack_test.drop(['Label', 'Category', 'Binary'], axis=1)
y_test_stage2 = attack_test['Category']

# Encode labels
label_encoder_s2 = LabelEncoder()
y_train_encoded_s2 = label_encoder_s2.fit_transform(y_train_stage2)
y_test_encoded_s2 = label_encoder_s2.transform(y_test_stage2)

# Scale features
scaler_stage2 = RobustScaler()
X_train_scaled_s2 = scaler_stage2.fit_transform(X_train_stage2)
X_test_scaled_s2 = scaler_stage2.transform(X_test_stage2)

# Balance classes with SMOTE
print("\nBalancing attack classes with SMOTE...")
print(f"Before SMOTE: {Counter(y_train_encoded_s2)}")

smote_stage2 = SMOTE(sampling_strategy='not majority', random_state=42, k_neighbors=3)
undersampler_s2 = RandomUnderSampler(sampling_strategy='not minority', random_state=42)

X_train_balanced_s2, y_train_balanced_s2 = smote_stage2.fit_resample(X_train_scaled_s2, y_train_encoded_s2)
X_train_balanced_s2, y_train_balanced_s2 = undersampler_s2.fit_resample(X_train_balanced_s2, y_train_balanced_s2)

print(f"After SMOTE+Undersample: {Counter(y_train_balanced_s2)}")

# Train LightGBM for multi-class
train_data_s2 = lgb.Dataset(X_train_balanced_s2, label=y_train_balanced_s2)

params_stage2 = {
    'objective': 'multiclass',
    'num_class': len(label_encoder_s2.classes_),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.03,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'max_depth': 10,
    'min_data_in_leaf': 10,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'verbose': -1,
    'n_jobs': -1
}

print("\nTraining LightGBM Multi-Class Classifier...")
stage2_model = lgb.train(
    params_stage2,
    train_data_s2,
    num_boost_round=500,
    valid_sets=[train_data_s2],
    callbacks=[lgb.log_evaluation(period=100)]
)

# Evaluate Stage 2
y_pred_proba_s2 = stage2_model.predict(X_test_scaled_s2)
y_pred_encoded_s2 = np.argmax(y_pred_proba_s2, axis=1)
y_pred_s2 = label_encoder_s2.inverse_transform(y_pred_encoded_s2)

print("\n" + "="*70)
print("STAGE 2 RESULTS: Multi-Class Attack Classification")
print("="*70)
print("\nClassification Report:")
print(classification_report(y_test_stage2, y_pred_s2, zero_division=0))

cm_stage2 = confusion_matrix(y_test_encoded_s2, y_pred_encoded_s2)
print("\nConfusion Matrix:")
print(cm_stage2)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_stage2, annot=True, fmt='d', cmap='Reds',
            xticklabels=label_encoder_s2.classes_,
            yticklabels=label_encoder_s2.classes_)
plt.title('STAGE 2: Multi-Class Attack Types', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('results/stage2_multiclass_confusion_matrix.png', dpi=300)
print("\n✓ Saved: results/stage2_multiclass_confusion_matrix.png")
plt.close()

# ============================================================================
# FULL 2-STAGE PIPELINE
# ============================================================================
print("\n[5/6] Testing Full 2-Stage Pipeline...")

X_all_test = df_test.drop(['Label', 'Category', 'Binary'], axis=1)
X_all_scaled_s1 = scaler_stage1.transform(X_all_test)
y_all_true = df_test['Category']

# Stage 1: Binary prediction
print("Running Stage 1 (Binary)...")
y_stage1_proba = stage1_model.predict(X_all_scaled_s1)
y_stage1_pred = (y_stage1_proba > 0.5).astype(int)

# Stage 2: Multi-class prediction for attacks
print("Running Stage 2 (Multi-Class for detected attacks)...")
final_predictions = []

for i in range(len(y_stage1_pred)):
    if y_stage1_pred[i] == 0:  # Predicted as Benign
        final_predictions.append('Benign')
    else:  # Predicted as Attack - run Stage 2
        X_sample = X_all_test.iloc[i:i+1]
        X_sample_scaled = scaler_stage2.transform(X_sample)
        proba = stage2_model.predict(X_sample_scaled)[0]
        pred_class = np.argmax(proba)
        category = label_encoder_s2.inverse_transform([pred_class])[0]
        final_predictions.append(category)

final_predictions = pd.Series(final_predictions)

print("\n" + "="*70)
print("FULL 2-STAGE PIPELINE RESULTS")
print("="*70)
print("\nClassification Report:")
print(classification_report(y_all_true, final_predictions, zero_division=0))

# Confusion matrix
all_labels = sorted(list(set(y_all_true) | set(final_predictions)))
cm_full = confusion_matrix(y_all_true, final_predictions, labels=all_labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Greens',
            xticklabels=all_labels,
            yticklabels=all_labels)
plt.title('FULL 2-STAGE PIPELINE: Stage 1 → Stage 2', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('results/2stage_full_pipeline_confusion_matrix.png', dpi=300)
print("\n✓ Saved: results/2stage_full_pipeline_confusion_matrix.png")
plt.close()

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n[6/6] Saving 2-Stage models...")

import pickle

# Save Stage 1
stage1_model.save_model('models/stage1_binary_lgb.txt')
with open('models/stage1_metadata.pkl', 'wb') as f:
    pickle.dump({
        'scaler': scaler_stage1,
        'smote': smote_stage1
    }, f)
print("✓ Saved: models/stage1_binary_lgb.txt")

# Save Stage 2
stage2_model.save_model('models/stage2_multiclass_lgb.txt')
with open('models/stage2_metadata.pkl', 'wb') as f:
    pickle.dump({
        'scaler': scaler_stage2,
        'label_encoder': label_encoder_s2,
        'smote': smote_stage2,
        'undersampler': undersampler_s2
    }, f)
print("✓ Saved: models/stage2_multiclass_lgb.txt")

print("\n" + "="*70)
print("2-STAGE TRAINING COMPLETE!")
print("="*70)
print(f"\nTraining samples: {len(df_train):,}")
print(f"Testing samples: {len(df_test):,}")
print("\nGenerated files:")
print("  - results/stage1_binary_confusion_matrix.png")
print("  - results/stage2_multiclass_confusion_matrix.png")
print("  - results/2stage_full_pipeline_confusion_matrix.png")
print("  - models/stage1_binary_lgb.txt")
print("  - models/stage2_multiclass_lgb.txt")
print("\nThis 2-stage approach should have:")
print("  ✓ Better benign detection (Stage 1 focused on binary)")
print("  ✓ Better attack classification (Stage 2 only on attacks)")
print("  ✓ Less DDoS/DoS confusion (separate stages)")
print("="*70 + "\n")
