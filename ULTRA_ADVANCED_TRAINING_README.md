# IoTGuard Ultra Advanced Training System

## Overview
The Ultra Advanced Training System is a state-of-the-art machine learning pipeline designed for maximum performance on the CIC-IoT-2023 IoT network intrusion detection dataset.

## Key Features

### 🎯 Advanced Architecture
- **4-Model Ensemble per Stage**:
  - LightGBM (gradient boosting)
  - XGBoost (extreme gradient boosting)
  - CatBoost (categorical boosting)
  - Neural Network (deep learning with TensorFlow)

### 📊 Intelligent Data Processing
- **Smart Sampling Strategy**:
  - Keeps ALL benign samples (critical minority class)
  - Balances attack categories (100k samples per type)
  - Splits into train/validation/test sets
  
- **Advanced Feature Engineering**:
  - Statistical features (ratios, coefficients)
  - Time-based features (burst detection)
  - Polynomial features (degree 2, selective)
  - Interaction features between key variables
  - 68+ total features from 39 original

### ⚖️ Sophisticated Balancing
- **Stage 1**: SMOTE-Tomek hybrid (removes noise while oversampling)
- **Stage 2**: SMOTE + Random Undersampling (balanced multi-class)

### 📈 Hierarchical Classification
1. **Stage 1**: Binary (Benign vs Attack) - 99%+ accuracy expected
2. **Stage 2**: Multi-Class (6 attack categories):
   - DDoS-DoS (16 attack types combined)
   - Mirai (3 variants)
   - Reconnaissance (5 types including scanning)
   - Spoofing (DNS, ARP)
   - Web-Based (SQL injection, XSS, Command injection, etc.)
   - BruteForce (dictionary attacks)

## Dataset Understanding

### Structure
```
dataset/CSV/
├── CSV/                    # Individual attack type folders
│   ├── Benign_Final/      # 4 benign traffic files (217MB)
│   ├── DDoS-SYN_Flood/    # 16 attack files (734MB)
│   ├── Mirai-greeth_flood/# 29 attack files
│   └── ... (34 attack types total)
│
└── MERGED_CSV/            # Pre-merged files (USED FOR TRAINING)
    ├── Merged01.csv       # 712k samples, 40 columns
    ├── Merged02.csv
    └── ... (63 files, ~8.7GB total)
```

### Dataset Statistics (from sampling analysis)
- **Total samples**: ~22 million (63 merged files)
- **Class imbalance**: 97.67% attacks, 2.33% benign
- **Features**: 39 network features + 1 label
- **Attack types**: 33 distinct attack types in original data
- **Grouped categories**: 7 categories (6 attacks + benign)

### Feature Columns (39 original)
```
Network Layer:
- Header_Length, Protocol Type, Time_To_Live
- fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number
- ack_flag_number, ece_flag_number, cwr_flag_number

Flow Statistics:
- Rate, IAT, Number, Variance
- Tot sum, Tot size, Min, Max, AVG, Std

Protocol Indicators (binary):
- HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC
- TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC

Packet Counts:
- ack_count, syn_count, fin_count, rst_count
```

## Configuration

### config_train_ultra_advanced.yaml

```yaml
dataset:
  merged_csv_path: "../../dataset/CSV/MERGED_CSV/"
  train_files: 40   # First 40 files for training
  val_files: 8      # Next 8 for validation  
  test_files: 15    # Last 15 for testing

data_processing:
  benign_sampling: "all"              # Keep all benign
  attack_sampling_per_class: 100000   # Max per attack category
  polynomial_features: true           # x^2, sqrt(x)
  statistical_features: true          # Ratios, CV, interactions
  time_based_features: true           # Burst detection
  scaler_type: "robust"               # Robust to outliers

stage1_binary:
  models:
    lightgbm:
      num_leaves: 127
      learning_rate: 0.03
      n_estimators: 500
    xgboost:
      max_depth: 12
      learning_rate: 0.03
      n_estimators: 500
    catboost:
      depth: 10
      iterations: 500
    neural_network:
      hidden_layers: [256, 128, 64, 32]
      dropout_rate: 0.3
      epochs: 50
  
  ensemble:
    weights:
      lightgbm: 0.35
      xgboost: 0.30
      catboost: 0.25
      neural_network: 0.10

stage2_multiclass:
  models:
    lightgbm:
      num_leaves: 127
      learning_rate: 0.02
      n_estimators: 600
    xgboost:
      max_depth: 12
      learning_rate: 0.02
      n_estimators: 600
    catboost:
      depth: 10
      iterations: 600
  
  ensemble:
    weights:
      lightgbm: 0.40
      xgboost: 0.35
      catboost: 0.25
```

## Installation

### Required Packages
```bash
# Core ML libraries
pip install pandas numpy scikit-learn

# Gradient boosting frameworks
pip install lightgbm xgboost catboost

# Imbalanced learning
pip install imbalanced-learn

# Deep learning (optional but recommended)
pip install tensorflow

# Visualization
pip install matplotlib seaborn

# Configuration
pip install pyyaml
```

Or install all at once:
```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost imbalanced-learn tensorflow matplotlib seaborn pyyaml
```

## Usage

### Basic Training
```bash
cd /path/to/iotguard_project/scripts/training
conda activate iotguard
python train_ultra_advanced.py
```

### Expected Output
```
models/train_ultra_advanced_models/
├── ultra_advanced_stage1_ensemble.pkl  # Stage 1 binary classifier
├── ultra_advanced_stage2_ensemble.pkl  # Stage 2 multi-class classifier
└── model_metadata.json                 # Complete training metadata

results/train_ultra_advanced_results/
├── ULTRA_ADVANCED_RESULTS_SUMMARY.txt  # Human-readable summary
├── stage1_confusion_matrix.png         # Binary classification matrix
└── stage2_confusion_matrix.png         # Multi-class matrix
```

### Training Time Estimates
- **With 40 train + 8 val + 15 test files** (~4M samples after balancing):
  - Data loading: 5-10 minutes
  - Feature engineering: 2-3 minutes
  - Stage 1 training: 10-15 minutes (4 models)
  - Stage 2 training: 15-20 minutes (3 models)
  - **Total: ~45-60 minutes on modern CPU**

- **With GPU acceleration** (if TensorFlow GPU available):
  - Neural network training: 5x faster
  - **Total: ~30-40 minutes**

## Performance Expectations

### Stage 1 (Binary Classification)
- **Accuracy**: 99%+ expected
- **Precision**: 99%+ (few false positives)
- **Recall**: 99%+ (few false negatives)
- **F1-Score**: 99%+
- **MCC**: 0.98+ (Matthews Correlation Coefficient)

### Stage 2 (Multi-Class Classification)
- **Accuracy**: 85-90% expected
- **Precision**: 87-92% (weighted)
- **Recall**: 85-90% (weighted)
- **F1-Score**: 86-91% (weighted)

### Full Pipeline
- **Overall Accuracy**: 99%+
- **Benign Detection**: 98%+ (critical for false alarm rate)
- **Attack Detection**: 99%+ (critical for security)

## Advantages Over train_advanced.py

| Feature | train_advanced.py | train_ultra_advanced.py |
|---------|------------------|-------------------------|
| Models per stage | 3 (LGB, XGB, RF) | 4 (LGB, XGB, CB, NN) |
| Data split | Train/Test only | Train/Val/Test |
| Feature engineering | 29 features | 30+ features + polynomial |
| Balancing | SMOTE only | SMOTE-Tomek hybrid |
| Validation | Test set only | Separate validation set |
| Configuration | Hardcoded | YAML config file |
| Metadata | Basic JSON | Comprehensive JSON |
| Neural Network | No | Yes (optional) |
| CatBoost | No | Yes (optional) |
| Cross-validation | No | Configurable |

## Metadata JSON Structure

```json
{
  "model_info": {
    "name": "IoTGuard Ultra Advanced 2-Stage Ensemble",
    "version": "2.0",
    "architecture": "2-Stage: Binary + Multi-Class",
    "created_at": "2025-12-13T23:45:00",
    "training_duration": "0:45:23",
    "framework": "LightGBM + XGBoost + CatBoost + TensorFlow",
    "dataset": "CIC-IoT-2023"
  },
  "training_data": {
    "train_samples": 2500000,
    "val_samples": 400000,
    "test_samples": 600000,
    "train_files": 40,
    "val_files": 8,
    "test_files": 15
  },
  "feature_engineering": {
    "original_features": 39,
    "engineered_features": 30,
    "total_features": 69
  },
  "stage1_binary": {
    "models": ["lightgbm", "xgboost", "catboost", "neural_network"],
    "ensemble_weights": {...},
    "performance": {
      "accuracy": 0.9912,
      "precision": 0.9935,
      "recall": 0.9889,
      "f1_score": 0.9912,
      "mcc": 0.9824
    },
    "hyperparameters": {...}
  },
  "stage2_multiclass": {
    "models": ["lightgbm", "xgboost", "catboost"],
    "attack_categories": ["DDoS-DoS", "Mirai", "Reconnaissance", ...],
    "performance": {
      "accuracy": 0.8756,
      "precision": 0.8912,
      "recall": 0.8756,
      "f1_score": 0.8823
    }
  },
  "full_pipeline": {
    "accuracy": 0.9908,
    "benign_detection_rate": 0.9834,
    "attack_detection_rate": 0.9912
  }
}
```

## Troubleshooting

### Memory Issues
If you encounter out-of-memory errors:
1. Reduce `train_files` in config (e.g., from 40 to 30)
2. Reduce `attack_sampling_per_class` (e.g., from 100000 to 75000)
3. Disable polynomial features: `polynomial_features: false`

### Missing Dependencies
- **CatBoost not available**: Training will continue with 3 models (LGB, XGB, NN)
- **TensorFlow not available**: Training will continue with gradient boosting only
- Install missing packages: `pip install catboost tensorflow`

### Slow Training
- Disable neural network: `neural_network: enabled: false`
- Reduce number of estimators in config
- Use fewer training files

### GPU Acceleration
To use GPU for TensorFlow:
```bash
pip install tensorflow-gpu  # If you have CUDA-compatible GPU
```

## Tips for Best Results

1. **Use all available data**: If you have sufficient RAM (16GB+), use all 63 files
2. **Tune ensemble weights**: Adjust based on validation performance
3. **Monitor validation metrics**: Check for overfitting (train >> val performance)
4. **Feature selection**: Enable if training is too slow or accuracy plateaus
5. **Cross-validation**: Enable for more robust model selection (slower but better)

## Comparison with Original train_advanced.py

Your current `train_advanced.py` achieves:
- Stage 1: 99.1% accuracy
- Stage 2: 77.9% accuracy  
- Full: 99.1% accuracy

Expected improvements with `train_ultra_advanced.py`:
- Stage 1: 99.2-99.5% accuracy (+0.1-0.4%)
- Stage 2: 85-90% accuracy (+7-12%) ⭐ **Major improvement**
- Full: 99.2-99.5% accuracy (+0.1-0.4%)

The biggest gain is in Stage 2 multi-class classification due to:
- Better balancing (SMOTE-Tomek)
- More sophisticated models (CatBoost + Neural Network)
- Validation-based tuning
- Advanced feature engineering

## Next Steps

After training:
1. Compare `model_metadata.json` with previous results
2. Analyze confusion matrices to identify weak spots
3. Fine-tune ensemble weights if needed
4. Deploy best model to Streamlit UI
5. Test on real-world traffic data

## License & Citation

Part of the IoTGuard IDS/IPS project.
Dataset: CIC-IoT-2023 from Canadian Institute for Cybersecurity.
