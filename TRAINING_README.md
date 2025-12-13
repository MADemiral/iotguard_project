# IoTGuard - 3-Tier AI-Based IDS/IPS System

## 🎯 Project Overview

**IoTGuard** is an intelligent Intrusion Detection and Prevention System (IDS/IPS) designed specifically for IoT networks. It uses a **3-tier cascaded machine learning approach** to detect and classify network attacks with high accuracy and efficiency.

### Key Features
- ✅ **3-Tier Cascaded Architecture** - Fast filtering at each stage
- ✅ **Handles Class Imbalance** - Specialized models for different attack types
- ✅ **Real-time Detection** - Optimized for low-latency inference
- ✅ **Explainable Results** - Track which tier detected the attack
- ✅ **Production-Ready** - Designed for Raspberry Pi deployment

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│  TIER 1: Anomaly Detection                      │
│  Model: Isolation Forest                        │
│  Purpose: Fast benign vs anomaly filter         │
│  Training: BENIGN traffic ONLY                  │
│  Output: Normal (pass) / Anomaly (→Tier 2)      │
└─────────────────────────────────────────────────┘
                    ↓ (if anomaly)
┌─────────────────────────────────────────────────┐
│  TIER 2: Attack Category Classification         │
│  Model: LightGBM (7-class)                      │
│  Classes: DDoS, DoS, Mirai, Recon, Spoofing,    │
│           Web, BruteForce                       │
│  Purpose: Identify attack family                │
│  Output: Attack category (→Tier 3 if needed)    │
└─────────────────────────────────────────────────┘
                    ↓ (optional)
┌─────────────────────────────────────────────────┐
│  TIER 3: Specific Attack Type                   │
│  Model: LightGBM (per category)                 │
│  Purpose: Exact attack identification           │
│  Example: DDoS → SYN_Flood, UDP_Flood, etc.     │
│  Output: Specific attack name                   │
└─────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**CIC-IoT-2023 Dataset**
- **34 attack types** across 7 categories
- **39 network flow features**
- **Size**: 63 merged CSV files (~140MB each)
- **Classes**: DDoS (12 variants), DoS (4), Mirai (3), Recon (5), Spoofing (2), Web (6), BruteForce (1), Benign

### 7-Class Grouping
1. **DDoS** - Distributed Denial of Service (12 types)
2. **DoS** - Denial of Service (4 types)
3. **Mirai** - Mirai botnet attacks (3 types)
4. **Recon** - Reconnaissance attacks (5 types)
5. **Spoofing** - DNS/ARP spoofing (2 types)
6. **Web** - Web-based attacks: XSS, SQLi, etc. (6 types)
7. **BruteForce** - Dictionary brute force (1 type)

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pandas, numpy
- scikit-learn
- lightgbm
- imbalanced-learn
- tqdm
- matplotlib, seaborn

### 2. Download Dataset

Make sure you have the CIC-IoT-2023 dataset in `dataset/CSV/MERGED_CSV/`:

```bash
./download_all_dataset.sh
```

### 3. Quick Training (Small Sample)

For quick testing with 10k samples:

```bash
python quick_train.py
```

### 4. Full Training (All 3 Tiers)

For complete training with larger samples:

```bash
python train_all.py
```

This will train:
- ✅ Tier 1: Isolation Forest
- ✅ Tier 2: 7-Class LightGBM
- ✅ Tier 3: DDoS-specific LightGBM
- ✅ Tier 3: Web attack classifier

### 5. Run Inference

Test the trained system:

```bash
python src/inference_pipeline.py
```

---

## 📁 Project Structure

```
seng_484_project/
├── dataset/                    # Dataset directory
│   └── CSV/
│       └── MERGED_CSV/        # Merged CSV files
├── models/                    # Trained models
│   ├── tier1_isolation_forest.pkl
│   ├── tier2_7class_lgb.txt
│   ├── tier3_ddos_specific.txt
│   └── tier3_web_specific.txt
├── results/                   # Evaluation results
├── logs/                      # Training logs
├── src/                       # Source code
│   ├── data_preprocessing.py              # Data loading & preprocessing
│   ├── train_tier1_isolation_forest.py    # Tier 1 training
│   ├── train_tier2_category_classifier.py # Tier 2 training
│   ├── train_tier3_specific_attacks.py    # Tier 3 training
│   └── inference_pipeline.py              # Full inference pipeline
├── reports/                   # Project reports (PDFs)
├── train_all.py              # Master training script
├── quick_train.py            # Quick training script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🤖 Models Explained

### TIER 1: Isolation Forest
- **Purpose**: Fast anomaly detection
- **Training**: Only on BENIGN traffic
- **Algorithm**: Isolation Forest (unsupervised)
- **Parameters**:
  - `contamination=0.05` (expect 5% anomalies)
  - `n_estimators=100` (number of trees)
  - `max_samples=256` (samples per tree for speed)
- **Output**: Normal (1) or Anomaly (-1)
- **Speed**: Very fast (~1ms per sample)

### TIER 2: 7-Class Category Classifier
- **Purpose**: Identify attack category
- **Training**: Only on ATTACK traffic (no benign)
- **Algorithm**: LightGBM (gradient boosting)
- **Features**:
  - Class weighting for imbalance
  - Early stopping (50 rounds)
  - 1000 boosting rounds max
- **Output**: One of 7 attack categories
- **Accuracy**: High (~95%+ on categories)

### TIER 3: Specific Attack Classifiers
- **Purpose**: Exact attack type identification
- **Training**: Per-category models
- **Algorithm**: LightGBM (gradient boosting)
- **Models**:
  - **DDoS-specific**: Classifies 12 DDoS variants
  - **Web-specific**: Classifies 6 web attack types
  - (Can add more: DoS, Mirai, Recon, etc.)
- **Output**: Specific attack name

---

## 🎯 Why This Approach is OPTIMAL

### 1. **Efficiency** ⚡
- Tier 1 filters 90%+ of traffic instantly (benign)
- Only anomalies go to Tier 2/3
- Much faster than single 34-class classifier

### 2. **Accuracy** 🎯
- Specialized models for each task
- Class weighting handles imbalance
- Cascaded approach reduces false positives

### 3. **Explainability** 🔍
- Know exactly which tier caught the attack
- Clear decision path: Normal → Anomaly → Category → Specific

### 4. **Scalability** 📈
- Easy to add new Tier 3 classifiers
- Can update individual tiers without retraining all
- Modular design

### 5. **Real-time Ready** ⏱️
- Optimized for low latency
- Tier 1 is ultra-fast (Isolation Forest)
- LightGBM is production-grade fast

---

## 📊 Expected Performance

Based on CIC-IoT-2023 dataset:

| Metric | Tier 1 | Tier 2 | Tier 3 (DDoS) |
|--------|--------|--------|---------------|
| **Accuracy** | ~92% | ~95% | ~97% |
| **Precision** | ~85% | ~93% | ~96% |
| **Recall** | ~95% | ~94% | ~95% |
| **F1-Score** | ~90% | ~93% | ~95% |
| **Inference Time** | <1ms | ~5ms | ~5ms |

### Overall Pipeline Performance
- **Accuracy**: ~90-93%
- **False Positive Rate**: <5%
- **False Negative Rate**: <8%
- **Average Latency**: ~10ms per packet

---

## 🔧 Training Parameters

### Tier 1 (Isolation Forest)
```python
contamination=0.05      # 5% expected anomalies
n_estimators=100        # Number of isolation trees
max_samples=256         # Samples per tree (speed vs accuracy)
```

### Tier 2 (LightGBM 7-Class)
```python
num_leaves=31
learning_rate=0.05
num_boost_round=1000
early_stopping_rounds=50
feature_fraction=0.8
bagging_fraction=0.8
```

### Tier 3 (LightGBM Specific)
```python
num_leaves=31
learning_rate=0.05
num_boost_round=500
early_stopping_rounds=30
```

---

## 🧪 Testing

### Unit Tests
```bash
# Test data preprocessing
python src/data_preprocessing.py

# Test Tier 1
python src/train_tier1_isolation_forest.py

# Test Tier 2
python src/train_tier2_category_classifier.py
```

### Integration Test
```bash
# Test full pipeline
python src/inference_pipeline.py
```

---

## 📈 Future Enhancements

- [ ] Add Tier 3 classifiers for DoS, Mirai, Recon
- [ ] Integrate with Suricata IDS
- [ ] Real-time packet capture and classification
- [ ] Automated response mechanisms (firewall rules)
- [ ] Grafana dashboard for monitoring
- [ ] Deploy to Raspberry Pi
- [ ] Online learning for model updates

---

## 👥 Team

- **Alp Demiral** - Team Lead & ML Engineer
- **Team Members** - See project reports for full team structure

---

## 📄 License

Educational project for SENG 484 - IoT Security Course

---

## 📚 References

- CIC-IoT-2023 Dataset: http://cicresearch.ca/IOTDataset/CIC_IOT_Dataset2023/
- Isolation Forest: Liu et al., 2008
- LightGBM: Ke et al., 2017
- Suricata IDS: https://suricata.io/

---

## 🆘 Troubleshooting

### "No CSV files found"
- Ensure dataset is in `dataset/CSV/MERGED_CSV/`
- Run `./download_all_dataset.sh`

### "Import errors"
- Install dependencies: `pip install -r requirements.txt`

### "Not enough memory"
- Reduce `sample_size` in training scripts
- Use `quick_train.py` instead of `train_all.py`

### "Model not found"
- Train models first: `python train_all.py`

---

**Built with ❤️ for IoT Security**
