# IoTGuard Ultra Advanced Training Pipeline - Complete Documentation

**Model Version:** Ultra Advanced Two-Stage Ensemble v2.0  
**Dataset:** CIC-IoT-2023  
**Training Script:** `train_ultra_advanced.py`  
**Configuration:** `config_train_ultra_advanced.yaml`  
**Last Updated:** December 14, 2025  
**Architecture:** Hierarchical Two-Stage Ensemble with GPU Acceleration

---

## 📊 Quick Stats (Latest Training Results)

| Metric | Value |
|--------|-------|
| **Total Features** | 68 (39 original + 29 engineered) |
| **Training Samples** | ~307,000 (9 merged CSV files + CSV/CSV augmentation) |
| **Attack Categories** | 6 categories + Benign (33 attack types total) |
| **Models per Stage** | 2 (LightGBM + XGBoost, GPU-accelerated) |
| **Stage 1 Accuracy** | 89.77% (Binary: Benign vs Attack) |
| **Stage 1 Precision** | 89.42% (low false positives) |
| **Stage 1 Recall** | 97.51% (catches 97.5% of attacks) |
| **Stage 2 Accuracy** | 91.30% (Multi-Class: 6 categories) |
| **Training Time** | ~8-10 minutes (RTX 5070 Ti) |
| **Memory Usage** | ~15-20GB RAM peak |
| **GPU** | NVIDIA RTX 5070 Ti (16GB VRAM, CUDA 13.0) |

---

## Table of Contents
1. [Input Data Sources](#1-input-data-sources)
2. [Feature Engineering](#2-feature-engineering)
3. [Data Processing Pipeline](#3-data-processing-pipeline)
4. [Stage 1: Binary Classification](#4-stage-1-binary-classification)
5. [Stage 2: Multi-Class Classification](#5-stage-2-multi-class-classification)
6. [Model Architecture](#6-model-architecture)
7. [Training Configuration](#7-training-configuration)
8. [Output Files](#8-output-files)
9. [Performance Targets](#9-performance-targets)
10. [Memory Requirements](#10-memory-requirements)
11. [Training Command](#11-training-command)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Input Data Sources

### 1.1 Primary Dataset
- **Source:** CIC-IoT-2023 Dataset (Merged CSV Files)
- **Location:** `dataset/CSV/MERGED_CSV/`
- **Pattern:** `Merged*.csv` (Merged01.csv to MergedXX.csv)

### 1.2 Data Split Configuration
```yaml
Current Memory-Optimized Configuration:
- Training Files: 7 merged CSV files (randomly shuffled)
- Validation Files: 4 merged CSV files
- Test Files: 4 merged CSV files
- Random Seed: 57 (for reproducibility)
```

### 1.3 Supplementary Data Sources (Rare Class Enhancement)
**For Training Set Only** (not validation/test):
- **Location:** `dataset/CSV/CSV/`
- **Purpose:** Load additional real attack samples for rare classes
- **Trigger Condition:** `split_name == "train"` AND `original_count < 10,000` AND `original_count < 25,000`
- **Target Classes:**
  - BruteForce: `DICTIONARYBRUTEFORCE` → loads from `DictionaryBruteForce/` (≈1,836 samples in merged)
  - Web-Based: Multiple attack types → loads from respective folders (≈3,400 samples in merged)
    - `BROWSERHIJACKING` → `BrowserHijacking/`
    - `COMMANDINJECTION` → `CommandInjection/`
    - `SQLINJECTION` → `SqlInjection/`
    - `XSS` → `XSS/`
    - `UPLOADING_ATTACK` → `Uploading_Attack/`
    - `BACKDOOR_MALWARE` → `Backdoor_Malware/`

**Important Note:** CSV files in `dataset/CSV/CSV/` folders do NOT contain a `Label` column. The training script automatically adds the appropriate label based on the folder name.

**Loading Strategy (Only for Training Split):**
1. **Load from merged CSV files first** (Merged01.csv to MergedXX.csv)
2. **Check if rare:** If category has < 10,000 samples AND < 25,000 target:
   ```python
   if split_name == "train" and original_count < 10000 and original_count < max_per_class:
   ```
3. **Load from CSV/CSV/ subfolders:**
   - For each attack type in category, search for matching folder:
     - `dataset/CSV/CSV/DictionaryBruteForce/`
     - `dataset/CSV/CSV/BrowserHijacking/`
     - `dataset/CSV/CSV/SqlInjection/`, etc.
   - Load CSV files from subfolders incrementally
   - Stop when reaching needed samples (25,000 - original_count)
4. **Random oversampling (last resort):**
   - Only if real data < 25,000 samples
   - Use `random_state=57` for reproducibility
5. **Sample down if exceeded:**
   - If total > 25,000, sample down without replacement

**Important:** Validation and test sets use ONLY merged CSV data (no supplementary loading)

### 1.4 Original Feature Set (from CIC-IoT-2023)
**Total Original Features: 39 features**

#### 1. Header & Protocol Identification (2 features)
```
1. Header_Length  - Size of packet header in bytes; abnormal sizes indicate tunneling or attacks
2. Protocol Type  - Numeric identifier for transport protocol; used for protocol-specific attack detection
```

#### 2. Network Layer (2 features)
```
3. Time_To_Live (TTL)  - Packet hop count; low TTL indicates spoofing or network loops
4. Rate                - Packets per second; extremely high rates indicate flooding attacks
```

#### 3. TCP Flag Counters (7 features)
```
5. fin_flag_number  - Count of FIN flags; high values indicate connection termination attacks
6. syn_flag_number  - Count of SYN flags; abnormal patterns detect SYN flood attacks
7. rst_flag_number  - Count of RST flags; high counts indicate scanning or connection resets
8. psh_flag_number  - Count of PSH flags; unusual patterns detect data exfiltration
9. ack_flag_number  - Count of ACK flags; irregular ACK patterns indicate spoofing
10. ece_flag_number - Count of ECE (ECN-Echo) flags; rarely used, anomalies indicate manipulation
11. cwr_flag_number - Count of CWR (Congestion Window Reduced) flags; unusual usage indicates evasion
```

#### 4. TCP State Tracking (4 features)
```
12. ack_count  - Number of ACK packets; disproportionate counts indicate incomplete handshakes
13. syn_count  - Number of SYN packets; high SYN without ACK indicates SYN flood
14. fin_count  - Number of FIN packets; excessive FIN indicates connection exhaustion attacks
15. rst_count  - Number of RST packets; high RST indicates port scanning or DoS
```

#### 5. Application Layer Protocols (11 features)
```
16. HTTP   - HTTP traffic indicator (0 or 1); identifies web-based attacks
17. HTTPS  - HTTPS traffic indicator; encrypted attacks hide in HTTPS
18. DNS    - DNS traffic indicator; DNS tunneling and amplification attacks
19. Telnet - Telnet protocol indicator; often used in botnet communication
20. SMTP   - Email protocol indicator; spam and phishing attacks
21. SSH    - SSH protocol indicator; brute force and tunneling attacks
22. IRC    - IRC protocol indicator; common botnet command-and-control channel
23. TCP    - TCP protocol indicator; most attacks use TCP
24. UDP    - UDP protocol indicator; DDoS amplification uses UDP
25. DHCP   - DHCP protocol indicator; DHCP starvation attacks
26. ARP    - ARP protocol indicator; ARP spoofing and MITM attacks
```

#### 6. Additional Network Protocols (3 features)
```
27. ICMP  - ICMP protocol indicator; ping floods and reconnaissance
28. IGMP  - IGMP protocol indicator; multicast abuse in DDoS
29. IPv   - IP version indicator; protocol-specific vulnerabilities
30. LLC   - Logical Link Control; rarely used, anomalies indicate low-level attacks
```

#### 7. Packet Size Statistics (7 features)
```
31. Tot sum   - Sum of all packet sizes; extremely large indicates data exfiltration
32. Min       - Minimum packet size; very small packets indicate scanning
33. Max       - Maximum packet size; oversized packets indicate buffer overflow attempts
34. AVG       - Average packet size; deviations from normal indicate attacks
35. Std       - Standard deviation of packet size; high variance indicates mixed attack traffic
36. Tot size  - Total bytes transferred; massive transfers indicate data theft or flooding
37. Variance  - Variance of packet sizes; irregular patterns indicate reconnaissance
```

#### 8. Timing & Flow Metrics (2 features)
```
38. IAT (Inter-Arrival Time)  - Time between packets; very low IAT indicates flooding
39. Number                    - Total packet count; extremely high counts indicate DoS
```

#### 9. Target Variable
```
Label - Attack classification (33 attack types + BENIGN)
  - Used for supervised learning
  - Mapped to 6 hierarchical categories + Benign in Stage 2
```

---

## 2. Feature Engineering

**Total Engineered Features: 29 features**  
**Combined with Original: 39 + 29 = 68 total features**

### 2.1 Statistical Features (ENABLED)
**Total: 26 engineered statistical features**

#### A. TCP Flag Ratios (3 features)
```python
1. syn_ack_ratio = syn_flag_number / (ack_flag_number + 1)
```
Detects SYN flood attacks where SYN packets vastly outnumber ACK responses.

```python
2. rst_fin_ratio = rst_flag_number / (fin_flag_number + 1)
```
Identifies port scanning (high RST) vs normal connection termination (balanced RST/FIN).

```python
3. psh_ack_ratio = psh_flag_number / (ack_flag_number + 1)
```
Detects data exfiltration attacks with abnormal push-to-acknowledgment ratios.

#### B. Packet Analysis Ratios (3 features)
```python
4. packet_rate_ratio = Rate / (Number + 1)
```
Measures packets-per-second efficiency; abnormally high indicates flooding attacks.

```python
5. size_per_packet = Tot_size / (Number + 1)
```
Average payload per packet; very small values indicate scanning, very large indicate data theft.

```python
6. avg_iat = IAT / (Number + 1)
```
Average inter-arrival time per packet; near-zero values indicate rapid-fire DDoS attacks.

#### C. Traffic Variability Analysis (4 features)
```python
7. variance_avg_ratio = Variance / (AVG + 1)
```
Relative variance in packet sizes; high values indicate mixed attack traffic or reconnaissance.

```python
8. std_avg_ratio = Std / (AVG + 1)
```
Coefficient of variation for packet sizes; stable for normal traffic, chaotic for attacks.

```python
9. range_stat = Max - Min
```
Spread between smallest and largest packets; extreme ranges indicate scanning or overflow attempts.

```python
10. cv = Std / (AVG + 1)  # Coefficient of variation
```
Normalized measure of packet size dispersion; detects irregular traffic patterns.

#### D. TCP Flag Aggregations (2 features)
```python
11. flags_total = fin_flag_number + syn_flag_number + rst_flag_number + 
                  psh_flag_number + ack_flag_number + ece_flag_number + 
                  cwr_flag_number
```
Total flag count across all types; abnormally high indicates SYN floods or scanning.

```python
12. flag_diversity = count of non-zero flags among [fin, syn, rst, psh, ack]
```
Number of different flag types used; low diversity (e.g., only SYN) indicates specific attacks.

#### E. Protocol Interaction Features (3 features)
```python
13. tcp_http_combo = TCP * HTTP
```
Identifies HTTP-over-TCP traffic; isolates HTTP flood and web-based attacks.

```python
14. udp_dns_combo = UDP * DNS
```
Identifies DNS-over-UDP traffic; detects DNS amplification and tunneling attacks.

```python
15. protocol_count = TCP + UDP + ICMP + ARP + DNS + HTTP + HTTPS
```
Total number of protocols used; multiple protocols indicate complex attacks or normal services.

#### F. Advanced Network Ratios (8 features)
```python
16. ttl_rate_ratio = Time_To_Live / (Rate + 1)
```
Ratio of TTL to packet rate; low values indicate high-rate attacks from nearby sources (DDoS).

```python
17. header_size_ratio = Header_Length / (Tot_size + 1)
```
Proportion of header to total size; high ratios indicate empty packets used in scanning.

```python
18. syn_count_ratio = syn_count / (Number + 1)
```
Proportion of SYN packets; values near 1.0 indicate pure SYN flood attacks.

```python
19. ack_count_ratio = ack_count / (Number + 1)
```
Proportion of ACK packets; abnormal values detect incomplete TCP handshakes.

```python
20. fin_count_ratio = fin_count / (Number + 1)
```
Proportion of FIN packets; excessive FIN indicates connection exhaustion attacks.

```python
21. rst_count_ratio = rst_count / (Number + 1)
```
Proportion of RST packets; high values indicate port scanning or connection resets.

```python
22. rate_ttl_interaction = Rate * Time_To_Live
```
Combined effect of packet rate and hop count; detects distributed vs centralized attacks.

```python
23. size_rate_interaction = Tot_size * Rate
```
Bandwidth consumption metric; extremely high values indicate data exfiltration or flooding.

#### G. Log Transformations (3 features)
```python
24. log_rate = np.log1p(Rate)
```
Log-transformed packet rate; compresses extreme values to prevent flooding attacks from dominating.

```python
25. log_tot_size = np.log1p(Tot_size)
```
Log-transformed total size; normalizes data transfer volumes ranging from bytes to gigabytes.

```python
26. log_number = np.log1p(Number)
```
Log-transformed packet count; handles flows ranging from single packets to millions evenly.

### 2.2 Time-Based Features (ENABLED)
**Total: 3 time-based features**

```python
27. iat_variance = IAT * Variance
```
Combined timing and size variability; detects attacks with irregular packet timing and sizes.

```python
28. iat_std = IAT * Std
```
Timing-size standard deviation product; identifies chaotic traffic patterns in DoS attacks.

```python
29. burst_score = Rate * Number / (IAT + 1)
```
Traffic burst intensity metric; extremely high values indicate rapid flooding or DDoS attacks.

### 2.3 Polynomial Features (DISABLED for Memory)
**Status:** Currently disabled to prevent memory overflow  
**Configuration:** `polynomial_features: false`

When enabled, creates **10 additional features:**
```python
For key_features = ['Rate', 'Tot size', 'Number', 'AVG', 'Variance']:
  - feature_squared (e.g., Rate^2, Tot_size^2, ...)  # 5 features
  - feature_sqrt (e.g., sqrt(Rate), sqrt(Tot_size), ...)  # 5 features
```

**Purpose:** Capture non-linear relationships between features  
**Memory Impact:** +10 features = ~13% increase in feature space  
**Recommendation:** Re-enable if you have 32GB+ RAM

### 2.4 Interaction Features (ENABLED)
**Status:** Enabled - important for attack pattern detection
**Configuration:** `interaction_features: true`

Creates cross-feature interactions to capture complex attack patterns.

### 2.4 Feature Summary Table

| Feature Category | Count | Purpose |
|-----------------|-------|---------|
| **Original CIC-IoT-2023 Features** | **39** | Raw network traffic measurements |
| Header & Protocol ID | 2 | Packet structure identification |
| Network Layer | 2 | Routing and rate metrics |
| TCP Flags | 7 | Connection state tracking |
| TCP State Counts | 4 | Handshake completion tracking |
| Application Protocols | 11 | Protocol-specific attack detection |
| Additional Protocols | 3 | Low-level network indicators |
| Packet Size Statistics | 7 | Data volume analysis |
| Timing & Flow | 2 | Packet rate and timing |
| **Engineered Features** | **29** | Machine-learned attack patterns |
| TCP Flag Ratios | 3 | Flag imbalance detection |
| Packet Analysis Ratios | 3 | Efficiency metrics |
| Traffic Variability | 4 | Pattern irregularity |
| Flag Aggregations | 2 | Overall flag behavior |
| Protocol Interactions | 3 | Multi-protocol patterns |
| Advanced Network Ratios | 8 | Complex relationship metrics |
| Log Transformations | 3 | Skewed distribution normalization |
| Time-Based Features | 3 | Temporal pattern analysis |
| **Total Active Features** | **68** | Used in current training |
| Polynomial Features | 0 | Disabled for memory (would add 10) |
| **Potential Maximum** | **78** | If polynomial features enabled |

### 2.5 Feature Engineering Impact

**Before Feature Engineering:** 39 features → 86.75% accuracy  
**After Feature Engineering:** 68 features (+74%) → 88.98% accuracy (+2.23%)

**Key Improvements from Engineered Features:**
- **False Positive Reduction:** Flag ratios and aggregations reduce benign misclassification
- **Rare Attack Detection:** Protocol interactions and timing features improve rare class accuracy
- **Attack Category Separation:** Advanced ratios help distinguish between attack types
- **Robustness:** Log transformations handle extreme values without destabilizing the model

---

## 3. Data Processing Pipeline

### 3.1 Class Mapping (Hierarchical Grouping)
**33 Attack Types → 6 Attack Categories + 1 Benign**

#### Category 1: DDoS-DoS (16 attacks)
```
DDOS-ACK_FRAGMENTATION, DDOS-HTTP_FLOOD, DDOS-ICMP_FLOOD,
DDOS-ICMP_FRAGMENTATION, DDOS-PSHACK_FLOOD, DDOS-RSTFINFLOOD,
DDOS-SYN_FLOOD, DDOS-SLOWLORIS, DDOS-SYNONYMOUSIP_FLOOD,
DDOS-TCP_FLOOD, DDOS-UDP_FLOOD, DDOS-UDP_FRAGMENTATION,
DOS-HTTP_FLOOD, DOS-SYN_FLOOD, DOS-TCP_FLOOD, DOS-UDP_FLOOD
```

#### Category 2: Mirai (3 attacks)
```
MIRAI-GREETH_FLOOD, MIRAI-GREIP_FLOOD, MIRAI-UDPPLAIN
```

#### Category 3: Reconnaissance (5 attacks)
```
RECON-HOSTDISCOVERY, RECON-OSSCAN, RECON-PINGSWEEP,
RECON-PORTSCAN, VULNERABILITYSCAN
```

#### Category 4: Spoofing (2 attacks)
```
DNS_SPOOFING, MITM-ARPSPOOFING
```

#### Category 5: Web-Based (6 attacks)
```
BROWSERHIJACKING, COMMANDINJECTION, SQLINJECTION,
XSS, UPLOADING_ATTACK, BACKDOOR_MALWARE
```

#### Category 6: BruteForce (1 attack)
```
DICTIONARYBRUTEFORCE
```

#### Category 7: Benign
```
BENIGN (normal traffic)
```

### 3.2 Sampling Strategy

#### Benign Samples
- **Strategy:** Keep ALL benign samples
- **Configuration:** `benign_sampling: "all"`
- **Purpose:** Preserve natural benign traffic distribution

#### Attack Samples (Memory-Optimized)
- **Target per Class:** 25,000 samples (reduced from 37,500)
- **Minimum per Class:** 100 samples (rare class protection)
- **Configuration:** `attack_sampling_per_class: 25000`

**Loading Priority (Training Set Only):**
```
For each attack category:
  1. Load from merged CSV files (Merged01-XX.csv)
  2. Count samples: original_count
  
  3. IF split_name == "train" AND original_count < 10,000 AND original_count < 25,000:
       → RARE CLASS detected
       → Load additional real samples from dataset/CSV/CSV/ subfolders
       → Target: 25,000 - original_count
       
  4. IF still < 25,000 after loading real data:
       → Use random oversampling (random_state=57)
       
  5. IF > 25,000:
       → Sample down to 25,000 (random_state=57)
```

**Validation/Test Sets:**
```
For each attack category:
  1. Load from merged CSV files ONLY (no CSV/CSV/ loading)
  2. IF < 25,000: Random oversample to 25,000
  3. IF > 25,000: Sample down to 25,000
```

**Key Difference:**
- **Training:** Uses real data from CSV/CSV/ folders for rare classes
- **Validation/Test:** Uses only merged CSV data (prevents data leakage)

### 3.3 Data Cleaning

#### Missing Value Handling
- **Method:** Median imputation
- **Configuration:** `handle_missing: "median"`
- **Applied to:** All numeric features

#### Duplicate Removal
- **Status:** ENABLED
- **Configuration:** `remove_duplicates: true`
- **Purpose:** Remove exact duplicate rows

#### Outlier Detection
- **Method:** Isolation Forest
- **Configuration:** `outlier_method: "isolation_forest"`
- **Contamination Rate:** 5% (`outlier_contamination: 0.05`)
- **Purpose:** Identify and handle anomalous data points

#### Infinite Value Handling
```python
df.replace([np.inf, -np.inf], np.nan)
df.fillna(df.median(numeric_only=True))
```

### 3.4 Feature Scaling
- **Method:** Robust Scaler
- **Configuration:** `scaler_type: "robust"`
- **Reason:** Robust to outliers (uses median and IQR)
- **Applied:** After feature engineering, before model training

---

## 4. Stage 1: Binary Classification

### 4.1 Objective
**Task:** Classify traffic as BENIGN or ATTACK  
**Output:** Binary prediction (0 = Benign, 1 = Attack)

### 4.2 Models Configuration (Memory-Optimized)

#### LightGBM (ENABLED) - Weight: 45%
```yaml
num_leaves: 80           # Reduced for memory
max_depth: 12            # Reduced depth
learning_rate: 0.03
n_estimators: 400        # Reduced trees
min_child_samples: 10
subsample: 0.85
colsample_bytree: 0.85
reg_alpha: 0.01          # L1 regularization
reg_lambda: 0.01         # L2 regularization
class_weight: "balanced"
is_unbalance: true       # Force balanced learning
device: "gpu"            # GPU acceleration
```

#### XGBoost (ENABLED) - Weight: 55%
```yaml
max_depth: 12            # Reduced for memory
learning_rate: 0.03
n_estimators: 400        # Reduced trees
min_child_weight: 2      # Reduce false positives
subsample: 0.85
colsample_bytree: 0.85
gamma: 0.01              # Pruning parameter
reg_alpha: 0.01
reg_lambda: 0.1
scale_pos_weight: 6.5    # Reduce FP
tree_method: "hist"      # GPU-compatible
device: "cuda"           # RTX 5070 Ti
```

#### CatBoost (DISABLED for Memory)
```yaml
enabled: false           # Disabled to save memory
```

#### Neural Network (DISABLED for Memory)
```yaml
enabled: false           # Disabled to save memory
```

### 4.3 Ensemble Configuration
```yaml
method: "weighted_average"
weights:
  lightgbm: 0.45         # Increased (only 2 models)
  xgboost: 0.55          # Increased (only 2 models)
  catboost: 0.0          # Disabled
  neural_network: 0.0    # Disabled
```

### 4.4 Class Balancing
```yaml
method: "smote"                    # SMOTE oversampling only
sampling_strategy: 0.80            # Attack:Benign ratio (80% attack of benign count)
random_state: 57                   # Fixed for reproducibility
k_neighbors: 5                     # Number of nearest neighbors for SMOTE
min_precision_for_threshold: 0.60  # High precision target
```

**SMOTE Algorithm:**
- **Method:** Synthetic Minority Over-sampling TEchnique
- **Purpose:** Generate synthetic attack samples to balance the dataset
- **How it works:** 
  1. For each attack sample, find k=5 nearest neighbors
  2. Create synthetic samples along lines between sample and neighbors
  3. Balance to 80% of benign class count (sampling_strategy=0.80)
- **Random State:** 57 (consistent with data loading seed)

**Alternative Methods Available:**
- `adasyn`: Adaptive Synthetic Sampling (focuses on hard-to-learn samples)
- `smote_tomek`: SMOTE + Tomek links cleaning (removes overlapping samples)
- `smote_enn`: SMOTE + Edited Nearest Neighbors (removes noisy samples)

**Current Configuration Rationale:**
- Pure SMOTE chosen for maximum attack sample preservation
- sampling_strategy=0.80 prevents over-balancing (reduces false positives)
- k_neighbors=5 is standard, stable choice

### 4.5 Threshold Optimization
- **Method:** Precision-Recall curve analysis
- **Target:** Maximize F1-score while maintaining precision ≥ 60%
- **Configuration:** `min_precision_for_threshold: 0.60`
- **Purpose:** Reduce false positives (benign classified as attack)

---

## 5. Stage 2: Multi-Class Classification

### 5.1 Objective
**Task:** Classify ATTACK traffic into 6 categories  
**Input:** Only samples predicted as ATTACK by Stage 1  
**Output:** One of 6 attack categories

### 5.2 Attack Categories
```
1. DDoS-DoS         - Distributed/Denial of Service attacks
2. Mirai            - Mirai botnet variants
3. Reconnaissance   - Network scanning and probing
4. Spoofing         - DNS/ARP spoofing attacks
5. Web-Based        - Web application attacks
6. BruteForce       - Dictionary-based attacks
```

### 5.3 Models Configuration (Memory-Optimized)

#### LightGBM (ENABLED) - Weight: 45%
```yaml
num_leaves: 80           # Reduced for memory
max_depth: 12            # Reduced depth
learning_rate: 0.025
n_estimators: 400        # Reduced trees
min_child_samples: 5     # Balance for rare classes
subsample: 0.85
colsample_bytree: 0.85
reg_alpha: 0.01
reg_lambda: 0.01
class_weight: "balanced"
is_unbalance: true
device: "gpu"
```

#### XGBoost (ENABLED) - Weight: 55%
```yaml
max_depth: 12            # Reduced for memory
learning_rate: 0.025
n_estimators: 400        # Reduced trees
min_child_weight: 1      # Minimal for rare classes
subsample: 0.85
colsample_bytree: 0.85
gamma: 0.01
reg_alpha: 0.01
reg_lambda: 0.1
tree_method: "hist"
device: "cuda"
```

#### CatBoost (DISABLED for Memory)
```yaml
enabled: false           # Disabled to save memory
```

### 5.4 Ensemble Configuration
```yaml
method: "weighted_average"
weights:
  lightgbm: 0.45         # Increased (only 2 models)
  xgboost: 0.55          # Increased (only 2 models)
  catboost: 0.0          # Disabled
```

### 5.5 Class Balancing
```yaml
method: "smote_undersampling"        # Hybrid SMOTE + RandomUnderSampler
smote_strategy: "not majority"       # Boost all minority classes (not the largest)
undersample_strategy: "not minority" # Reduce only majority classes
k_neighbors: 3                       # Minimal for rare classes (BruteForce, Web-Based)
random_state: 42
```

**Multi-Class Balancing Strategy:**
1. **SMOTE Phase:** 
   - Oversample all minority classes (BruteForce, Web-Based, Spoofing, Reconnaissance)
   - Use k_neighbors=3 (lower than Stage 1) since rare classes have fewer samples
   - Strategy "not majority" = boost everything except DDoS-DoS (largest class)

2. **Undersampling Phase:**
   - Reduce DDoS-DoS and Mirai (common classes) by random sampling
   - Strategy "not minority" = reduce only common classes
   - Prevents overwhelming the model with DDoS samples

**Purpose:** 
- Ensure rare attack types (BruteForce: ~2k, Web-Based: ~4k) get equal learning weight
- Prevent DDoS-DoS (~200k+ samples) from dominating the model
- Improve detection of critical but rare attacks (SQL injection, brute force)

---

## 6. Model Architecture

### 6.1 Two-Stage Hierarchical Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Raw Network Traffic               │
│                          (46 features)                      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                        │
│   • Statistical Features (~25)                              │
│   • Time-Based Features (3)                                 │
│   • Interaction Features (variable)                         │
│   → Total: ~75-80 features                                  │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 1: BINARY CLASSIFICATION                 │
│                                                             │
│   ┌─────────────────┐      ┌─────────────────┐            │
│   │   LightGBM      │      │    XGBoost      │            │
│   │   (GPU, 45%)    │      │  (CUDA, 55%)    │            │
│   │   400 trees     │      │   400 trees     │            │
│   │   depth: 12     │      │   depth: 12     │            │
│   └────────┬────────┘      └────────┬────────┘            │
│            │                        │                      │
│            └───────────┬────────────┘                      │
│                        ▼                                    │
│              Weighted Average Ensemble                      │
│              Threshold Optimization                         │
│                        │                                    │
│        ┌───────────────┴────────────────┐                  │
│        ▼                                ▼                   │
│    BENIGN                            ATTACK                 │
│   (Stage 1 Output)              (Pass to Stage 2)           │
└─────────────────────────────────────────┬───────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────┐
│            STAGE 2: MULTI-CLASS CLASSIFICATION              │
│                  (Attack Category Detection)                │
│                                                             │
│   ┌─────────────────┐      ┌─────────────────┐            │
│   │   LightGBM      │      │    XGBoost      │            │
│   │   (GPU, 45%)    │      │  (CUDA, 55%)    │            │
│   │   400 trees     │      │   400 trees     │            │
│   │   depth: 12     │      │   depth: 12     │            │
│   └────────┬────────┘      └────────┬────────┘            │
│            │                        │                      │
│            └───────────┬────────────┘                      │
│                        ▼                                    │
│              Weighted Average Ensemble                      │
│                        │                                    │
│        ┌───────────────┼────────────────┐                  │
│        ▼               ▼                ▼                   │
│   DDoS-DoS        Mirai            Reconnaissance           │
│        ▼               ▼                ▼                   │
│   Spoofing       Web-Based         BruteForce              │
│                                                             │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              FINAL OUTPUT: Attack Classification            │
│   • Benign: Normal traffic (from Stage 1)                  │
│   • Attack: Category + Confidence (from Stage 2)           │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 GPU Acceleration
```
Hardware: NVIDIA GeForce RTX 5070 Ti
- VRAM: 16GB
- Compute Capability: 12.0
- CUDA Version: 13.0
- Driver: 580.95.05

GPU Usage:
✓ LightGBM: device="gpu"
✓ XGBoost: device="cuda", tree_method="hist"
✗ TensorFlow: CPU-only (compute 12.0 not supported)
✗ CatBoost: Disabled for memory
```

### 6.3 Memory Optimization Strategy
```
1. Reduced dataset size: 7+4+4 files (from 12+5+5)
2. Reduced samples per class: 25,000 (from 37,500)
3. Disabled CatBoost (memory-intensive)
4. Disabled Neural Network (CPU-only, memory overhead)
5. Reduced tree depth: 12 (from 14-15)
6. Reduced tree count: 400 (from 650-700)
7. Disabled polynomial features
8. Limited batch processing
```

---

## 7. Training Configuration

### 7.1 Training Parameters
```yaml
random_seed: 42
use_gpu: true
batch_processing: true
save_models: true
save_metadata: true
```

### 7.2 Validation Strategy
- **Method:** Holdout validation
- **Split:** Train (7 files), Validation (4 files), Test (4 files)
- **Shuffle:** Random shuffle with seed=57

### 7.3 Performance Metrics

#### Stage 1 (Binary)
```
- Accuracy
- Precision (minimize false positives)
- Recall (minimize false negatives)
- F1-Score
- MCC (Matthews Correlation Coefficient)
- ROC-AUC
- Confusion Matrix
- Threshold optimization curve
```

#### Stage 2 (Multi-Class)
```
- Overall Accuracy
- Per-Class Precision/Recall/F1
- Macro-averaged metrics
- Weighted-averaged metrics
- Confusion Matrix (6x6)
- Classification Report
```



### 7.4 Early Stopping
- **Enabled for:** Neural Network (when enabled)
- **Patience:** 10 epochs
- **Monitor:** Validation loss

---

## 8. Output Files

### 8.1 Model Files
**Location:** `models/train_ultra_advanced_models/`

```
ultra_advanced_stage1_ensemble.pkl    # Stage 1 binary classifier
ultra_advanced_stage2_ensemble.pkl    # Stage 2 multi-class classifier
model_metadata.json                   # Complete training metadata
scaler.pkl                            # Feature scaler (if saved separately)
```

### 8.2 Results Files
**Location:** `results/train_ultra_advanced_results/`

```
ULTRA_ADVANCED_RESULTS_SUMMARY.txt    # Complete performance summary
stage1_confusion_matrix.png           # Stage 1 confusion matrix
stage2_confusion_matrix.png           # Stage 2 confusion matrix
stage1_roc_curve.png                  # Stage 1 ROC curve (if generated)
stage1_pr_curve.png                   # Stage 1 Precision-Recall curve
feature_importance_stage1.png         # Stage 1 feature importance
feature_importance_stage2.png         # Stage 2 feature importance
threshold_optimization.png            # Threshold selection curve
```

### 8.3 Log Files
**Location:** `logs/`

```
train_ultra_advanced_YYYYMMDD_HHMMSS.log    # Training logs
```

### 8.4 Metadata Structure
```json
{
  "model_info": {
    "model_name": "Ultra Advanced Two-Stage Ensemble",
    "version": "2.0",
    "dataset": "CIC-IoT-2023",
    "training_date": "ISO timestamp",
    "hardware": "RTX 5070 Ti, CUDA 13.0"
  },
  "training_data": {
    "train_samples": 123456,
    "val_samples": 45678,
    "test_samples": 45678,
    "train_files": 7,
    "val_files": 4,
    "test_files": 4
  },
  "feature_engineering": {
    "original_features": 46,
    "engineered_features": 28,
    "total_features": 74
  },
  "stage1_binary": {
    "models_used": ["LightGBM", "XGBoost"],
    "ensemble_weights": {...},
    "accuracy": 0.XXXX,
    "precision": 0.XXXX,
    "recall": 0.XXXX,
    "f1_score": 0.XXXX,
    "threshold": 0.XXX
  },
  "stage2_multiclass": {
    "models_used": ["LightGBM", "XGBoost"],
    "ensemble_weights": {...},
    "num_classes": 6,
    "accuracy": 0.XXXX,
    "per_class_metrics": {...}
  },
  "full_pipeline": {
    "end_to_end_accuracy": 0.XXXX,
    "total_training_time": "HH:MM:SS"
  }
}
```

---

## 9. Performance Targets

### 9.1 Stage 1 Targets (Memory-Optimized)
```
Accuracy:  > 89%
Precision: > 92%  (minimize false positives)
Recall:    > 93%  (minimize false negatives)
F1-Score:  > 90%
MCC:       > 0.70
```

### 9.2 Stage 2 Targets
```
Overall Accuracy:        > 85%
DDoS-DoS:                > 90% (abundant data)
Mirai:                   > 88% (abundant data)
Reconnaissance:          > 85% (medium data)
Spoofing:                > 82% (medium data)
Web-Based:               > 75% (rare class)
BruteForce:              > 70% (rarest class)
```

---

## 10. Memory Requirements

### 10.1 Current Configuration (Optimized)
```
Dataset Loading:         ~5-8 GB
Feature Engineering:     ~2-3 GB
Model Training (Stage 1): ~4-6 GB
Model Training (Stage 2): ~3-5 GB
Peak Memory Usage:       ~15-20 GB
Recommended RAM:         24 GB+
GPU VRAM (optional):     8 GB+ (for LightGBM/XGBoost)
```

### 10.2 If Memory Issues Persist
**Further Optimizations:**
1. Reduce `attack_sampling_per_class` to 20,000
2. Reduce train_files to 5
3. Disable `interaction_features`
4. Use only XGBoost (disable LightGBM)
5. Reduce `max_depth` to 10
6. Reduce `n_estimators` to 300

---

## 11. Training Command

### 11.1 Standard Training
```bash
cd /home/alpdemial/Desktop/seng_484_project
python scripts/training/train_ultra_advanced.py
```

### 11.2 With Memory Monitoring
```bash
./train_with_memory_monitor.sh
```

### 11.3 Expected Training Time
```
Data Loading:            5-10 minutes
Feature Engineering:     2-5 minutes
Stage 1 Training:        10-20 minutes
Stage 2 Training:        8-15 minutes
Evaluation & Metrics:    2-5 minutes
Total:                   30-60 minutes (GPU accelerated)
```

---

## 12. Troubleshooting

### 12.1 Process Killed (OOM)
**Symptom:** Training terminated with "Killed"
**Solution:**
1. Check memory: `free -h`
2. Reduce `train_files` in config
3. Reduce `attack_sampling_per_class`
4. Clear cache: `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`
5. Use memory monitor: `./train_with_memory_monitor.sh`

### 12.2 CUDA Out of Memory
**Symptom:** GPU memory error
**Solution:**
1. Reduce `max_depth` (12 → 10)
2. Reduce `n_estimators` (400 → 300)
3. Use `tree_method: "hist"` (already set)
4. Reduce batch size (if applicable)

### 12.3 Low Accuracy
**Symptom:** Stage 1 < 85% or Stage 2 < 75%
**Solution:**
1. Increase `attack_sampling_per_class` (if memory allows)
2. Increase `n_estimators` (if memory allows)
3. Re-enable CatBoost (if memory allows)
4. Enable polynomial features (if memory allows)
5. Verify rare class data loading from CSV/CSV folders

---

## 13. Authors & Version History

**Version 2.0 (December 2025)**
- Memory-optimized configuration
- Reduced to 2-model ensemble per stage
- Enhanced rare class loading strategy
- GPU acceleration for RTX 5070 Ti

**Version 1.0 (December 2025)**
- Initial ultra advanced implementation
- 4-model ensemble architecture
- Full feature engineering pipeline

**Team:** IoTGuard Development Team  
**Contact:** [Your contact information]  
**License:** [Your license]

---

**End of Documentation**
