# Scripts Organization

This directory contains all scripts organized by purpose.

## Directory Structure

### `bash_scripts/`
Shell scripts for setup and utilities:
- `download_all_dataset.sh` - Download complete CIC-IoT-2023 dataset
- `download_dataset2.sh` - Alternative dataset download script

### `preprocessing/`
Data preprocessing and pipeline scripts:
- `data_preprocessing.py` - Clean and prepare network traffic data
- `inference_pipeline.py` - Real-time inference pipeline for deployment

### `training/`
Model training scripts:
- `train_advanced.py` - **Main training script** - 2-stage ensemble (LightGBM + XGBoost)
- `train_2stage.py` - Alternative 2-stage training implementation
- `train_tier1_isolation_forest.py` - Tier 1: Anomaly detection
- `train_tier2_category_classifier.py` - Tier 2: Attack categorization
- `train_tier3_specific_attacks.py` - Tier 3: Specific attack classification

## Usage

### Training the Advanced Model
```bash
cd scripts/training
python train_advanced.py
```

### Preprocessing Data
```bash
cd scripts/preprocessing
python data_preprocessing.py
```

### Download Dataset
```bash
cd scripts/bash_scripts
bash download_all_dataset.sh
```

## Configuration

Model configuration is stored in `config_train_advanced.yaml` in the root directory.
