"""
IoTGuard - Data Preprocessing Module
Handles loading, cleaning, and feature engineering for CIC-IoT-2023 dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import glob
import os
from typing import Tuple, List, Dict
import pickle
from tqdm import tqdm


class IoTDataPreprocessor:
    """Preprocessor for CIC-IoT-2023 dataset"""
    
    def __init__(self, dataset_path: str = "dataset/CSV/MERGED_CSV"):
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Define 7-class grouping (from your report)
        self.class_mapping = {
            'DDoS': [
                'DDOS-RSTFINFLOOD', 'DDOS-PSHACK_FLOOD', 'DDOS-SYN_FLOOD',
                'DDOS-UDP_FLOOD', 'DDOS-TCP_FLOOD', 'DDOS-ICMP_FLOOD',
                'DDOS-SYNONYMOUSIP_FLOOD', 'DDOS-ACK_FRAGMENTATION',
                'DDOS-UDP_FRAGMENTATION', 'DDOS-ICMP_FRAGMENTATION',
                'DDOS-SLOWLORIS', 'DDOS-HTTP_FLOOD'
            ],
            'DoS': [
                'DOS-UDP_FLOOD', 'DOS-SYN_FLOOD', 'DOS-TCP_FLOOD', 'DOS-HTTP_FLOOD'
            ],
            'Mirai': [
                'MIRAI-GREETH_FLOOD', 'MIRAI-GREIP_FLOOD', 'MIRAI-UDPPLAIN'
            ],
            'Recon': [
                'RECON-PINGSWEEP', 'RECON-OSSCAN', 'RECON-PORTSCAN',
                'VULNERABILITYSCAN', 'RECON-HOSTDISCOVERY'
            ],
            'Spoofing': [
                'DNS_SPOOFING', 'MITM-ARPSPOOFING'
            ],
            'Web': [
                'BROWSERHIJACKING', 'BACKDOOR_MALWARE', 'XSS',
                'UPLOADING_ATTACK', 'SQLINJECTION', 'COMMANDINJECTION'
            ],
            'BruteForce': [
                'DICTIONARYBRUTEFORCE'
            ],
            'Benign': [
                'BENIGN'
            ]
        }
        
        # Reverse mapping for quick lookup
        self.label_to_group = {}
        for group, labels in self.class_mapping.items():
            for label in labels:
                self.label_to_group[label] = group
    
    def load_data(self, sample_size: int = None, use_streaming: bool = False) -> pd.DataFrame:
        """Load CSV files from dataset"""
        csv_files = sorted(glob.glob(os.path.join(self.dataset_path, "*.csv")))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.dataset_path}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        if use_streaming:
            return self._load_streaming(csv_files, sample_size)
        else:
            return self._load_batch(csv_files, sample_size)
    
    def _load_batch(self, csv_files: List[str], sample_size: int = None) -> pd.DataFrame:
        """Load all data at once"""
        dfs = []
        
        for file in tqdm(csv_files, desc="Loading CSV files"):
            try:
                df = pd.read_csv(file)
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        data = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(data):,} total samples")
        return data
    
    def _load_streaming(self, csv_files: List[str], sample_size: int = None):
        """Generator for streaming data loading"""
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                if sample_size and len(df) > sample_size:
                    df = df.sample(n=sample_size, random_state=42)
                yield df
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Handle missing values with median (for numerical columns)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        
        # Remove zero-variance features
        variance = df[numerical_cols].var()
        zero_var_cols = variance[variance == 0].index.tolist()
        if zero_var_cols:
            print(f"Removing {len(zero_var_cols)} zero-variance columns")
            df = df.drop(columns=zero_var_cols)
        
        return df
    
    def extract_features_labels(self, df: pd.DataFrame, 
                               label_type: str = '7class') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and labels
        
        Args:
            label_type: '34class', '7class', or 'binary'
        """
        # Separate features and labels
        if 'Label' in df.columns:
            # These are the ACTUAL 39 features in your CSV files:
            # Header_Length, Protocol Type, Time_To_Live, Rate, 
            # fin_flag_number, syn_flag_number, rst_flag_number, psh_flag_number,
            # ack_flag_number, ece_flag_number, cwr_flag_number,
            # ack_count, syn_count, fin_count, rst_count,
            # HTTP, HTTPS, DNS, Telnet, SMTP, SSH, IRC, TCP, UDP, DHCP, ARP, ICMP, IGMP, IPv, LLC,
            # Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance
            X = df.drop('Label', axis=1)
            y_raw = df['Label']
        else:
            raise ValueError("'Label' column not found in dataset")
        
        # Convert labels based on type
        if label_type == 'binary':
            y = y_raw.apply(lambda x: 0 if x == 'BENIGN' else 1)
        elif label_type == '7class':
            y = y_raw.apply(lambda x: self.label_to_group.get(x, 'Unknown'))
        else:  # 34class
            y = y_raw
        
        return X, y
    
    def get_benign_only(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract only benign traffic for Isolation Forest"""
        return df[df['Label'] == 'BENIGN'].drop('Label', axis=1)
    
    def get_attack_only(self, df: pd.DataFrame, attack_group: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract only attack traffic, optionally filtered by group"""
        attack_df = df[df['Label'] != 'BENIGN'].copy()
        
        if attack_group:
            attack_labels = self.class_mapping.get(attack_group, [])
            attack_df = attack_df[attack_df['Label'].isin(attack_labels)]
        
        X = attack_df.drop('Label', axis=1)
        y = attack_df['Label']
        
        return X, y
    
    def prepare_for_training(self, df: pd.DataFrame, 
                            label_type: str = '7class',
                            test_size: float = 0.2) -> Dict:
        """
        Complete preprocessing pipeline
        
        Returns dict with X_train, X_test, y_train, y_test, scaler, label_encoder
        """
        # Clean data
        df = self.clean_data(df)
        
        # Extract features and labels
        X, y = self.extract_features_labels(df, label_type=label_type)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Encode labels if categorical
        if label_type in ['7class', '34class']:
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            y_test_encoded = self.label_encoder.transform(y_test)
        else:
            y_train_encoded = y_train
            y_test_encoded = y_test
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_encoded,
            'y_test': y_test_encoded,
            'y_train_raw': y_train,
            'y_test_raw': y_test,
            'feature_names': X.columns.tolist(),
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
    
    def save_preprocessor(self, filepath: str):
        """Save scaler and encoders"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'class_mapping': self.class_mapping,
                'label_to_group': self.label_to_group
            }, f)
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load scaler and encoders"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.scaler = data['scaler']
            self.label_encoder = data['label_encoder']
            self.class_mapping = data['class_mapping']
            self.label_to_group = data['label_to_group']
        print(f"Preprocessor loaded from {filepath}")


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = IoTDataPreprocessor()
    
    print("Loading data...")
    df = preprocessor.load_data(sample_size=10000)
    
    print("\nData shape:", df.shape)
    print("\nLabel distribution:")
    print(df['Label'].value_counts())
    
    print("\nPreparing for 7-class training...")
    data = preprocessor.prepare_for_training(df, label_type='7class')
    
    print(f"\nTraining set: {data['X_train'].shape}")
    print(f"Test set: {data['X_test'].shape}")
    print(f"Features: {len(data['feature_names'])}")
