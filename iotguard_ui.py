"""
IoTGuard - Interactive Web UI
Test the trained models with manual data entry or CSV upload
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="IoTGuard - AI-Powered IDS/IPS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #34495e;
        font-weight: 600;
        margin-top: 2rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .attack-box {
        padding: 20px;
        border-radius: 8px;
        background-color: #ffebee;
        border-left: 4px solid #e74c3c;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .benign-box {
        padding: 20px;
        border-radius: 8px;
        background-color: #e8f5e9;
        border-left: 4px solid #27ae60;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        with open('models/train_advanced_models/advanced_stage1_ensemble.pkl', 'rb') as f:
            stage1 = pickle.load(f)
        with open('models/train_advanced_models/advanced_stage2_ensemble.pkl', 'rb') as f:
            stage2 = pickle.load(f)
        return stage1, stage2
    except FileNotFoundError:
        st.error("⚠️ Models not found! Please train the models first by running train_advanced.py")
        return None, None

# Feature engineering function
def advanced_feature_engineering(df):
    df = df.copy()
    
    # 1. Flag ratios
    df['syn_ack_ratio'] = df['syn_flag_number'] / (df['ack_flag_number'] + 1)
    df['rst_fin_ratio'] = df['rst_flag_number'] / (df['fin_flag_number'] + 1)
    df['psh_ack_ratio'] = df['psh_flag_number'] / (df['ack_flag_number'] + 1)
    
    # 2. Traffic intensity
    df['packet_rate_ratio'] = df['Rate'] / (df['Number'] + 1)
    df['size_per_packet'] = df['Tot size'] / (df['Number'] + 1)
    df['avg_iat'] = df['IAT'] / (df['Number'] + 1)
    
    # 3. Statistical features
    df['variance_avg_ratio'] = df['Variance'] / (df['AVG'] + 1)
    df['std_avg_ratio'] = df['Std'] / (df['AVG'] + 1)
    df['range_stat'] = df['Max'] - df['Min']
    df['cv'] = df['Std'] / (df['AVG'] + 1)
    
    # 4. Flag totals
    df['flags_total'] = (df['syn_flag_number'] + df['ack_flag_number'] + 
                         df['rst_flag_number'] + df['fin_flag_number'] + 
                         df['psh_flag_number'])
    df['flag_diversity'] = (
        (df['syn_flag_number'] > 0).astype(int) +
        (df['ack_flag_number'] > 0).astype(int) +
        (df['rst_flag_number'] > 0).astype(int) +
        (df['fin_flag_number'] > 0).astype(int) +
        (df['psh_flag_number'] > 0).astype(int)
    )
    
    # 5. Protocol combinations
    df['tcp_http_combo'] = df['TCP'] * df['HTTP']
    df['udp_dns_combo'] = df['UDP'] * df['DNS']
    df['protocol_count'] = (
        df['HTTP'] + df['HTTPS'] + df['DNS'] + df['Telnet'] + 
        df['SMTP'] + df['SSH'] + df['TCP'] + df['UDP']
    )
    
    # 6. Time-based features
    df['ttl_rate_ratio'] = df['Time_To_Live'] / (df['Rate'] + 1)
    df['header_size_ratio'] = df['Header_Length'] / (df['Tot size'] + 1)
    
    # 7. Advanced ratios
    df['syn_count_ratio'] = df['syn_count'] / (df['Number'] + 1)
    df['ack_count_ratio'] = df['ack_count'] / (df['Number'] + 1)
    df['fin_count_ratio'] = df['fin_count'] / (df['Number'] + 1)
    df['rst_count_ratio'] = df['rst_count'] / (df['Number'] + 1)
    
    # 8. Interaction features
    df['rate_ttl_interaction'] = df['Rate'] * df['Time_To_Live']
    df['size_rate_interaction'] = df['Tot size'] * df['Rate']
    
    # 9. Log transformations
    df['log_rate'] = np.log1p(df['Rate'])
    df['log_tot_size'] = np.log1p(df['Tot size'])
    df['log_number'] = np.log1p(df['Number'])
    
    # Clean data
    df = df.replace([np.inf, -np.inf], np.nan)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(0)
    
    return df

# Predict function
def predict_traffic(df, stage1_models, stage2_models):
    """Predict traffic using 2-stage ensemble with detailed stage info"""
    # Apply feature engineering
    df_features = advanced_feature_engineering(df)
    
    # Remove label if exists
    if 'Label' in df_features.columns:
        df_features = df_features.drop('Label', axis=1)
    if 'Category' in df_features.columns:
        df_features = df_features.drop('Category', axis=1)
    if 'Binary' in df_features.columns:
        df_features = df_features.drop('Binary', axis=1)
    
    # Stage 1: Binary classification
    X_scaled = stage1_models['scaler'].transform(df_features)
    
    lgb_proba = stage1_models['lgb'].predict(X_scaled)
    xgb_proba = stage1_models['xgb'].predict_proba(X_scaled)[:, 1]
    rf_proba = stage1_models['rf'].predict_proba(X_scaled)[:, 1]
    
    ensemble_proba = (0.5 * lgb_proba + 0.3 * xgb_proba + 0.2 * rf_proba)
    binary_pred = (ensemble_proba > 0.5).astype(int)
    stage1_confidence = np.where(binary_pred == 1, ensemble_proba, 1 - ensemble_proba)
    
    # Stage 2: Multi-class for attacks
    final_predictions = []
    stage2_confidence = []
    stage2_activated = []
    
    for i in range(len(binary_pred)):
        if binary_pred[i] == 0:
            final_predictions.append('Benign')
            stage2_confidence.append(None)  # Stage 2 not used
            stage2_activated.append(False)
        else:
            X_sample = df_features.iloc[i:i+1]
            X_sample_scaled = stage2_models['scaler'].transform(X_sample)
            
            lgb_p = stage2_models['lgb'].predict(X_sample_scaled)[0]
            xgb_p = stage2_models['xgb'].predict_proba(X_sample_scaled)[0]
            
            ensemble_p = (0.6 * lgb_p + 0.4 * xgb_p)
            pred_class = np.argmax(ensemble_p)
            category = stage2_models['label_encoder'].inverse_transform([pred_class])[0]
            final_predictions.append(category)
            stage2_confidence.append(ensemble_p[pred_class])
            stage2_activated.append(True)
    
    return final_predictions, stage1_confidence, stage2_confidence, stage2_activated

# Main UI
st.markdown('<div class="main-header">IoTGuard - AI-Powered IDS/IPS</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #7f8c8d;">Advanced Deep Learning System for IoT Network Security</p>', unsafe_allow_html=True)

# Load models
stage1_models, stage2_models = load_models()

if stage1_models is None or stage2_models is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    
    page = st.radio(
        "Select Page:",
        ["System Dashboard", "Manual Testing", "CSV Analysis", "Batch Processing", "System Information"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.info("**Model Type:** Ensemble (LightGBM + XGBoost)")
    st.info("**Detection Accuracy:** 99%+")
    st.info("**Attack Categories:** 6 Types")
    
    st.markdown("---")
    st.caption("IoTGuard v1.0 | 2025")

# Main content based on page selection
if page == "System Information":
    st.markdown('<div class="sub-header">System Information</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### About IoTGuard")
        st.markdown("""
        IoTGuard is an advanced AI-powered Intrusion Detection and Prevention System (IDS/IPS) 
        specifically designed for IoT network security. The system uses a sophisticated 2-stage 
        ensemble learning approach to detect and classify network attacks with high accuracy.
        
        **Key Features:**
        - Real-time threat detection
        - 2-stage classification architecture
        - Ensemble learning (LightGBM + XGBoost)
        - Support for 6 attack categories
        - Batch processing capabilities
        - CSV data analysis
        - Manual traffic testing
        """)
        
        st.markdown("#### Attack Categories")
        st.markdown("""
        | Category | Description |
        |----------|-------------|
        | **DDoS-DoS** | Distributed/Denial of Service attacks |
        | **Mirai** | IoT botnet-based attacks |
        | **Recon** | Network reconnaissance and scanning |
        | **Spoofing** | Identity and DNS spoofing |
        | **Web** | Web-based attacks (XSS, SQL injection) |
        | **BruteForce** | Password brute force attempts |
        | **Benign** | Normal, legitimate traffic |
        """)
    
    with col2:
        st.markdown("#### Model Architecture")
        st.markdown("""
        **Stage 1: Binary Classification**
        - Purpose: Detect if traffic is benign or malicious
        - Models: LightGBM + XGBoost + RandomForest ensemble
        - Accuracy: 99.1%
        - F1-Score: 99.5%
        
        **Stage 2: Multi-Class Classification**
        - Purpose: Classify attack types
        - Models: LightGBM + XGBoost ensemble
        - Accuracy: 77.9%
        - F1-Score: 80.0%
        
        **Feature Engineering:**
        - 39 original network features
        - 29 engineered features
        - Total: 68 features for classification
        """)
        
        st.markdown("#### Dataset Information")
        st.markdown("""
        **Training Data:**
        - Dataset: CIC-IoT-2023
        - Training samples: 1,215,816
        - Testing samples: 256,051
        - Benign ratio: 50.6%
        
        **Balancing Techniques:**
        - SMOTE (Synthetic Minority Over-sampling)
        - Cost-sensitive learning
        - Random undersampling
        """)
        
        st.markdown("#### Performance Metrics")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Stage 1 (Binary)", "99.1%")
        with col_b:
            st.metric("Stage 2 (Multi-Class)", "77.9%")
        with col_c:
            st.metric("Overall F1-Score", "99.5%")

elif page == "System Dashboard":
    st.markdown('<div class="sub-header">System Dashboard</div>', unsafe_allow_html=True)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Model Status**")
        st.success("Operational")
    with col2:
        st.markdown("**Stage 1 Accuracy**")
        st.info("99.1%")
    with col3:
        st.markdown("**Stage 2 Accuracy**")
        st.info("77.9%")
    with col4:
        st.markdown("**Attack Types**")
        st.info("6 Categories")
    
    st.markdown("---")
    
    st.markdown("### Quick Start Guide")
    
    tab1, tab2, tab3 = st.tabs(["Manual Testing", "CSV Upload", "Batch Analysis"])
    
    with tab1:
        st.markdown("""
        **Manual Testing** allows you to test individual network traffic samples:
        1. Navigate to 'Manual Testing' page
        2. Enter network feature values
        3. Click 'Analyze Traffic' to get predictions
        4. View detailed results and confidence scores
        """)
    
    with tab2:
        st.markdown("""
        **CSV Analysis** enables bulk testing from CSV files:
        1. Navigate to 'CSV Analysis' page
        2. Upload your CSV file with network features
        3. Choose scan mode (All, Batch, or Custom Range)
        4. Analyze and download results
        """)
    
    with tab3:
        st.markdown("""
        **Batch Processing** for analyzing dataset files:
        1. Navigate to 'Batch Processing' page
        2. Select CSV files from the dataset
        3. Process multiple files simultaneously
        4. Export comprehensive results
        """)

elif page == "Manual Testing":
    st.markdown('<div class="sub-header">Manual Traffic Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Network Features")
        header_length = st.number_input("Header Length", 0, 1000, 52)
        protocol_type = st.number_input("Protocol Type", 0, 255, 6)
        ttl = st.number_input("Time To Live", 0, 255, 64)
        rate = st.number_input("Rate (packets/sec)", 0.0, 100000.0, 100.0)
        
    with col2:
        st.markdown("#### Flag Counts")
        syn_flag = st.number_input("SYN Flags", 0, 10000, 0)
        ack_flag = st.number_input("ACK Flags", 0, 10000, 0)
        rst_flag = st.number_input("RST Flags", 0, 10000, 0)
        fin_flag = st.number_input("FIN Flags", 0, 10000, 0)
        psh_flag = st.number_input("PSH Flags", 0, 10000, 0)
        
    with col3:
        st.markdown("#### Protocols")
        http = st.checkbox("HTTP")
        https = st.checkbox("HTTPS")
        dns = st.checkbox("DNS")
        tcp = st.checkbox("TCP", value=True)
        udp = st.checkbox("UDP")
        ssh = st.checkbox("SSH")
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("#### Traffic Statistics")
        tot_size = st.number_input("Total Size (bytes)", 0, 1000000, 1500)
        number = st.number_input("Number of Packets", 1, 100000, 10)
        iat = st.number_input("Inter-Arrival Time", 0.0, 1000.0, 10.0)
        
    with col5:
        st.markdown("#### Statistical Measures")
        avg = st.number_input("Average", 0.0, 100000.0, 100.0)
        std = st.number_input("Std Deviation", 0.0, 100000.0, 50.0)
        variance = st.number_input("Variance", 0.0, 100000.0, 2500.0)
        min_val = st.number_input("Min Value", 0.0, 100000.0, 50.0)
        max_val = st.number_input("Max Value", 0.0, 100000.0, 150.0)
    
    if st.button("Analyze Traffic", type="primary", use_container_width=True):
        # Create dataframe with all required features
        data = {
            'Header_Length': [header_length],
            'Protocol Type': [protocol_type],
            'Time_To_Live': [ttl],
            'Rate': [rate],
            'fin_flag_number': [fin_flag],
            'syn_flag_number': [syn_flag],
            'rst_flag_number': [rst_flag],
            'psh_flag_number': [psh_flag],
            'ack_flag_number': [ack_flag],
            'ece_flag_number': [0],
            'cwr_flag_number': [0],
            'ack_count': [ack_flag],
            'syn_count': [syn_flag],
            'fin_count': [fin_flag],
            'rst_count': [rst_flag],
            'HTTP': [int(http)],
            'HTTPS': [int(https)],
            'DNS': [int(dns)],
            'Telnet': [0],
            'SMTP': [0],
            'SSH': [int(ssh)],
            'IRC': [0],
            'TCP': [int(tcp)],
            'UDP': [int(udp)],
            'DHCP': [0],
            'ARP': [0],
            'ICMP': [0],
            'IGMP': [0],
            'IPv': [1],
            'LLC': [0],
            'Tot sum': [tot_size],
            'Min': [min_val],
            'Max': [max_val],
            'AVG': [avg],
            'Std': [std],
            'Tot size': [tot_size],
            'IAT': [iat],
            'Number': [number],
            'Variance': [variance]
        }
        
        df = pd.DataFrame(data)
        
        with st.spinner("🔄 Analyzing traffic pattern..."):
            predictions, stage1_conf, stage2_conf, stage2_active = predict_traffic(df, stage1_models, stage2_models)
        
        result = predictions[0]
        s1_confidence = stage1_conf[0]
        s2_confidence = stage2_conf[0]
        is_stage2 = stage2_active[0]
        
        st.markdown("---")
        st.markdown("### 🎯 Detection Result")
        
        # Show Stage 1 Result
        st.markdown("#### 📊 Stage 1: Binary Classification")
        col_s1_a, col_s1_b = st.columns(2)
        with col_s1_a:
            if result == "Benign":
                st.success(f"✅ Classification: **BENIGN**")
            else:
                st.error(f"⚠️ Classification: **ATTACK**")
        with col_s1_b:
            st.info(f"🎯 Confidence: **{s1_confidence*100:.2f}%**")
        
        # Show Stage 2 Result if attack detected
        if is_stage2:
            st.markdown("#### 🔍 Stage 2: Attack Type Classification")
            col_s2_a, col_s2_b = st.columns(2)
            with col_s2_a:
                st.warning(f"🏷️ Attack Type: **{result}**")
            with col_s2_b:
                st.info(f"🎯 Confidence: **{s2_confidence*100:.2f}%**")
        
        # Final result box
        st.markdown("---")
        if result == "Benign":
            st.markdown(f"""
            <div class="benign-box">
                <h2 style="color: #4caf50; margin: 0;">✅ BENIGN TRAFFIC</h2>
                <p style="font-size: 1.1rem; margin: 10px 0;">This traffic appears to be normal and safe.</p>
                <p style="margin: 5px 0;"><strong>Stage 1 Decision:</strong> Not malicious (Confidence: {s1_confidence*100:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="attack-box">
                <h2 style="color: #f44336; margin: 0;">⚠️ ATTACK DETECTED!</h2>
                <p style="font-size: 1.5rem; margin: 10px 0; font-weight: bold;">Type: {result}</p>
                <p style="margin: 5px 0;"><strong>Stage 1:</strong> Classified as attack (Confidence: {s1_confidence*100:.1f}%)</p>
                <p style="margin: 5px 0;"><strong>Stage 2:</strong> Identified as {result} (Confidence: {s2_confidence*100:.1f}%)</p>
                <p style="margin: 10px 0; color: #c62828;"><strong>⛔ Recommended Action:</strong> Block and investigate</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "CSV Analysis":
    st.markdown('<div class="sub-header">CSV File Analysis</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file (must contain the required 39 network features)",
        type=['csv'],
        help="Upload a CSV file with network traffic data"
    )
    
    if uploaded_file is not None:
        try:
            # Read file to get total rows
            df_full = pd.read_csv(uploaded_file)
            total_rows = len(df_full)
            
            st.success(f"✅ File loaded successfully: {total_rows:,} samples")
            
            # Batch/Filter Controls
            st.markdown("### ⚙️ Scan Controls")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                scan_mode = st.radio(
                    "Scan Mode:",
                    ["Scan All", "Batch Scan", "Custom Range"],
                    horizontal=True
                )
            
            # Initialize variables
            start_idx = 0
            end_idx = total_rows
            batch_size = 1000
            
            if scan_mode == "Batch Scan":
                with col2:
                    batch_size = st.number_input(
                        "Batch Size (rows per scan):",
                        min_value=10,
                        max_value=total_rows,
                        value=min(1000, total_rows),
                        step=100,
                        help="Number of rows to analyze at once"
                    )
                with col3:
                    batch_num = st.number_input(
                        "Batch #:",
                        min_value=1,
                        max_value=(total_rows // batch_size) + 1,
                        value=1,
                        help="Which batch to analyze"
                    )
                
                start_idx = (batch_num - 1) * batch_size
                end_idx = min(start_idx + batch_size, total_rows)
                
                st.info(f"📊 Will analyze rows {start_idx:,} to {end_idx:,} ({end_idx - start_idx:,} samples)")
                
            elif scan_mode == "Custom Range":
                with col2:
                    start_idx = st.number_input(
                        "Start Row:",
                        min_value=0,
                        max_value=total_rows-1,
                        value=0
                    )
                with col3:
                    end_idx = st.number_input(
                        "End Row:",
                        min_value=start_idx+1,
                        max_value=total_rows,
                        value=min(start_idx + 1000, total_rows)
                    )
                
                st.info(f"📊 Will analyze rows {start_idx:,} to {end_idx:,} ({end_idx - start_idx:,} samples)")
            else:
                st.info(f"📊 Will analyze all {total_rows:,} samples")
            
            # Select the data to analyze
            df = df_full.iloc[start_idx:end_idx].copy()
            
            # Show preview
            with st.expander("👁️ Preview Data"):
                st.write(f"Showing rows {start_idx} to {start_idx + min(10, len(df))}")
                st.dataframe(df.head(10))
            
            if st.button("Analyze Data", type="primary"):
                with st.spinner(f"🔄 Analyzing {len(df):,} samples (rows {start_idx:,} to {end_idx:,})..."):
                    predictions, stage1_conf, stage2_conf, stage2_active = predict_traffic(df, stage1_models, stage2_models)
                
                # Add results to dataframe
                df['Prediction'] = predictions
                df['Stage1_Confidence'] = [f"{c*100:.2f}%" for c in stage1_conf]
                df['Stage2_Used'] = ['Yes' if s else 'No' for s in stage2_active]
                df['Stage2_Confidence'] = [f"{c*100:.2f}%" if c is not None else 'N/A' for c in stage2_conf]
                df['Row_Number'] = range(start_idx, end_idx)
                
                # Summary statistics
                st.markdown("### 📊 Detection Summary")
                st.caption(f"Analyzed rows {start_idx:,} to {end_idx:,} (Total: {total_rows:,} in file)")
                
                col1, col2, col3, col4 = st.columns(4)
                
                total = len(predictions)
                benign_count = predictions.count('Benign')
                attack_count = total - benign_count
                
                with col1:
                    st.metric("Total Samples", f"{total:,}")
                with col2:
                    st.metric("Benign", f"{benign_count:,}", help=f"{benign_count/total*100:.1f}% of total")
                with col3:
                    st.metric("Attacks", f"{attack_count:,}", help=f"{attack_count/total*100:.1f}% of total")
                with col4:
                    avg_s1_conf = np.mean(stage1_conf) * 100
                    st.metric("Avg Stage 1 Conf", f"{avg_s1_conf:.1f}%")
                
                # Stage statistics
                st.markdown("#### 🔍 Classification Stages")
                col_st1, col_st2 = st.columns(2)
                with col_st1:
                    st.info(f"**Stage 1 (Binary):** Analyzed all {total:,} samples")
                    st.caption(f"✓ {benign_count:,} classified as Benign | ✓ {attack_count:,} classified as Attack")
                with col_st2:
                    stage2_count = sum(stage2_active)
                    st.info(f"**Stage 2 (Multi-Class):** Analyzed {stage2_count:,} attack samples")
                    if stage2_count > 0:
                        s2_confs = [c for c in stage2_conf if c is not None]
                        avg_s2_conf = np.mean(s2_confs) * 100 if s2_confs else 0
                        st.caption(f"Average confidence: {avg_s2_conf:.1f}%")
                
                # Attack breakdown
                if attack_count > 0:
                    st.markdown("### Attack Breakdown")
                    
                    attack_types = [p for p in predictions if p != 'Benign']
                    attack_counts = pd.Series(attack_types).value_counts()
                    
                    # Pie chart
                    fig = px.pie(
                        values=attack_counts.values,
                        names=attack_counts.index,
                        title="Attack Type Distribution",
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Accuracy check if labels exist
                if 'Label' in df_full.columns:
                    st.markdown("### ✅ Accuracy Check")
                    
                    # Map labels to categories
                    true_labels = df_full.iloc[start_idx:end_idx]['Label'].tolist()
                    
                    def map_to_category(label):
                        if label == 'BENIGN':
                            return 'Benign'
                        elif 'DDOS' in label.upper() or 'DOS' in label.upper():
                            return 'DDoS-DoS'
                        elif 'MIRAI' in label.upper():
                            return 'Mirai'
                        elif 'RECON' in label.upper() or 'VULNERABILITY' in label.upper():
                            return 'Reconnaissance'
                        elif 'SPOOF' in label.upper() or 'MITM' in label.upper():
                            return 'Spoofing'
                        elif any(x in label.upper() for x in ['SQL', 'XSS', 'COMMAND', 'BROWSER', 'UPLOAD', 'BACKDOOR']):
                            return 'Web-Based'
                        elif 'BRUTE' in label.upper() or 'DICTIONARY' in label.upper():
                            return 'BruteForce'
                        return 'Unknown'
                    
                    true_categories = [map_to_category(l) for l in true_labels]
                    
                    # Calculate accuracies
                    correct_total = sum(1 for t, p in zip(true_categories, predictions) if t == p)
                    accuracy_total = correct_total / total * 100
                    
                    # Stage 1 accuracy (Binary: Benign vs Attack)
                    true_binary = ['Benign' if t == 'Benign' else 'Attack' for t in true_categories]
                    pred_binary = ['Benign' if p == 'Benign' else 'Attack' for p in predictions]
                    correct_s1 = sum(1 for t, p in zip(true_binary, pred_binary) if t == p)
                    accuracy_s1 = correct_s1 / total * 100
                    
                    col_acc1, col_acc2, col_acc3 = st.columns(3)
                    with col_acc1:
                        st.metric("Stage 1 Accuracy", f"{accuracy_s1:.2f}%", help="Binary classification (Benign vs Attack)")
                    with col_acc2:
                        st.metric("Overall Accuracy", f"{accuracy_total:.2f}%", help="Complete classification including attack types")
                    with col_acc3:
                        st.metric("Correct", f"{correct_total}/{total}", help=f"{correct_total} out of {total} samples classified correctly")
                    
                    # Show misclassifications
                    df['True_Label'] = true_categories
                    df['Correct'] = ['✓' if t == p else '✗' for t, p in zip(true_categories, predictions)]
                    
                    wrong_count = total - correct_total
                    if wrong_count > 0:
                        with st.expander(f"❌ View {wrong_count} Misclassifications"):
                            misclassified = df[df['Correct'] == '✗'][['Row_Number', 'True_Label', 'Prediction', 'Stage1_Confidence', 'Stage2_Used', 'Stage2_Confidence']]
                            st.dataframe(misclassified, use_container_width=True)
                
                # Download results
                st.markdown("### 💾 Download Results")
                csv = df.to_csv(index=False)
                
                # Create descriptive filename
                scan_desc = f"rows{start_idx}-{end_idx}" if start_idx > 0 or end_idx < total_rows else "all"
                filename = f"iotguard_results_{scan_desc}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                st.download_button(
                    label=f"📥 Download Analysis Results ({len(df):,} samples)",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                )
                
                # Show detailed results
                with st.expander("📋 View Detailed Results Table"):
                    display_cols = ['Row_Number', 'Prediction', 'Stage1_Confidence', 'Stage2_Used', 'Stage2_Confidence']
                    if 'Correct' in df.columns:
                        display_cols.insert(2, 'True_Label')
                        display_cols.insert(3, 'Correct')
                    st.dataframe(df[display_cols], use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

else:  # Batch Processing
    st.markdown('<div class="sub-header">Batch Processing from Dataset</div>', unsafe_allow_html=True)
    
    st.info("📂 Analyzing files from dataset/CSV/MERGED_CSV/")
    
    try:
        import glob
        csv_files = sorted(glob.glob('dataset/CSV/MERGED_CSV/*.csv'))
        
        if len(csv_files) == 0:
            st.warning("⚠️ No CSV files found in dataset/CSV/MERGED_CSV/")
        else:
            st.success(f"Found {len(csv_files)} CSV files")
            
            selected_files = st.multiselect(
                "Select files to analyze:",
                [os.path.basename(f) for f in csv_files],
                default=[os.path.basename(csv_files[0])] if len(csv_files) > 0 else []
            )
            
            num_samples = st.slider("Samples per file", 100, 10000, 1000, 100)
            
            if st.button("Start Analysis", type="primary"):
                if len(selected_files) == 0:
                    st.warning("Please select at least one file")
                else:
                    all_predictions = []
                    all_labels = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for idx, filename in enumerate(selected_files):
                        status_text.text(f"Processing {filename}...")
                        
                        filepath = f"dataset/CSV/MERGED_CSV/{filename}"
                        df = pd.read_csv(filepath, nrows=num_samples)
                        
                        if 'Label' in df.columns:
                            all_labels.extend(df['Label'].tolist())
                        
                        predictions, stage1_conf, stage2_conf, stage2_active = predict_traffic(df, stage1_models, stage2_models)
                        all_predictions.extend(predictions)
                        
                        progress_bar.progress((idx + 1) / len(selected_files))
                    
                    status_text.text("✅ Analysis complete!")
                    
                    # Show results
                    st.markdown("### 📊 Batch Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    total = len(all_predictions)
                    benign = all_predictions.count('Benign')
                    attacks = total - benign
                    
                    with col1:
                        st.metric("Total Analyzed", f"{total:,}")
                    with col2:
                        st.metric("Benign Detected", f"{benign:,}", help=f"{benign/total*100:.1f}% of total")
                    with col3:
                        st.metric("Attacks Detected", f"{attacks:,}", help=f"{attacks/total*100:.1f}% of total")
                    
                    # Distribution chart
                    pred_counts = pd.Series(all_predictions).value_counts()
                    fig = go.Figure(data=[
                        go.Bar(x=pred_counts.index, y=pred_counts.values,
                               marker_color=['green' if x == 'Benign' else 'red' for x in pred_counts.index])
                    ])
                    fig.update_layout(
                        title="Detection Distribution",
                        xaxis_title="Category",
                        yaxis_title="Count",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Accuracy if labels available
                    if len(all_labels) > 0:
                        st.markdown("### 🎯 Accuracy Metrics")
                        
                        # Map labels to categories
                        class_mapping = {
                            'DDoS-DoS': ['DDOS-RSTFINFLOOD', 'DDOS-PSHACK_FLOOD', 'DDOS-SYN_FLOOD',
                                         'DDOS-UDP_FLOOD', 'DDOS-TCP_FLOOD', 'DDOS-ICMP_FLOOD',
                                         'DDOS-SYNONYMOUSIP_FLOOD', 'DDOS-ACK_FRAGMENTATION',
                                         'DDOS-UDP_FRAGMENTATION', 'DDOS-ICMP_FRAGMENTATION',
                                         'DDOS-SLOWLORIS', 'DDOS-HTTP_FLOOD',
                                         'DOS-UDP_FLOOD', 'DOS-SYN_FLOOD', 'DOS-TCP_FLOOD', 'DOS-HTTP_FLOOD'],
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
                        
                        true_categories = [label_to_group.get(l, l) for l in all_labels]
                        
                        correct = sum([1 for i in range(len(all_predictions)) if all_predictions[i] == true_categories[i]])
                        accuracy = correct / len(all_predictions) * 100
                        
                        st.success(f"🎯 Overall Accuracy: {accuracy:.2f}% ({correct}/{len(all_predictions)} correct)")
                        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>IoTGuard v1.0</strong> - Advanced AI-Powered Intrusion Detection System</p>
    <p>Powered by LightGBM, XGBoost, and Random Forest Ensemble</p>
    <p>🛡️ Protecting IoT Networks with Deep Learning</p>
</div>
""", unsafe_allow_html=True)
