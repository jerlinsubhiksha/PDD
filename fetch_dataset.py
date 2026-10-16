"""
fetch_dataset.py
----------------
Fetches real industrial semiconductor manufacturing product defect data from 
the online UCI SECOM (SemiConductor Operations and Manufacturing) repository.

Constructs a clean, ready-to-train dataset with exactly 200 rows and 200 columns:
- Column 1: 'product_id' (Identifier for each manufactured unit: PROD_001 to PROD_200)
- Columns 2 to 199: 198 continuous sensor and inline metrology features (sensor_001 to sensor_198)
  representing process temperatures, chamber pressures, gas flow rates, RF power, wafer thickness,
  etch rates, and electrical test metrics.
- Column 200: 'defect_status' (Target label: 0 = Normal / Non-Defective, 1 = Defective product)

Class balance:
- 140 non-defective units (70%)
- 60 defective units (30%)
Total: 200 rows x 200 columns, with 0 missing values.
"""

import sys
import os
import urllib.request
import pandas as pd
import numpy as np

OUTPUT_CSV = "product_defect_dataset.csv"
SOURCE_URLS = [
    "https://cdn.jsdelivr.net/gh/sharmaroshan/SECOM-Detecting-Defected-Items@master/uci-secom.csv",
    "https://raw.githubusercontent.com/sharmaroshan/SECOM-Detecting-Defected-Items/master/uci-secom.csv",
    "https://raw.githubusercontent.com/mmyy2513/UCI-secom/master/uci-secom.csv"
]

def download_secom_data():
    print("Connecting to online data source...")
    for url in SOURCE_URLS:
        try:
            print(f"Attempting download from: {url}")
            df = pd.read_csv(url)
            print(f"Successfully downloaded SECOM dataset with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Failed to fetch from {url}: {e}")
    raise RuntimeError("Unable to download dataset from any available mirror.")

def build_200x200_dataset(raw_df):
    print("Processing and structuring dataset to 200 rows x 200 columns...")
    
    # Identify target column: 'Pass/Fail' where -1 = Pass (Normal), 1 = Fail (Defective)
    target_col = 'Pass/Fail' if 'Pass/Fail' in raw_df.columns else raw_df.columns[-1]
    
    defects = raw_df[raw_df[target_col] == 1].copy()
    passes = raw_df[raw_df[target_col] == -1].copy()
    print(f"Available samples in raw dataset: {len(passes)} normal, {len(defects)} defective")

    # Filter informative numeric feature columns (exclude Time and Target)
    feature_candidates = [c for c in raw_df.columns if c not in ['Time', target_col]]
    
    # 1. Eliminate constant columns (zero variance)
    variances = raw_df[feature_candidates].var()
    non_const_cols = variances[variances > 1e-4].index.tolist()
    
    # 2. Prefer columns with the lowest missing value rates (<5% nulls)
    null_rates = raw_df[non_const_cols].isnull().mean()
    clean_cols = null_rates[null_rates < 0.05].index.tolist()
    
    # Sort remaining features by variance to keep highest dynamic range signals
    sorted_features = variances[clean_cols].sort_values(ascending=False).index.tolist()
    
    # Select exactly 198 features
    selected_features = sorted_features[:198]
    print(f"Selected {len(selected_features)} high-variance, clean sensor measurement features.")

    # Sample exactly 200 rows with a representative defect distribution:
    # 60 defective products (30%) and 140 non-defective products (70%)
    sampled_defects = defects.sample(n=60, random_state=42)
    sampled_passes = passes.sample(n=140, random_state=42)
    
    combined = pd.concat([sampled_defects, sampled_passes])
    # Shuffle rows reproducibly
    shuffled = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Extract features and perform median imputation for any sparse missing values
    X_raw = shuffled[selected_features].copy()
    X_imputed = X_raw.fillna(X_raw.median())
    
    # Construct final dataframe with clean column names and metadata
    data_dict = {}
    data_dict['product_id'] = [f"PROD_{i+1:03d}" for i in range(len(shuffled))]
    
    for i, col in enumerate(selected_features):
        data_dict[f"sensor_{i+1:03d}"] = X_imputed[col].round(4).values
        
    # Map target: 0 = Normal / Non-Defective, 1 = Defective
    data_dict['defect_status'] = (shuffled[target_col] == 1).astype(int).values
    
    final_df = pd.DataFrame(data_dict)
    return final_df

def main():
    raw_df = download_secom_data()
    final_df = build_200x200_dataset(raw_df)
    
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved final dataset to: {os.path.abspath(OUTPUT_CSV)}")
    print(f"Dimensions: {final_df.shape[0]} rows x {final_df.shape[1]} columns")
    print(f"Defect status distribution:\n{final_df['defect_status'].value_counts().to_dict()}")
    print(f"Total null values: {final_df.isnull().sum().sum()}")
    print("\nFirst 3 rows and selected columns preview:")
    preview_cols = ['product_id', 'sensor_001', 'sensor_002', 'sensor_003', 'sensor_198', 'defect_status']
    print(final_df[preview_cols].head(3))

if __name__ == "__main__":
    main()
