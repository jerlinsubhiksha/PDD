"""
predict_single_product.py
-------------------------
Command-line utility to run real-time defect prediction on any product in the dataset
or on new incoming sensor readings.

Usage:
    python predict_single_product.py PROD_014
    python predict_single_product.py --random
    python predict_single_product.py --batch 5
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
import argparse
import joblib
import pandas as pd
import numpy as np

PIPELINE_PATH = os.path.join("lifecycle_outputs", "defect_detection_pipeline.joblib")
DATASET_PATH = "product_defect_dataset.csv"

def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Pipeline not found at {PIPELINE_PATH}. Please run run_lifecycle_pipeline.py first.")
    return joblib.load(PIPELINE_PATH)

def predict_product(product_id=None, row_index=None, sample_row=None):
    bundle = load_pipeline()
    scaler = bundle["scaler"]
    selector = bundle["selector"]
    model = bundle["model"]
    threshold = bundle["optimal_threshold"]

    df = pd.read_csv(DATASET_PATH)

    if sample_row is None:
        if product_id is not None:
            matches = df[df["product_id"] == product_id]
            if matches.empty:
                print(f"Error: Product '{product_id}' not found in {DATASET_PATH}.")
                return
            row = matches.iloc[0]
        elif row_index is not None:
            row = df.iloc[row_index]
        else:
            row = df.sample(n=1, random_state=None).iloc[0]
    else:
        row = sample_row

    prod_id = row["product_id"]
    true_label = int(row["defect_status"])
    sensors = row.drop(labels=["product_id", "defect_status"]).values

    # Preprocess & Predict
    scaled = scaler.transform([sensors])
    selected = selector.transform(scaled)
    defect_prob = float(model.predict_proba(selected)[0, 1])
    is_defective = int(defect_prob >= threshold)

    if defect_prob >= 0.50:
        verdict = "CRITICAL DEFECT DETECTED"
        color = "\033[91m" # Red
        action = "REJECT WAFER - HALT LOT & AUDIT CHAMBER"
    elif defect_prob >= threshold:
        verdict = "MODERATE DEFECT RISK"
        color = "\033[93m" # Yellow
        action = "FLAG FOR SECONDARY AUTOMATED RETEST"
    else:
        verdict = "PASSED QUALITY INSPECTION"
        color = "\033[92m" # Green
        action = "APPROVE FOR ASSEMBLY & PACKAGING"
    reset = "\033[0m"

    print("=" * 65)
    print(f"  PRODUCT QUALITY INSPECTION REPORT: {prod_id}")
    print("=" * 65)
    print(f"  [+] Ground Truth Inspection Label: {'DEFECTIVE (1)' if true_label == 1 else 'NORMAL / PASS (0)'}")
    print(f"  [+] Model Defect Probability:      {defect_prob * 100:.2f}%")
    print(f"  [+] Operating Decision Threshold:  {threshold:.2f}")
    print(f"  [+] Inspection Status:             {verdict}")
    print(f"  [+] Action Recommendation:         {action}")
    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="Real-time Product Defect Prediction Engine")
    parser.add_argument("product_id", nargs="?", default=None, help="Product ID (e.g. PROD_014)")
    parser.add_argument("--random", action="store_true", help="Pick a random product from dataset")
    parser.add_argument("--batch", type=int, default=None, help="Run prediction on first N products")
    args = parser.parse_args()

    if args.batch:
        df = pd.read_csv(DATASET_PATH)
        for i in range(min(args.batch, len(df))):
            predict_product(row_index=i)
            print()
    elif args.random or args.product_id is None:
        predict_product()
    else:
        predict_product(product_id=args.product_id)

if __name__ == "__main__":
    main()
