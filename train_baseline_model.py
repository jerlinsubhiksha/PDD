"""
train_baseline_model.py
-----------------------
Demonstrates loading the 200x200 product defect dataset, splitting the data,
training baseline machine learning models (Random Forest & XGBoost),
evaluating defect detection metrics, and identifying top predictive sensors.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def main():
    csv_file = "product_defect_dataset.csv"
    print(f"Loading {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # Verify shape
    assert df.shape == (200, 200), f"Expected (200, 200), got {df.shape}"
    
    # Separate identifiers, features, and target
    product_ids = df['product_id']
    X = df.drop(columns=['product_id', 'defect_status'])
    y = df['defect_status']
    
    print(f"Features: {X.shape[1]} columns | Target: '{y.name}' with classes: {dict(y.value_counts())}")
    
    # Stratified train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape[0]} samples ({sum(y_train)} defects)")
    print(f"Test set:  {X_test.shape[0]} samples ({sum(y_test)} defects)")

    # 1. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("Random Forest Classifier Results:")
    print("="*50)
    print(classification_report(y_test, rf_preds, target_names=["Non-Defect (0)", "Defect (1)"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_probs):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, rf_preds))
    
    # 2. XGBoost Classifier
    xgb = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("XGBoost Classifier Results:")
    print("="*50)
    print(classification_report(y_test, xgb_preds, target_names=["Non-Defect (0)", "Defect (1)"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, xgb_probs):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, xgb_preds))
    
    # Top 10 Most Predictive Sensor Features
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_10 = importances.sort_values(ascending=False).head(10)
    print("\n" + "="*50)
    print("Top 10 Sensors Contributing to Product Defect Detection:")
    print("="*50)
    for rank, (sensor, score) in enumerate(top_10.items(), 1):
        print(f"{rank:2d}. {sensor:<15} Importance: {score:.4f}")

if __name__ == "__main__":
    main()
