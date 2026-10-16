"""
generate_ui_data.py
-------------------
Extracts all data, predictions, PCA projections, and model metrics into
a consolidated JSON object for the interactive web UI.
"""

import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import joblib

DATASET_PATH = "product_defect_dataset.csv"
PIPELINE_PATH = "lifecycle_outputs/defect_detection_pipeline.joblib"
SUMMARY_PATH = "lifecycle_outputs/lifecycle_summary.json"
OUTPUT_JSON = "ui_data.json"

def main():
    df = pd.read_csv(DATASET_PATH)
    bundle = joblib.load(PIPELINE_PATH)
    with open(SUMMARY_PATH, "r") as f:
        lifecycle_summary = json.load(f)

    scaler = bundle["scaler"]
    selector = bundle["selector"]
    model = bundle["model"]
    threshold = bundle["optimal_threshold"]

    feature_cols = [c for c in df.columns if c not in ["product_id", "defect_status"]]
    X_raw = df[feature_cols].values
    y = df["defect_status"].values

    # Transform
    X_scaled = scaler.transform(X_raw)
    X_selected = selector.transform(X_scaled)
    probs = model.predict_proba(X_selected)[:, 1]

    # PCA 2D for interactive scatter
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(X_scaled)
    pca_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]

    # Key sensors to track
    key_sensors = ["sensor_111", "sensor_058", "sensor_059", "sensor_008", "sensor_062", 
                   "sensor_178", "sensor_165", "sensor_021", "sensor_100", "sensor_093"]

    sensor_stats = {}
    for s in key_sensors:
        sensor_stats[s] = {
            "min": round(float(df[s].min()), 2),
            "max": round(float(df[s].max()), 2),
            "mean": round(float(df[s].mean()), 2),
            "median": round(float(df[s].median()), 2)
        }

    # Products list
    products = []
    for i in range(len(df)):
        prod_id = df.iloc[i]["product_id"]
        true_label = int(y[i])
        prob = round(float(probs[i]), 4)
        px = round(float(pca_2d[i, 0]), 3)
        py = round(float(pca_2d[i, 1]), 3)
        
        sensor_vals = {}
        for s in key_sensors:
            sensor_vals[s] = round(float(df.iloc[i][s]), 2)

        products.append({
            "id": prod_id,
            "true_label": true_label,
            "prob": prob,
            "pred_label": int(prob >= threshold),
            "pca_x": px,
            "pca_y": py,
            "sensors": sensor_vals
        })

    # Threshold curve simulation
    threshold_curve = []
    for t in np.linspace(0.05, 0.95, 19):
        t_float = round(float(t), 2)
        preds = (probs >= t_float).astype(int)
        cm = confusion_matrix(y, preds)
        # [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        rec = round(float(recall_score(y, preds, zero_division=0)), 3)
        prec = round(float(precision_score(y, preds, zero_division=0)), 3)
        f1 = round(float(f1_score(y, preds, zero_division=0)), 3)
        acc = round(float((tp + tn) / len(y)), 3)
        threshold_curve.append({
            "threshold": t_float,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": rec, "precision": prec, "f1": f1, "accuracy": acc
        })

    ui_data = {
        "dataset_meta": {
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "feature_count": len(feature_cols),
            "defects_count": int(np.sum(y == 1)),
            "normals_count": int(np.sum(y == 0)),
            "optimal_threshold": threshold,
            "pca_variance": pca_var
        },
        "key_sensors": key_sensors,
        "sensor_stats": sensor_stats,
        "products": products,
        "threshold_curve": threshold_curve,
        "lifecycle_summary": lifecycle_summary
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(ui_data, f, indent=2)

    print(f"Generated {OUTPUT_JSON} successfully ({len(products)} products, {len(threshold_curve)} threshold steps)")

if __name__ == "__main__":
    main()
