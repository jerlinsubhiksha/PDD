# Product Defect Detection Dataset (200 x 200)

This dataset was sourced from the **UCI SECOM (SemiConductor Operations and Manufacturing) Benchmark Dataset**, collected from real-world semiconductor manufacturing lines. It is structured specifically for training and benchmarking machine learning models for industrial **Product Defect Detection**.

---

## 📊 Dataset Specifications

| Property | Value |
| :--- | :--- |
| **Total Rows** | **200** (Manufactured product units) |
| **Total Columns** | **200** (Product ID + 198 Sensor Features + Defect Status) |
| **Normal Products (`defect_status = 0`)** | **140** (70.0%) |
| **Defective Products (`defect_status = 1`)** | **60** (30.0%) |
| **Missing Values** | **0** (All values cleaned and imputed via median imputation) |
| **Source** | UCI Machine Learning Repository / SECOM |

---

## 🗂 Column Structure

The dataset contains exactly **200 columns**:

1. **`product_id`** (Column 1):
   - Unique product tracking identifier (`PROD_001` to `PROD_200`).
2. **`sensor_001` through `sensor_198`** (Columns 2 to 199):
   - 198 continuous numeric features capturing manufacturing telemetry, inline process sensor signals (chamber pressure, temperature, RF power, gas flows, etch rates), and electrical/metrological test readings.
   - Selected for high signal variance and low noise.
3. **`defect_status`** (Column 200 - Target Variable):
   - Binary label indicating quality inspection outcome:
     - `0`: Passed inspection (Non-defective / Normal product)
     - `1`: Failed inspection (Defective product)

---

## 🚀 Quickstart: Loading and Training

### 1. Load with Pandas
```python
import pandas as pd

df = pd.read_csv("product_defect_dataset.csv")
print(f"Shape: {df.shape}")  # Output: (200, 200)

# Separate features and target
X = df.drop(columns=["product_id", "defect_status"])
y = df["defect_status"]
```

### 2. Run the Provided Baseline Model
A ready-to-run script [`train_baseline_model.py`](train_baseline_model.py) is included in this workspace:
```bash
python train_baseline_model.py
```
This script trains Random Forest and XGBoost classifiers, prints classification metrics (Precision, Recall, F1, ROC-AUC), and displays the top 10 most predictive sensor features.

### 3. Re-fetch or Modify Dataset Parameters
To customize row count, defect proportion, or random seed:
```bash
python fetch_dataset.py
```
