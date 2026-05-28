"""
src/fairness_analysis.py
-------------------------
Compute SHAP feature importance and Fairlearn fairness metrics from the
already-trained model and residuals. Run AFTER train.py has been executed.

Outputs:
    data/processed/shap_values.csv     — mean |SHAP| per feature (grouped)
    data/processed/fairlearn_metrics.csv — MAE + R² per racial group
"""

import joblib
import numpy as np
import pandas as pd
import shap
from fairlearn.metrics import MetricFrame
from sklearn.metrics import mean_absolute_error, r2_score


# ── 1. Load saved pipeline and residuals ─────────────────────────────────────
pipeline   = joblib.load("models/best_model.pkl")
prep_step  = pipeline.named_steps["prep"]
model_step = pipeline.named_steps["model"]

resid_df = pd.read_csv("data/processed/residuals.csv")
print(f"Loaded {len(resid_df):,} test-set rows from residuals.csv")


# ── 2. Reconstruct X_test in the same form as training ───────────────────────
NUMERIC_FEATURES = [
    "AGE", "SMIN1", "SMAX1", "GLMIN", "GLMAX",
    "CRIMPTS", "TOTCHPTS", "XCRHISSR", "NUMDEPEN",
    "NOCOUNTS", "DRUGMIN", "MONSEX", "WEAPON", "NEWEDUC",
]
CATEGORICAL_FEATURES = [
    "ZONE", "DISTRICT", "DISPOSIT", "CITIZEN", "FISCAL_YEAR", "OFFGUIDE",
]

available_numeric = [f for f in NUMERIC_FEATURES if f in resid_df.columns]
available_categ   = [f for f in CATEGORICAL_FEATURES if f in resid_df.columns]

X_test = resid_df[available_numeric + available_categ].copy()

# Match dtype conversions from train.py
X_test["DISTRICT"]    = X_test["DISTRICT"].astype(str)
X_test["FISCAL_YEAR"] = X_test["FISCAL_YEAR"].astype(str)
X_test["OFFGUIDE"]    = X_test["OFFGUIDE"].astype(str)

y_test = resid_df["SENTTOT_actual"]
y_pred = resid_df["SENTTOT_predicted"]
race   = resid_df["NEWRACE"]


# ── 3. Transform test data through the fitted preprocessor ───────────────────
X_transformed = prep_step.transform(X_test)

# Build full feature name list after one-hot encoding
cat_encoder    = prep_step.named_transformers_["cat"].named_steps["encode"]
cat_feat_names = cat_encoder.get_feature_names_out(available_categ).tolist()
all_feat_names = available_numeric + cat_feat_names


# ── 4. SHAP — which legal factors drive predictions most ─────────────────────
# TreeExplainer is fast and exact for tree-based models (GBM, RF, etc.)
print("\nComputing SHAP values...")
explainer   = shap.TreeExplainer(model_step)
shap_values = explainer.shap_values(X_transformed)


# Collapse one-hot encoded columns back to their original feature group
def get_original_feature(feat_name, cat_features):
    for cat in cat_features:
        if feat_name.startswith(cat + "_") or feat_name == cat:
            return cat
    return feat_name


mean_abs_shap = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature":       all_feat_names,
    "mean_abs_shap": mean_abs_shap,
})
shap_df["feature_group"] = shap_df["feature"].apply(
    lambda f: get_original_feature(f, available_categ)
)

shap_grouped = (
    shap_df.groupby("feature_group")["mean_abs_shap"]
    .sum()
    .reset_index()
    .rename(columns={"feature_group": "feature"})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)

print("\nTop 10 features by mean |SHAP| value:")
print(shap_grouped.head(10).to_string(index=False))

shap_grouped.to_csv("data/processed/shap_values.csv", index=False)
print("Saved → data/processed/shap_values.csv")


# ── 5. Fairlearn — model performance broken down by racial group ──────────────
# MetricFrame computes MAE and R² separately for each group.
# If the model has higher error for one race, its legal baseline is less certain.
race_labels = race.map({1: "White", 2: "Black", 3: "Hispanic"}).fillna("Other")

mf = MetricFrame(
    metrics={"MAE": mean_absolute_error, "R2": r2_score},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=race_labels,
)

print("\nFairlearn — model performance by racial group:")
print(mf.by_group.round(3))
print(f"\nMAE gap (max − min across groups): {mf.difference()['MAE']:.1f} months")
print(f"R²  gap (max − min across groups): {mf.difference()['R2']:.3f}")

# Save — rename the index column to "Race" for dashboard use
fl_df = mf.by_group.reset_index()
fl_df.columns = ["Race", "MAE", "R2"]
fl_df["MAE"] = fl_df["MAE"].round(1)
fl_df["R2"]  = fl_df["R2"].round(3)

fl_df.to_csv("data/processed/fairlearn_metrics.csv", index=False)
print("Saved → data/processed/fairlearn_metrics.csv")
