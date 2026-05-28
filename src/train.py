""
src/train.py
------------
Predict federal drug offense sentence length (SENTTOT in months) using only
legally justifiable features — race is excluded from training entirely.

After training, residuals (actual − predicted) are compared between
people of color (Black + Hispanic) and White defendants to surface
sentencing disparity that cannot be explained by legal factors alone.

Research question:
    Do people of color receive harsher drug sentences than the model
    predicts, after controlling for criminal history, offense severity,
    plea type, district, and other legal factors?
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score


# ── 1. Load processed dataset ─────────────────────────────────────────────────
# Drug offenses only (OFFGUIDE 7 & 8), FY2020–FY2025, all sentence lengths
df = pd.read_csv("data/processed/drug_sentences_fy20_fy25.csv")
df = df.dropna(subset=["SENTTOT"])
print(f"Loaded {len(df):,} rows across FY2020–FY2025")


# ── 2. Feature definitions — race excluded entirely ───────────────────────────
# Numeric: legally relevant quantities the judge considers
NUMERIC_FEATURES = [
    "AGE",          # defendant age (mitigating factor)
    "SMIN1",        # guideline minimum sentence (months)
    "SMAX1",        # guideline maximum sentence (months)
    "GLMIN",        # computed guideline range minimum
    "GLMAX",        # computed guideline range maximum
    "CRIMPTS",      # criminal history points
    "TOTCHPTS",     # total offense level points
    "XCRHISSR",     # criminal history category (I–VI)
    "NUMDEPEN",     # number of dependents (mitigating)
    "NOCOUNTS",     # number of counts charged
    "DRUGMIN",      # statutory drug mandatory minimum (months)
    "MONSEX",       # gender: 0=male, 1=female (binary → numeric)
    "WEAPON",       # weapon involved: 0/1
    "NEWEDUC",      # education level 1–6 (ordinal → numeric)
]

# Categorical: non-ordinal legal factors
CATEGORICAL_FEATURES = [
    "ZONE",         # guideline zone A/B/C/D (severity bracket)
    "DISTRICT",     # federal district court (geographic variation)
    "DISPOSIT",     # 1=guilty plea, 3=jury trial, 4=bench trial
    "CITIZEN",      # citizenship status (affects enhancement risk)
    "FISCAL_YEAR",  # year of sentencing (controls for policy changes)
    "OFFGUIDE",     # 7=drug trafficking, 8=simple possession — model must know this
]

TARGET   = "SENTTOT"   # total sentence imposed in months
RACE_COL = "NEWRACE"   # held aside — never passed to the model


# ── 3. Build X, y, and race series (race kept separate) ──────────────────────
# Convert DISTRICT and FISCAL_YEAR to string so OneHotEncoder handles them cleanly
df["DISTRICT"]    = df["DISTRICT"].astype(str)
df["FISCAL_YEAR"] = df["FISCAL_YEAR"].astype(str)

# Only keep columns the model will actually see
available_numeric = [f for f in NUMERIC_FEATURES if f in df.columns]
available_categ   = [f for f in CATEGORICAL_FEATURES if f in df.columns]

# Convert OFFGUIDE to string so OneHotEncoder treats it as a label not a number
df["OFFGUIDE"] = df["OFFGUIDE"].astype(str)

X    = df[available_numeric + available_categ]
y    = df[TARGET]
race = df[RACE_COL]   # held aside for residual analysis only


# ── 4. Train / test split — simple random, no race stratification ─────────────
# Race is NOT used to split — the model must never know race at any stage
X_train, X_test, y_train, y_test, race_train, race_test = train_test_split(
    X, y, race, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ── 5. Preprocessing pipeline ─────────────────────────────────────────────────
# Numeric: impute missing with median, then scale
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])

# Categorical: impute missing with most frequent, then one-hot encode
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, available_numeric),
    ("cat", categorical_pipe, available_categ),
])


# ── 6. Define 4 models with hyperparameter grids ──────────────────────────────
# Linear regression excluded: sentencing follows step-functions and thresholds,
# not a linear relationship — tree-based models are more appropriate here.
MODELS = {
    "decision_tree": (
        DecisionTreeRegressor(random_state=42),
        {
            "model__max_depth":        [5, 10, 15],
            "model__min_samples_leaf": [10, 20, 50],
        },
    ),
    "random_forest": (
        RandomForestRegressor(random_state=42, n_jobs=-1),
        {
            "model__n_estimators":     [100, 200],
            "model__max_depth":        [10, 20, None],
            "model__min_samples_leaf": [10, 20],
        },
    ),
    "gradient_boosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "model__n_estimators":  [100, 200],
            "model__max_depth":     [3, 5],
            "model__learning_rate": [0.05, 0.1],
        },
    ),
    "extra_trees": (
        ExtraTreesRegressor(random_state=42, n_jobs=-1),
        {
            "model__n_estimators":     [100, 200],
            "model__max_depth":        [10, 20, None],
            "model__min_samples_leaf": [10, 20],
        },
    ),
}


# ── 7. Train all models with 5-fold GridSearchCV ──────────────────────────────
results = {}
for name, (estimator, param_grid) in MODELS.items():
    print(f"\nTraining {name}...")

    pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
    gs   = GridSearchCV(pipe, param_grid, cv=5, scoring="r2", n_jobs=-1)
    gs.fit(X_train, y_train)

    y_pred = gs.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)

    results[name] = {"gs": gs, "y_pred": y_pred, "mae": mae, "r2": r2}
    print(f"  Best params : {gs.best_params_}")
    print(f"  MAE         : {mae:.1f} months  ({mae/12:.1f} yrs)")
    print(f"  R²          : {r2:.3f}")


# ── 8. Select and save the best model ────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["r2"])
best      = results[best_name]
print(f"\n{'─'*55}")
print(f"Best model : {best_name}")
print(f"R²         : {best['r2']:.3f}")
print(f"MAE        : {best['mae']:.1f} months  ({best['mae']/12:.1f} yrs)")

os.makedirs("models", exist_ok=True)
joblib.dump(best["gs"].best_estimator_, "models/best_model.pkl")
print("Saved → models/best_model.pkl")


# ── 9. Residual analysis: simple possession only, people of color vs white ────
# Race introduced HERE for the first time — only to read results, never to train.
# Model trained on ALL drug cases (robust); residuals sliced to simple possession
# (OFFGUIDE=8) where small-amount bias is most clearly observable.
#
# People of color = Black (NEWRACE=2) + Hispanic (NEWRACE=3)
# White           = NEWRACE=1

y_pred_best = best["y_pred"]
residuals   = y_test.values - y_pred_best   # positive = sentenced MORE than predicted

# Build a results frame for slicing
results_df = X_test.copy()
results_df["SENTTOT_actual"]    = y_test.values
results_df["SENTTOT_predicted"] = y_pred_best
results_df["residual"]          = residuals
results_df["NEWRACE"]           = race_test.values
results_df["POC"]               = race_test.isin([2, 3]).values

# ── Slice to simple possession only ──────────────────────────────────────────
poss = results_df[results_df["OFFGUIDE"] == "8"].copy()
print(f"\n{'─'*60}")
print(f"Simple possession cases in test set: {len(poss)}")

poc_mask   = poss["POC"].values
white_mask = (poss["NEWRACE"] == 1).values
residuals_poss = poss["residual"].values

print(f"\n{'─'*60}")
print("RESIDUAL ANALYSIS — Simple Possession (OFFGUIDE=8) only")
print("Actual sentence minus model prediction (months)")
print(f"{'─'*60}")
print(f"{'Group':<22} {'n':>5} {'Mean':>8} {'Median':>8} {'Std':>8}")
print(f"{'─'*60}")

group_residuals = {}
for label, mask in [("White", white_mask), ("People of Color", poc_mask)]:
    r = residuals_poss[mask]
    group_residuals[label] = r
    print(
        f"{label:<22} {mask.sum():>5} "
        f"{r.mean():>+8.1f} {np.median(r):>+8.1f} {r.std():>8.1f}"
    )

# Disparity gap: extra months POC receive vs White, unexplained by legal factors
gap_months = group_residuals["People of Color"].mean() - group_residuals["White"].mean()
gap_years  = gap_months / 12

print(f"\nDisparity gap (POC − White) : {gap_months:+.1f} months  ({gap_years:+.2f} yrs)")
if gap_months > 0:
    print("→ People of color sentenced LONGER than model predicts for simple possession.")
    print("  This gap cannot be explained by legal factors alone.")
elif gap_months < 0:
    print("→ Model over-predicts for POC — they receive less than legally expected.")
else:
    print("→ No measurable disparity detected.")

# ── Also print all-drug summary for reference ─────────────────────────────────
print(f"\n{'─'*60}")
print("ALL DRUG OFFENSES — reference (training universe)")
for label, mask in [
    ("White",           results_df["NEWRACE"].eq(1).values),
    ("People of Color", results_df["POC"].values),
]:
    r = results_df["residual"].values[mask]
    print(f"  {label:<20} n={mask.sum():>4}  mean={r.mean():>+7.1f}  median={np.median(r):>+7.1f}")

# Save full residuals for dashboard use
os.makedirs("data/processed", exist_ok=True)
results_df.to_csv("data/processed/residuals.csv", index=False)
print("\nSaved residuals → data/processed/residuals.csv")
