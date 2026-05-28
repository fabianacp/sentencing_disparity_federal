# Sentencing Disparity — U.S. Federal Courts

A supervised machine learning model that identifies sentencing disparities in U.S. federal courts. We train on historical sentences, then flag cases where actual sentences diverge significantly from model predictions — exposing potential bias by race, offense type, and legal representation.

Built at NOD Coding Stockholm.

---

## Dataset

**U.S. Sentencing Commission (USSC) — FY2020–FY2025 Non-Identifiable Data**

- **Source:** https://www.ussc.gov/research/datafiles/commission-datafiles
- **Files:** `data/raw/opafy20nid.csv` … `data/raw/opafy25nid.csv`
- **Records:** ~66,000 real federal sentencing cases per year · 6 fiscal years
- **Processed:** `data/processed/drug_sentences_fy20_fy25.csv` — 8,372 drug offense cases
- **Target variable:** `SENTTOT` — total sentence imposed in months

---

## Project Structure

```
notebooks/01_eda.ipynb       — Exploratory data analysis
src/train.py                 — sklearn Pipeline, models, GridSearchCV
src/fairness_analysis.py     — SHAP feature importance + Fairlearn group metrics
app/dashboard.py             — Dash dashboard (6 tabs)
models/best_model.pkl        — Saved best model (joblib)
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dashboard Screenshots

### Tab 1 — Dataset Overview
![Dataset Overview](images/sentencing_tab_1_A.png)
![Train / Test Split](images/sentencing_Tab_1_B.png)
![Methodology](images/sentencing_Tab_1_C.png)
![Model Performance & Initial Finding](images/sentencing_Tab_1_D.png)

### Tab 2 — Why Not Linear Regression?
![Dumbbell — White](images/sentencing_Tab_2_A.png)
![Dumbbell — Black & Hispanic](images/sentencing_Tab_2_B.png)

### Tab 3 — Finding 1: Aggregate
![Finding 1](images/sentencing_Tab_3_A.png)

### Tab 4 — Finding 2: Simple Possession
![Finding 2](images/sentencing_Tab_4_A.png)

### Tab 5 — SHAP: What Drives Sentences?
![SHAP Feature Importance](images/sentencing_Tab_5_A.png)

### Tab 6 — Fairlearn: Model Fairness
![Fairlearn Group Metrics](images/sentencing_tab_6_A.png)

---

## How to Run

Run the three steps in order:

```bash
# 1. Train the model — saves models/best_model.pkl and data/processed/residuals.csv
python src/train.py

# 2. Generate SHAP and Fairlearn outputs
python src/fairness_analysis.py

# 3. Launch the dashboard at http://127.0.0.1:8050
python app/dashboard.py
```

## Dashboard

Six tabs:

| Tab | What it shows |
|---|---|
| Dataset Overview | Case counts, race breakdown, train/test split, methodology |
| Why Not Linear Regression? | Dumbbell charts — actual vs predicted per individual, by race |
| Finding 1: Aggregate | Mean sentence and crime severity by race across all drug offenses |
| Finding 2: Simple Possession | Sentencing gap for Black defendants after controlling for legal factors |
| SHAP: What Drives Sentences? | Top-10 features by mean absolute SHAP value — what the model actually learned |
| Fairlearn: Model Fairness | MAE and R² broken down by racial group — model accuracy audit |
