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
notebooks/01_eda.ipynb   — Exploratory data analysis
src/train.py             — sklearn Pipeline, models, GridSearchCV
app/dashboard.py         — Dash dashboard
models/best_model.pkl    — Saved best model (joblib)
```

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
