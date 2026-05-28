# Sentencing Disparity Federal — Claude Code Instructions

## What this project does
A supervised machine learning model that identifies sentencing disparities 
in U.S. federal courts. We train on historical sentences, then flag 
cases where actual sentences diverge significantly from model predictions — 
exposing potential bias by race, offense type, and legal representation.

## Dataset
U.S. Sentencing Commission (USSC) — FY2020–FY2025 Non-Identifiable Data
- ~66,000 real federal sentencing records per year across 6 fiscal years
- Files: data/raw/opafy20nid.csv … opafy25nid.csv
- Processed: data/processed/drug_sentences_fy20_fy25.csv (8,372 drug cases)
- Source: https://www.ussc.gov/research/datafiles/commission-datafiles
- Target variable: SENTTOT (total sentence in months)
- Key features: SMIN1/SMAX1 (guideline range), CRIMPTS, ZONE, DISPOSIT, DISTRICT

## Coding rules — follow these every time
- Use sklearn Pipeline for all preprocessing + model code
- Always add a comment above every code block explaining what it does
- Use the least amount of code possible — prefer built-in sklearn tools
- Never modify test data after the train/test split
- Save the best model to /models/best_model.pkl using joblib

## Project structure
- notebooks/01_eda.ipynb — Agent 1: data exploration only
- src/train.py — Agent 2: pipeline, models, GridSearchCV
- app/dashboard.py — Agent 3: Dash dashboard, display only

## NOD requirements checklist
- [ ] Real dataset, not synthetic
- [ ] Regression or classification — clearly stated
- [ ] EDA of features and target variable
- [ ] Baseline model defined
- [ ] Train/test split
- [ ] Preprocessing: scaling, OneHotEncoding, imputation
- [ ] At least 4 models tried on validation data
- [ ] GridSearchCV with K-fold cross validation
- [ ] Final model evaluated on test data once
- [ ] Trello board
- [ ] GitHub with README