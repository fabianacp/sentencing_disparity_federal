# Skills — Reusable Patterns for Claude Code

## Train/test split
Always use stratify=y to ensure balanced classes across split.

## Sklearn Pipeline structure
Always wrap preprocessing + model in a single Pipeline object.
Preprocessing steps: imputation → scaling → OneHotEncoding.

## GridSearchCV template
Use cv=5 for K-fold cross validation.
Score on neg_mean_absolute_error for regression problems.

## Dash tab template
Three tabs: Data Overview, Model Comparison, Flagged Cases.
Load model from models/best_model.pkl at app startup.

## Code comment rule
Every block of code gets one plain English comment above it
explaining what it does and why — not how.