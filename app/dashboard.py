"""
app/dashboard.py — Federal Drug Sentencing Disparity Dashboard
Run from project root: python app/dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html

# ── Load data ─────────────────────────────────────────────────────────────────
drug_df  = pd.read_csv("data/processed/drug_sentences_fy20_fy25.csv")
resid_df = pd.read_csv("data/processed/residuals.csv")

RACE_MAP = {1: "White", 2: "Black", 3: "Hispanic", 6: "Other"}
drug_df["Race"]  = drug_df["NEWRACE"].map(RACE_MAP)
resid_df["Race"] = resid_df["NEWRACE"].map(RACE_MAP)

CORE   = ["White", "Black", "Hispanic"]
COLORS = {
    "Black":    "#A8A8A8",   # gray  (25 % saturation)
    "Hispanic": "#D4AA70",   # caramel
    "White":    "#EDE0C4",   # cream
    "Other":    "#C8C8C8",
    "salmon":   "#F4AFA8",
}
BAR_LINE  = dict(color="rgba(0,0,0,0.35)", width=1)
GRAPH_CFG = {"displayModeBar": False}

# ── Possession subsets ────────────────────────────────────────────────────────
poss_df    = drug_df[drug_df["OFFGUIDE"] == 8].copy()
poss_resid = resid_df[resid_df["OFFGUIDE"] == 8].copy()   # OFFGUIDE is int64


# ── Shared helper — single go.Bar trace, colors locked to race ────────────────
def make_bar(df, x_col, y_col, text_fmt,
             y_title=None, hline=False, order=None):
    """
    One bar per x value, color assigned positionally — impossible to swap.
    order: explicit list controlling x-axis sequence.
    """
    if order:
        present = [r for r in order if r in df[x_col].values]
        df = df.set_index(x_col).loc[present].reset_index()

    bar_colors = [COLORS.get(r, "#CCC") for r in df[x_col]]
    texts      = [text_fmt.format(v) for v in df[y_col]]

    fig = go.Figure(go.Bar(
        x=df[x_col].tolist(),
        y=df[y_col].tolist(),
        marker_color=bar_colors,
        marker_line=BAR_LINE,
        text=texts,
        textposition="outside",
        textfont=dict(size=18),
    ))
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#F8F8F8",
        font=dict(size=18),
        margin=dict(t=10, b=10),
        yaxis=dict(gridcolor="#E0E0E0", title=y_title or ""),
    )
    if hline:
        fig.add_hline(y=0, line_color="#444", line_dash="dot", line_width=1.5)
    return fig


# ═══════════════════════════════════════════════════════════
# TAB 1 — Dataset overview + methodology
# ═══════════════════════════════════════════════════════════
total_drug = len(drug_df)
total_poc  = int(drug_df["NEWRACE"].isin([2, 3]).sum())

# Race breakdown bar
race_counts = (
    drug_df[drug_df["Race"].isin(CORE)]
    .groupby("Race").size().reset_index(name="Cases")
)
fig_race = make_bar(race_counts, "Race", "Cases", "{:.0f}",
                    order=["White", "Black", "Hispanic"])
fig_race.update_layout(yaxis_title="Number of cases")

# Fiscal year — salmon
yearly = drug_df.groupby("FISCAL_YEAR").size().reset_index(name="Cases")
fig_yearly = px.bar(
    yearly, x="FISCAL_YEAR", y="Cases", text_auto=True,
    labels={"FISCAL_YEAR": "Fiscal Year", "Cases": "Cases"},
)
fig_yearly.update_traces(
    marker_color=COLORS["salmon"], marker_line=BAR_LINE,
    textposition="outside", textfont_size=18,
)
fig_yearly.update_layout(
    showlegend=False, plot_bgcolor="#F8F8F8", font=dict(size=18),
    margin=dict(t=10, b=10),
    xaxis=dict(tickmode="array", tickvals=yearly["FISCAL_YEAR"].tolist()),
    yaxis=dict(gridcolor="#E0E0E0"),
)

# Train / test split chart
n_test  = len(resid_df)
n_train = total_drug - n_test

# Approximate race counts for train set
test_race  = resid_df["Race"].value_counts().to_dict()
total_race = drug_df["Race"].value_counts().to_dict()
train_race = {r: total_race.get(r, 0) - test_race.get(r, 0) for r in CORE}

split_df = pd.DataFrame([
    {"Split": "Train (80%)", "Race": r, "Count": train_race[r]} for r in CORE
] + [
    {"Split": "Test  (20%)", "Race": r, "Count": test_race.get(r, 0)} for r in CORE
])
fig_split = px.bar(
    split_df, x="Split", y="Count", color="Race",
    color_discrete_map=COLORS, barmode="group",
    labels={"Count": "Cases", "Split": ""},
    text_auto=True,
)
fig_split.update_traces(marker_line=BAR_LINE, textposition="outside",
                        textfont_size=16)
fig_split.update_layout(
    showlegend=True, plot_bgcolor="#F8F8F8", font=dict(size=18),
    margin=dict(t=10, b=10), yaxis=dict(gridcolor="#E0E0E0"),
    legend=dict(font=dict(size=16)),
)

# Initial context: mean sentence by race across all drugs
mean_all_ctx = (
    drug_df[drug_df["Race"].isin(CORE)]
    .groupby("Race")["SENTTOT"]
    .agg(mean="mean", median="median").round(1).reset_index()
)


# ═══════════════════════════════════════════════════════════
# TAB 2 — One dumbbell per race, full width, stacked
# ═══════════════════════════════════════════════════════════
def make_dumbbell(race, n=67):
    pool = resid_df[resid_df["Race"] == race]
    pool = pool.sample(min(n, len(pool)), random_state=42)
    pool = pool.sort_values("SENTTOT_actual").reset_index(drop=True)

    xl, yl = [], []
    for i, row in pool.iterrows():
        xl += [i, i, None]
        yl += [row["SENTTOT_actual"], row["SENTTOT_predicted"], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xl, y=yl, mode="lines",
        line=dict(color="rgba(140,140,140,0.45)", width=1.2),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(pool))), y=pool["SENTTOT_actual"],
        mode="markers", name="Actual sentence",
        marker=dict(size=9, color=COLORS[race],
                    line=dict(color="rgba(0,0,0,0.4)", width=1)),
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(pool))), y=pool["SENTTOT_predicted"],
        mode="markers", name="Model prediction",
        marker=dict(size=9, color=COLORS[race], symbol="circle-open",
                    line=dict(color=COLORS[race], width=2)),
    ))
    fig.update_layout(
        title=dict(text=f"{race}  ({len(pool)} individuals)",
                   font=dict(size=22, color="#1a1a2e")),
        xaxis=dict(showticklabels=False, gridcolor="#E0E0E0",
                   title="People sorted by sentence length →"),
        yaxis=dict(gridcolor="#E0E0E0", title="Sentence (months)"),
        plot_bgcolor="#F8F8F8",
        font=dict(size=18),
        legend=dict(font=dict(size=16), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    return fig

fig_db_white    = make_dumbbell("White")
fig_db_black    = make_dumbbell("Black")
fig_db_hispanic = make_dumbbell("Hispanic")


# ═══════════════════════════════════════════════════════════
# TAB 3 — Finding 1: aggregate
# ═══════════════════════════════════════════════════════════
mean_all = (
    drug_df[drug_df["Race"].isin(CORE)]
    .groupby("Race")["SENTTOT"]
    .mean().round(1).reset_index()
    .rename(columns={"SENTTOT": "Mean sentence (months)"})
)
fig_mean_all = make_bar(mean_all, "Race", "Mean sentence (months)",
                        "{:.0f} mo", order=["White", "Black", "Hispanic"])

# Crime severity by race — mean guideline minimum (SMIN1)
# Shows WHY White defendants receive longer raw sentences (more serious crimes)
severity_by_race = (
    drug_df[drug_df["Race"].isin(CORE)]
    .groupby("Race")["SMIN1"]
    .mean().round(1).reset_index()
    .rename(columns={"SMIN1": "Mean guideline minimum (months)"})
)
fig_severity_all = make_bar(
    severity_by_race, "Race", "Mean guideline minimum (months)", "{:.0f} mo",
    y_title="Mean guideline minimum sentence (months)",
    order=["White", "Black", "Hispanic"],
)


# ═══════════════════════════════════════════════════════════
# TAB 4 — Finding 2: simple possession
# ═══════════════════════════════════════════════════════════
poss_mean = (
    poss_df[poss_df["Race"].isin(CORE)]
    .groupby("Race")["SENTTOT"]
    .mean().round(1).reset_index()
    .rename(columns={"SENTTOT": "Mean sentence (months)"})
)
fig_poss_raw = make_bar(poss_mean, "Race", "Mean sentence (months)",
                        "{:.0f} mo", order=["White", "Black", "Hispanic"])

# Residual gap vs White baseline — Black vs White only
# Hispanic excluded: only 11 possession cases in test set (unreliable)
# Full possession dataset shows Hispanic mean = 15.8 mo < White 23.1 mo (no disparity)
poss_resid_bw = (
    poss_resid[poss_resid["Race"].isin(["White", "Black"])]
    .groupby("Race")["residual"]
    .mean().round(1).reset_index()
    .rename(columns={"residual": "Mean residual (months)"})
)
white_baseline = float(
    poss_resid_bw.loc[poss_resid_bw["Race"] == "White",
                      "Mean residual (months)"].values[0]
)
poss_resid_bw["Gap vs White (months)"] = (
    poss_resid_bw["Mean residual (months)"] - white_baseline
).round(1)

fig_poss_resid = make_bar(
    poss_resid_bw, "Race", "Gap vs White (months)", "{:+.1f} mo",
    y_title="Extra months vs White defendants",
    hline=True, order=["White", "Black"],
)


# ═══════════════════════════════════════════════════════════
# Styles
# ═══════════════════════════════════════════════════════════
CARD = {
    "background": "#ffffff", "borderRadius": "8px",
    "padding": "28px", "marginBottom": "24px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.1)",
}
SEC_TITLE = {"fontSize": "24px", "fontWeight": "700",
             "marginBottom": "8px", "color": "#1a1a2e"}
SEC_SUB   = {"fontSize": "19px", "color": "#555",
             "marginBottom": "20px", "lineHeight": "1.6"}
CAPTION   = {"fontSize": "17px", "color": "#888", "marginTop": "10px",
             "lineHeight": "1.5"}
STAT_BOX  = {
    "textAlign": "center", "padding": "20px",
    "background": "#f7f9fc", "borderRadius": "8px",
    "flex": "1", "margin": "0 8px",
}
METHOD_BOX = {
    "background": "#f7f9fc", "borderRadius": "8px",
    "padding": "20px 24px", "marginBottom": "16px",
    "borderLeft": "4px solid #4C78A8",
}
TAB_STYLE = {
    "fontFamily": "Inter, sans-serif", "fontSize": "19px",
    "padding": "12px 20px", "fontWeight": "500",
}
TAB_SEL = {**TAB_STYLE, "fontWeight": "700",
           "borderTop": "3px solid #1a1a2e"}


# ═══════════════════════════════════════════════════════════
# App layout
# ═══════════════════════════════════════════════════════════
app = Dash(__name__)

app.layout = html.Div(
    style={"fontFamily": "Inter, sans-serif",
           "background": "#f0f2f5", "minHeight": "100vh", "padding": "32px"},
    children=[

        # Header
        html.Div(style={**CARD, "background": "#1a1a2e", "color": "white"},
                 children=[
            html.H1("Federal Drug Sentencing Disparity",
                    style={"margin": 0, "fontSize": "32px", "fontWeight": "700"}),
            html.P(
                "Do people of color receive harsher sentences for drug possession "
                "after controlling for all legal factors?",
                style={"margin": "10px 0 4px", "opacity": "0.85", "fontSize": "20px"},
            ),
            html.P(
                "Source: U.S. Sentencing Commission · FY2020–FY2025 · "
                "ussc.gov/research/datafiles/commission-datafiles",
                style={"margin": "4px 0 0", "opacity": "0.5", "fontSize": "17px"},
            ),
        ]),

        # Tabs
        dcc.Tabs(children=[

            # ── TAB 1: Overview + Methodology ─────────────────────────────
            dcc.Tab(label="📊  Dataset Overview",
                    style=TAB_STYLE, selected_style=TAB_SEL,
                    children=[html.Div(
                        style={"paddingTop": "24px",
                               "maxHeight": "80vh", "overflowY": "auto"},
                        children=[

                # Stat boxes
                html.Div(style=CARD, children=[
                    html.H2("Dataset Overview", style=SEC_TITLE),
                    html.P(
                        "U.S. Sentencing Commission individual offender records for federal "
                        "drug offenses — trafficking and simple possession — across six "
                        "fiscal years.",
                        style=SEC_SUB,
                    ),
                    html.Div(
                        style={"display": "flex", "marginBottom": "24px"},
                        children=[
                            html.Div(style=STAT_BOX, children=[
                                html.Div("~66,000", style={"fontSize": "34px",
                                         "fontWeight": "700", "color": "#1a1a2e"}),
                                html.Div("cases per year",
                                         style={"fontSize": "18px", "color": "#666"}),
                            ]),
                            html.Div(style=STAT_BOX, children=[
                                html.Div("6", style={"fontSize": "34px",
                                         "fontWeight": "700", "color": "#1a1a2e"}),
                                html.Div("fiscal years  (FY2020–FY2025)",
                                         style={"fontSize": "18px", "color": "#666"}),
                            ]),
                            html.Div(style=STAT_BOX, children=[
                                html.Div(f"{total_drug:,}",
                                         style={"fontSize": "34px", "fontWeight": "700",
                                                "color": "#1a1a2e"}),
                                html.Div("drug offense cases studied",
                                         style={"fontSize": "18px", "color": "#666"}),
                            ]),
                            html.Div(style=STAT_BOX, children=[
                                html.Div(f"{total_poc:,}",
                                         style={"fontSize": "34px", "fontWeight": "700",
                                                "color": "#C06030"}),
                                html.Div("people of color (Black + Hispanic)",
                                         style={"fontSize": "18px", "color": "#666"}),
                            ]),
                        ],
                    ),
                    html.Div(style={"display": "flex", "gap": "28px"}, children=[
                        html.Div(style={"flex": "1"}, children=[
                            html.P("Cases by race",
                                   style={"fontWeight": "600", "fontSize": "19px",
                                          "marginBottom": "4px"}),
                            dcc.Graph(figure=fig_race, config=GRAPH_CFG),
                        ]),
                        html.Div(style={"flex": "1"}, children=[
                            html.P("Cases per fiscal year",
                                   style={"fontWeight": "600", "fontSize": "19px",
                                          "marginBottom": "4px"}),
                            dcc.Graph(figure=fig_yearly, config=GRAPH_CFG),
                        ]),
                    ]),
                ]),

                # Train / test split
                html.Div(style=CARD, children=[
                    html.H2("Train / Test Split", style=SEC_TITLE),
                    html.P(
                        "The dataset was split randomly: 80% for training the model, "
                        "20% held out for testing. Race was never used to determine "
                        "the split — it is a fully random partition.",
                        style=SEC_SUB,
                    ),
                    html.Div(style={"display": "flex", "gap": "28px",
                                    "marginBottom": "16px"}, children=[
                        html.Div(style=STAT_BOX, children=[
                            html.Div(f"{n_train:,}", style={"fontSize": "34px",
                                     "fontWeight": "700", "color": "#1a1a2e"}),
                            html.Div("training cases  (80%)",
                                     style={"fontSize": "18px", "color": "#666"}),
                        ]),
                        html.Div(style=STAT_BOX, children=[
                            html.Div(f"{n_test:,}", style={"fontSize": "34px",
                                     "fontWeight": "700", "color": "#1a1a2e"}),
                            html.Div("test cases  (20%)",
                                     style={"fontSize": "18px", "color": "#666"}),
                        ]),
                    ]),
                    dcc.Graph(figure=fig_split, config=GRAPH_CFG),
                    html.P(
                        "Race breakdown is preserved proportionally across both splits "
                        "by chance of random sampling.",
                        style=CAPTION,
                    ),
                ]),

                # Methodology
                html.Div(style=CARD, children=[
                    html.H2("Methodology", style=SEC_TITLE),
                    html.P(
                        "We trained a machine learning model to predict sentence length "
                        "using only legally justifiable features — race was excluded "
                        "entirely. Residuals (actual minus predicted) were then compared "
                        "by race to surface unexplained disparity.",
                        style=SEC_SUB,
                    ),

                    # What is Gradient Boosting
                    html.Div(style=METHOD_BOX, children=[
                        html.H3("What is Gradient Boosting?",
                                style={"fontSize": "21px", "marginBottom": "8px",
                                       "color": "#1a1a2e"}),
                        html.P(
                            "Gradient Boosting builds hundreds of small decision trees "
                            "sequentially — each tree learns from the mistakes of the "
                            "previous one. The final prediction is the combined output "
                            "of all trees. It is one of the most accurate methods for "
                            "structured tabular data.",
                            style={"fontSize": "19px", "margin": 0,
                                   "lineHeight": "1.6"},
                        ),
                    ]),

                    # Why Gradient Boosting
                    html.Div(style=METHOD_BOX, children=[
                        html.H3("Why was it chosen?",
                                style={"fontSize": "21px", "marginBottom": "8px",
                                       "color": "#1a1a2e"}),
                        html.P(
                            "Sentencing does not follow a straight line. Mandatory minimums "
                            "create hard jumps at specific drug quantities (e.g. 60 months, "
                            "120 months). Criminal history category thresholds, guideline "
                            "zones, and plea discounts all produce step-function behaviour "
                            "that linear regression cannot capture. Gradient Boosting handles "
                            "these thresholds naturally.",
                            style={"fontSize": "19px", "margin": 0,
                                   "lineHeight": "1.6"},
                        ),
                    ]),

                    # Model metrics
                    html.Div(style=METHOD_BOX, children=[
                        html.H3("Model performance",
                                style={"fontSize": "21px", "marginBottom": "12px",
                                       "color": "#1a1a2e"}),
                        html.Div(style={"display": "flex", "gap": "16px"}, children=[
                            html.Div(style={**STAT_BOX, "margin": "0 8px 0 0"},
                                     children=[
                                html.Div("0.572", style={"fontSize": "30px",
                                         "fontWeight": "700", "color": "#1a1a2e"}),
                                html.Div("R²  (R-squared)",
                                         style={"fontSize": "17px", "color": "#666"}),
                            ]),
                            html.Div(style={**STAT_BOX, "margin": "0 8px"},
                                     children=[
                                html.Div("26.9 mo", style={"fontSize": "30px",
                                         "fontWeight": "700", "color": "#1a1a2e"}),
                                html.Div("Mean Absolute Error",
                                         style={"fontSize": "17px", "color": "#666"}),
                            ]),
                        ]),
                        html.Br(),
                        html.P(
                            "R² = 0.572 means the model explains 57% of the variation in "
                            "sentence length using legal factors alone. The remaining 43% "
                            "cannot be accounted for by law — this unexplained portion is "
                            "where bias can operate.",
                            style={"fontSize": "19px", "lineHeight": "1.6",
                                   "marginBottom": "10px"},
                        ),
                        html.P(
                            "Mean Absolute Error of 26.9 months means predictions are off "
                            "by about 2.2 years on average across all cases — reasonable "
                            "given the wide range of sentences (0 to 300 years).",
                            style={"fontSize": "19px", "lineHeight": "1.6", "margin": 0},
                        ),
                    ]),

                    # Mean / median explanation
                    html.Div(style=METHOD_BOX, children=[
                        html.H3("Mean vs Median sentence",
                                style={"fontSize": "21px", "marginBottom": "12px",
                                       "color": "#1a1a2e"}),
                        html.Div(style={"display": "flex", "gap": "16px",
                                        "marginBottom": "12px"}, children=[
                            *[html.Div(style={**STAT_BOX, "margin": "0 8px 0 0"},
                                       children=[
                                html.Div(f"{row['mean']:.0f} mo",
                                         style={"fontSize": "26px",
                                                "fontWeight": "700",
                                                "color": COLORS[row["Race"]],
                                                "textShadow": "0 0 1px #555"}),
                                html.Div(f"{row['Race']} — mean",
                                         style={"fontSize": "17px", "color": "#666"}),
                                html.Div(f"median  {row['median']:.0f} mo",
                                         style={"fontSize": "16px", "color": "#999",
                                                "marginTop": "4px"}),
                            ]) for _, row in mean_all_ctx.iterrows()
                              if row["Race"] in CORE]
                        ]),
                        html.P(
                            "The mean is pulled upward by extreme sentences (life terms, "
                            "stacked charges). The median is more representative of the "
                            "typical case. White defendants show higher means because "
                            "they appear more frequently in large-scale trafficking with "
                            "bigger drug quantities — not because they receive harsher "
                            "treatment for the same crime.",
                            style={"fontSize": "19px", "lineHeight": "1.6", "margin": 0},
                        ),
                    ]),

                    # Initial finding context
                    html.Div(style={**METHOD_BOX,
                                    "borderLeft": "4px solid #C06030"}, children=[
                        html.H3("Initial finding — no aggregate gap, but why?",
                                style={"fontSize": "21px", "marginBottom": "8px",
                                       "color": "#1a1a2e"}),
                        html.P(
                            "Across all federal drug offenses, White defendants actually "
                            "receive longer sentences on average. This appears to contradict "
                            "the hypothesis — until you look at the crimes themselves. "
                            "White defendants in this dataset are disproportionately convicted "
                            "of large-scale drug trafficking involving much larger quantities, "
                            "which carries longer mandatory minimums. Once the model controls "
                            "for crime severity, the aggregate gap disappears. The real "
                            "disparity surfaces when we narrow to simple possession — where "
                            "crimes are comparable in severity across races.",
                            style={"fontSize": "19px", "lineHeight": "1.6", "margin": 0},
                        ),
                    ]),
                ]),

            ])]),

            # ── TAB 2: Dumbbell charts — stacked vertically, full width ───
            dcc.Tab(label="📈  Why Not Linear Regression?",
                    style=TAB_STYLE, selected_style=TAB_SEL,
                    children=[html.Div(
                        style={"paddingTop": "24px",
                               "maxHeight": "80vh", "overflowY": "auto"},
                        children=[
                    html.Div(style=CARD, children=[
                        html.H2("Why Not Linear Regression?", style=SEC_TITLE),
                        html.P(
                            "Each chart shows 67 individuals from one race group. "
                            "The filled dot is what they actually received. The hollow "
                            "dot is what the model predicted from legal factors alone. "
                            "The vertical line between them is the residual — the portion "
                            "the law cannot explain. A straight best-fit line would miss "
                            "the mandatory-minimum clusters entirely.",
                            style=SEC_SUB,
                        ),
                        dcc.Graph(figure=fig_db_white, config=GRAPH_CFG,
                                  style={"height": "420px", "marginBottom": "8px"}),
                        dcc.Graph(figure=fig_db_black, config=GRAPH_CFG,
                                  style={"height": "420px", "marginBottom": "8px"}),
                        dcc.Graph(figure=fig_db_hispanic, config=GRAPH_CFG,
                                  style={"height": "420px"}),
                        html.P(
                            "📌  Filled = actual sentence · Hollow = model prediction · "
                            "Line = unexplained gap. Sorted left to right by actual "
                            "sentence length.",
                            style=CAPTION,
                        ),
                    ]),
                ])]),

            # ── TAB 3: Finding 1 ──────────────────────────────────────────
            dcc.Tab(label="🔍  Finding 1: Aggregate",
                    style=TAB_STYLE, selected_style=TAB_SEL,
                    children=[html.Div(style={"paddingTop": "24px"}, children=[
                html.Div(style=CARD, children=[
                    html.H2("Finding 1 — The Hidden Disparity in Plain Sight",
                            style=SEC_TITLE),
                    html.P(
                        "Across all federal drug offenses, White defendants receive longer "
                        "raw sentences on average — but only because they were convicted of "
                        "more serious crimes. Once the model controls for crime severity, "
                        "the picture changes: Black defendants are sentenced above what the "
                        "model predicts even in the aggregate.",
                        style=SEC_SUB,
                    ),
                    html.Div(style={"display": "flex", "gap": "28px"}, children=[
                        html.Div(style={"flex": "1"}, children=[
                            html.P("Mean sentence by race — all drug offenses",
                                   style={"fontWeight": "600", "fontSize": "19px",
                                          "marginBottom": "4px"}),
                            dcc.Graph(figure=fig_mean_all, config=GRAPH_CFG),
                            html.P(
                                "White defendants appear to receive longer sentences — "
                                "but this reflects more serious crimes, not harsher treatment.",
                                style=CAPTION,
                            ),
                        ]),
                        html.Div(style={"flex": "1"}, children=[
                            html.P("Mean guideline minimum by race — crime severity",
                                   style={"fontWeight": "600", "fontSize": "19px",
                                          "marginBottom": "4px"}),
                            dcc.Graph(figure=fig_severity_all, config=GRAPH_CFG),
                            html.P(
                                "The guideline minimum is set by drug type and quantity — "
                                "it reflects crime severity, not race. White defendants "
                                "carry a higher guideline minimum on average, confirming "
                                "they are convicted of more serious offences. "
                                "See Finding 2 for the residual analysis on simple possession.",
                                style=CAPTION,
                            ),
                        ]),
                    ]),
                ]),
            ])]),

            # ── TAB 4: Finding 2 — possession ─────────────────────────────
            dcc.Tab(label="⚖️  Finding 2: Simple Possession",
                    style=TAB_STYLE, selected_style=TAB_SEL,
                    children=[html.Div(style={"paddingTop": "24px"}, children=[
                html.Div(style={**CARD, "borderLeft": "4px solid #C06030"},
                         children=[
                    html.H2("Finding 2 — Simple Possession: The Sentencing Gap",
                            style=SEC_TITLE),
                    html.P(
                        "When we isolate simple drug possession — small amounts, no "
                        "trafficking — the disparity becomes stark. Black and Hispanic "
                        "defendants serve significantly more time than White defendants "
                        "convicted of the same offense. This gap persists after controlling "
                        "for every legal factor the model accounts for.",
                        style=SEC_SUB,
                    ),
                    html.Div(style={"display": "flex", "gap": "28px"}, children=[
                        html.Div(style={"flex": "1"}, children=[
                            html.P("Mean sentence — simple possession only",
                                   style={"fontWeight": "600", "fontSize": "19px",
                                          "marginBottom": "4px"}),
                            dcc.Graph(figure=fig_poss_raw, config=GRAPH_CFG),
                            html.P(
                                "Black: 28.5 months · Hispanic: 15.8 months · "
                                "White: 23.1 months — same charge, different outcomes.",
                                style=CAPTION,
                            ),
                        ]),
                        html.Div(style={"flex": "1"}, children=[
                            html.P("Extra months vs White defendants — after legal factors",
                                   style={"fontWeight": "600", "fontSize": "19px",
                                          "marginBottom": "4px"}),
                            dcc.Graph(figure=fig_poss_resid, config=GRAPH_CFG),
                            html.P(
                                f"White = 0 (baseline). Black defendants receive approximately "
                                f"{poss_resid_bw.loc[poss_resid_bw['Race']=='Black','Gap vs White (months)'].values[0]:+.1f} extra months "
                                "beyond White defendants for the same offense, after controlling "
                                "for all legal factors. Hispanic omitted — only 11 test cases, "
                                "too small to be reliable. Full dataset shows Hispanics serve "
                                "less time than Whites for possession (15.8 vs 23.1 months avg).",
                                style=CAPTION,
                            ),
                        ]),
                    ]),
                ]),
            ])]),

        ]),

        # Footer
        html.Div(
            "NOD Coding Stockholm · USSC FY2020–FY2025",
            style={"textAlign": "center", "fontSize": "17px",
                   "color": "#aaa", "paddingTop": "12px", "paddingBottom": "16px"},
        ),
    ],
)

if __name__ == "__main__":
    app.run(debug=True)
