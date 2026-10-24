"""
Student Performance Predictor - Streamlit web app
=================================================

    streamlit run app.py

Four pages (menu at the top, so it is easy to reach on phones):
  Home | Dataset Analysis | Model Performance | Predict Marks

The app only reads files created by `python train.py`, so it never retrains
and opens quickly.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.data_prep import load_clean, load_raw, outlier_table

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# --------------------------------------------------------------------------- #
# Look and feel
# --------------------------------------------------------------------------- #
INK, BLUE, AMBER, GREEN, RED, GRID = "#1E2A44", "#3A5BD9", "#E0952B", "#1F8A5B", "#C2413B", "#DCE3EF"
MODEL_COLORS = {"Linear Regression": BLUE, "Random Forest": "#7A8BB5", "Decision Tree": "#B8C2D9"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, label {{ font-family: 'Lexend', 'Segoe UI', system-ui, sans-serif; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; text-wrap: balance; }}
h1 {{ font-weight: 700; }}
.block-container {{ padding-top: 4.2rem; max-width: 1180px; }}
p, li {{ max-width: 72ch; }}
[data-testid="stMetricValue"] {{ font-weight: 600; }}
.score {{ background: {BLUE}; color: #fff; border-radius: 10px; padding: 1.2rem 1.4rem; }}
.score .num {{ font-size: 3.4rem; font-weight: 700; line-height: 1; }}
.score .lab {{ opacity: .9; margin-top: .3rem; }}
.note {{ background: #FFF6E6; border-left: 4px solid {AMBER}; border-radius: 0 6px 6px 0; padding: .6rem .9rem; }}
</style>""", unsafe_allow_html=True)


def style(fig, height=380):
    fig.update_layout(height=height, font_family="Lexend", font_color=INK, margin=dict(l=10, r=10, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", legend_title_text="")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data
def get_data():
    raw, source = load_raw()
    df, report, _ = load_clean()
    return raw, df, report, source


@st.cache_resource
def get_model():
    return joblib.load(config.MODELS_DIR / "best_model.joblib")


@st.cache_data
def get_reports():
    meta = json.loads((config.MODELS_DIR / "meta.json").read_text())
    comp = pd.read_csv(config.REPORTS_DIR / "model_comparison.csv", index_col=0)
    imp = pd.read_csv(config.REPORTS_DIR / "feature_importance.csv")
    preds = pd.read_csv(config.REPORTS_DIR / "test_predictions.csv")
    return meta, comp, imp, preds


if not (config.MODELS_DIR / "best_model.joblib").exists():
    st.error("No trained model found. Open a terminal in the project folder, run `python train.py`, then reload this page.")
    st.stop()

meta, comp, imp, test_preds = get_reports()
L = config.LABELS


def grade(m: float) -> str:
    """Example grading scale; colleges use different scales."""
    return "A" if m >= 85 else "B" if m >= 70 else "C" if m >= 55 else "D" if m >= 40 else "F"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def home():
    st.title("Student Performance Predictor")
    st.write("This app predicts a student's final exam marks from four everyday factors: daily study hours, "
             "class attendance, sleep, and marks in the previous exam. It was built with Python, scikit-learn and Streamlit "
             "as a machine learning project.")
    tm = meta["test_metrics"]
    st.subheader(f"Best model: {meta['best_model']}")
    c = st.columns(4)
    c[0].metric("Accuracy (±5 marks)", f"{tm['Accuracy (±5 marks)']:.1%}", help="Share of test students whose prediction was within 5 marks of their real mark.")
    c[1].metric("MAE", f"{tm['MAE']:.2f} marks", help="Average size of the error.")
    c[2].metric("RMSE", f"{tm['RMSE']:.2f} marks", help="Typical error; punishes big mistakes more.")
    c[3].metric("R² score", f"{tm['R2']:.3f}", help="Share of the differences in marks the model explains (1 = perfect).")
    st.caption(f"Measured on {meta['n_test']} test students the model never saw during training.")

    st.subheader("How it works")
    st.markdown(
        "1. **Clean the data**: remove duplicates, fix impossible values, fill missing values.\n"
        "2. **Explore it**: charts show how each factor relates to marks.\n"
        "3. **Engineer features**: add study-with-attendance, diminishing returns of study, and distance from 8 hours of sleep.\n"
        "4. **Train and compare** Linear Regression, Decision Tree and Random Forest with 5-fold cross-validation.\n"
        "5. **Pick the best** model and **predict** marks for new students instantly.")
    st.markdown(f"<div class='note'><b>Data used:</b> {meta['data_source']}</div>", unsafe_allow_html=True)

    st.subheader("Go to")
    c = st.columns(3)
    c[0].page_link(PAGES["predict"], label="Predict marks", icon=":material/calculate:", use_container_width=True)
    c[1].page_link(PAGES["dataset"], label="Dataset analysis", icon=":material/bar_chart:", use_container_width=True)
    c[2].page_link(PAGES["performance"], label="Model performance", icon=":material/insights:", use_container_width=True)
    st.caption("On a phone, the menu is also under the » button at the top left.")


def dataset_analysis():
    raw, df, report, source = get_data()
    st.title("Dataset Analysis")
    st.caption(source)

    c = st.columns(4)
    c[0].metric("Rows loaded", f"{report['rows_loaded']:,}")
    c[1].metric("Duplicates removed", report["duplicates_removed"])
    c[2].metric("Impossible values fixed", sum(report["impossible_values_set_missing"].values()))
    c[3].metric("Rows after cleaning", f"{report['rows_after_cleaning']:,}")

    with st.expander("Data cleaning details", expanded=False):
        clean_tbl = pd.DataFrame({
            "Missing in raw file": report["missing_before"],
            "Impossible values found": report["impossible_values_set_missing"],
        }).rename(index=L).fillna(0).astype(int)
        st.dataframe(clean_tbl, width="stretch")
        st.write("Impossible values (for example attendance above 100% or negative study hours) are treated as "
                 "data-entry errors and turned into missing values. Missing inputs are then filled with the median of "
                 "the training data inside the model pipeline, so test data never influences them.")
        st.write("**Outliers (IQR rule).** These are unusual but possible values, so they are kept.")
        st.dataframe(outlier_table(df), width="stretch", hide_index=True)

    st.subheader("Summary statistics")
    st.dataframe(df.describe().T.rename(index=L).round(2), width="stretch")

    st.subheader("How each factor is spread")
    col = st.selectbox("Choose a column", config.FEATURES + [config.TARGET], format_func=lambda c: L[c])
    fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=[BLUE], labels=L)
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    fig.update_layout(title=f"Distribution of {L[col].lower()}", yaxis_title="Students")
    st.plotly_chart(style(fig, 340), width="stretch")

    st.subheader("What goes with higher marks?")
    corr = df[config.FEATURES + [config.TARGET]].corr().round(2)
    fig = px.imshow(corr.rename(index=L, columns=L), text_auto=True, color_continuous_scale="RdBu", zmin=-1, zmax=1)
    fig.update_layout(title="Correlation between factors (1 = move together, -1 = opposite)")
    st.plotly_chart(style(fig, 440), width="stretch")
    r = corr[config.TARGET].drop(config.TARGET).sort_values(ascending=False)
    st.write("Correlation with final marks: " + ", ".join(f"**{L[k]}** {v:+.2f}" for k, v in r.items()) +
             ". Correlation only measures straight-line links, so it underrates curved patterns like sleep (below).")

    st.subheader("Each factor against final marks")
    st.caption("Dots are students; the line is the average mark within each range of the factor.")
    cols = st.columns(2)
    for i, f in enumerate(config.FEATURES):
        d = df[[f, config.TARGET]].dropna()
        bins = pd.cut(d[f], 12)
        trend = d.groupby(bins, observed=True).agg(x=(f, "mean"), y=(config.TARGET, "mean"), n=(f, "size"))
        trend = trend[trend["n"] >= 8]
        fig = px.scatter(d, x=f, y=config.TARGET, opacity=0.35, color_discrete_sequence=["#9AA9CC"], labels=L)
        fig.add_trace(go.Scatter(x=trend["x"], y=trend["y"], mode="lines+markers", name="Average",
                                 line=dict(color=AMBER, width=3)))
        fig.update_layout(title=f"{L[f]} vs final marks", showlegend=False)
        cols[i % 2].plotly_chart(style(fig, 330), width="stretch")

    st.markdown(
        "**What the charts show, and the features engineered from them:**\n"
        "- **Study hours:** marks rise quickly for the first few hours, then more slowly → `log_study`.\n"
        "- **Sleep:** marks peak around 7–9 hours and fall on both sides (an upside-down U) → `sleep_gap` = distance from 8 hours.\n"
        "- **Study and attendance together:** study pays off more for students who attend → `effective_study` = study hours × attendance.\n"
        "- **Previous marks:** a steady straight-line link; past performance is a strong guide.")

    st.subheader("Marks by attendance band")
    d = df.dropna(subset=["attendance"]).copy()
    d["band"] = pd.cut(d["attendance"], [0, 70, 80, 90, 100], labels=["Below 70%", "70–80%", "80–90%", "90–100%"])
    fig = px.box(d, x="band", y=config.TARGET, color_discrete_sequence=[BLUE], labels={"band": "Attendance", **L},
                 category_orders={"band": ["Below 70%", "70–80%", "80–90%", "90–100%"]})
    fig.update_layout(title="Higher attendance bands have higher typical marks")
    st.plotly_chart(style(fig, 360), width="stretch")


def model_performance():
    st.title("Model Performance")
    best = meta["best_model"]
    st.write(f"Three models were trained on {meta['n_train']} students and tested on {meta['n_test']} unseen students. "
             f"**{best}** was chosen. {meta['selection_reason']}")

    show = comp[["CV RMSE", "Accuracy (±5 marks)", "MAE", "RMSE", "R2", "Best parameters"]].copy()
    show = show.sort_values("CV RMSE")
    st.dataframe(show.style.format({"CV RMSE": "{:.2f}", "Accuracy (±5 marks)": "{:.1%}", "MAE": "{:.2f}",
                                    "RMSE": "{:.2f}", "R2": "{:.3f}"})
                 .highlight_min(subset=["CV RMSE", "MAE", "RMSE"], color="#DCE6FF")
                 .highlight_max(subset=["Accuracy (±5 marks)", "R2"], color="#DCE6FF"), width="stretch")
    st.caption("CV RMSE = average error across 5-fold cross-validation on the training set (used to choose). "
               "Other columns are on the test set. Lower MAE/RMSE is better; higher accuracy and R² are better.")

    with st.expander("What do these numbers mean?"):
        st.markdown(
            f"- **Accuracy (±{meta['accuracy_tolerance']} marks):** the share of students whose predicted mark was within {meta['accuracy_tolerance']} marks of the real one. "
            "Ordinary accuracy is for yes/no answers, so for marks we count a prediction as correct when it is close enough.\n"
            "- **MAE (mean absolute error):** on average, how many marks the prediction is off by.\n"
            "- **RMSE (root mean squared error):** like MAE, but big mistakes count extra.\n"
            "- **R² score:** how much of the difference between students' marks the model explains. 1 is perfect, 0 is no better than guessing the average.")

    c1, c2 = st.columns(2)
    long = comp.reset_index()[["Model", "MAE", "RMSE"]].melt(id_vars="Model", var_name="Metric", value_name="Marks")
    fig = px.bar(long, x="Metric", y="Marks", color="Model", barmode="group", text_auto=".2f", color_discrete_map=MODEL_COLORS)
    fig.update_layout(title="Error in marks (lower is better)")
    c1.plotly_chart(style(fig, 360), width="stretch")
    long = comp.reset_index()[["Model", "Accuracy (±5 marks)", "R2"]].melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group", text_auto=".2f", color_discrete_map=MODEL_COLORS)
    fig.update_layout(title="Accuracy and R² (higher is better)", yaxis_range=[0, 1])
    c2.plotly_chart(style(fig, 360), width="stretch")

    st.subheader(f"{best}: predicted vs real marks")
    c1, c2 = st.columns(2)
    p = test_preds[f"pred_{best}"]
    a = test_preds["actual"]
    fig = px.scatter(x=a, y=p, opacity=0.6, color_discrete_sequence=[BLUE], labels={"x": "Real marks", "y": "Predicted marks"})
    lo, hi = float(min(a.min(), p.min())) - 2, float(max(a.max(), p.max())) + 2
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=INK, dash="dash"), name="Perfect prediction"))
    fig.add_trace(go.Scatter(x=[lo, hi, hi, lo], y=[lo + 5, hi + 5, hi - 5, lo - 5], fill="toself", mode="none",
                             fillcolor="rgba(224,149,43,0.15)", name="Within ±5 marks"))
    fig.update_layout(title="Points near the dashed line are good predictions", legend=dict(orientation="h", y=-0.2))
    c1.plotly_chart(style(fig, 400), width="stretch")
    err = p - a
    fig = px.histogram(x=err, nbins=30, color_discrete_sequence=[BLUE], labels={"x": "Prediction error (marks)"})
    fig.update_traces(marker_line_color="white", marker_line_width=1)
    fig.add_vline(x=0, line_color=INK, line_dash="dash")
    fig.update_layout(title="Errors are centred on zero (no systematic bias)", yaxis_title="Students")
    c2.plotly_chart(style(fig, 400), width="stretch")

    st.subheader("Which factors matter most?")
    d = imp.copy()
    d["feature"] = d["feature"].map(L)
    fig = px.bar(d.sort_values("importance"), x="importance", y="feature", orientation="h", error_x="std",
                 color_discrete_sequence=[BLUE], labels={"importance": "Increase in RMSE when the factor is scrambled (marks)", "feature": ""})
    fig.update_layout(title="Permutation importance on the test set")
    st.plotly_chart(style(fig, 320), width="stretch")
    st.caption("Each factor is shuffled in turn; the bigger the rise in error, the more the model relies on it.")

    st.subheader("Did feature engineering help?")
    fe = meta["feature_engineering_effect"]
    before, after = fe["without engineered features"]["CV RMSE"], fe["with engineered features"]["CV RMSE"]
    c = st.columns(3)
    c[0].metric("Linear Regression, 4 raw inputs", f"{before:.2f} RMSE")
    c[1].metric("With engineered features", f"{after:.2f} RMSE", delta=f"{after - before:.2f} marks", delta_color="inverse")
    c[2].metric("Error reduced by", f"{100 * (before - after) / before:.0f}%")
    st.caption("Cross-validated RMSE on the training set. The added features let a straight-line model follow curves "
               "(diminishing returns of study, the sleep U-shape) and the study × attendance interaction.")


def predict_page():
    st.title("Predict Marks")
    st.write("Move the sliders. The prediction updates instantly.")
    model = get_model()
    rng = meta["feature_ranges"]
    tm = meta["test_metrics"]

    def slider(f, step, fmt):
        lo, hi = config.VALID_RANGES[f]
        lo = max(lo, np.floor(rng[f]["min"]))
        hi = min(hi, np.ceil(rng[f]["max"]))
        return st.slider(L[f], float(lo), float(hi), float(round(rng[f]["median"] / step) * step), step=step, format=fmt, key=f)

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        study = slider("study_hours", 0.5, "%.1f h/day")
        attendance = slider("attendance", 1.0, "%.0f%%")
        sleep = slider("sleep_hours", 0.5, "%.1f h/night")
        previous = slider("previous_marks", 1.0, "%.0f")

    x = pd.DataFrame([[study, attendance, sleep, previous]], columns=config.FEATURES)
    pred = float(np.clip(model.predict(x)[0], 0, 100))
    err = tm["RMSE"]

    with c2:
        st.markdown(f"<div class='score'><div class='num'>{pred:.1f}</div>"
                    f"<div class='lab'>Predicted final marks out of 100 · grade {grade(pred)}</div></div>", unsafe_allow_html=True)
        st.write("")
        st.write(f"Likely range: **{max(0, pred - err):.0f} – {min(100, pred + err):.0f} marks** "
                 f"(± one typical error of {err:.1f} marks).")
        fig = go.Figure(go.Indicator(mode="gauge", value=pred, gauge=dict(
            axis=dict(range=[0, 100]), bar=dict(color=BLUE, thickness=0.35),
            steps=[dict(range=[0, 40], color="#F6DCDA"), dict(range=[40, 55], color="#FBEBD3"),
                   dict(range=[55, 70], color="#F1F4EA"), dict(range=[70, 85], color="#E3EEE6"),
                   dict(range=[85, 100], color="#CFE6D8")])))
        st.plotly_chart(style(fig, 220), width="stretch")
        st.caption("Example grading scale: A ≥ 85, B ≥ 70, C ≥ 55, D ≥ 40, F < 40. Your college's scale may differ.")

    st.subheader("What could change the result?")
    ideas = []
    for label, change in [("Study 1 more hour per day", {"study_hours": min(study + 1, 12)}),
                          ("Raise attendance to 95%", {"attendance": max(attendance, 95)}),
                          ("Sleep 8 hours", {"sleep_hours": 8.0})]:
        x2 = x.copy()
        for k, v in change.items():
            x2[k] = v
        delta = float(np.clip(model.predict(x2)[0], 0, 100)) - pred
        ideas.append({"Change": label, "Predicted change": delta})
    ideas = pd.DataFrame(ideas)
    fig = px.bar(ideas, x="Predicted change", y="Change", orientation="h", text_auto="+.1f",
                 color=ideas["Predicted change"] >= 0, color_discrete_map={True: GREEN, False: RED})
    fig.update_layout(title="Change in predicted marks", showlegend=False, xaxis_title="Marks", yaxis_title="")
    st.plotly_chart(style(fig, 260), width="stretch")

    hours = np.arange(0, 12.5, 0.5)
    curve = pd.DataFrame({"study_hours": hours, "attendance": attendance, "sleep_hours": sleep, "previous_marks": previous})
    curve["pred"] = np.clip(model.predict(curve[config.FEATURES]), 0, 100)
    fig = px.line(curve, x="study_hours", y="pred", labels={"study_hours": "Study hours per day", "pred": "Predicted marks"},
                  color_discrete_sequence=[BLUE])
    fig.add_trace(go.Scatter(x=[study], y=[pred], mode="markers", marker=dict(size=13, color=AMBER), name="You"))
    fig.update_layout(title="Predicted marks for different study hours (other inputs unchanged)", showlegend=False)
    st.plotly_chart(style(fig, 340), width="stretch")
    st.caption("These are the model's estimates from the training data, not guarantees. They show patterns, not cause and effect.")


# --------------------------------------------------------------------------- #
# Navigation (top bar works well on phones)
# --------------------------------------------------------------------------- #
PAGES = {
    "home": st.Page(home, title="Home", icon=":material/home:", default=True),
    "dataset": st.Page(dataset_analysis, title="Dataset Analysis", icon=":material/bar_chart:", url_path="dataset"),
    "performance": st.Page(model_performance, title="Model Performance", icon=":material/insights:", url_path="performance"),
    "predict": st.Page(predict_page, title="Predict Marks", icon=":material/calculate:", url_path="predict"),
}
nav = st.navigation(list(PAGES.values()), position="top")
nav.run()
