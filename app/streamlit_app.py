"""Read-only viewer for locally generated experiment measurements."""
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="High-Beta Stock Analysis", layout="wide")
st.title("Analysis of High-Beta Stocks")
st.caption("AI-infrastructure equity research · chronological evaluation")
st.write("Compare models and market-only versus market-plus-FinBERT experiments."
         " These are research results, not live signals or trading advice.")
path = Path("reports/fold_metrics.csv")
if not path.exists():
    path = Path("reports/real-market-pilot/market_only_metrics.csv")
if not path.exists():
    st.info("No evaluation results yet. Run high-beta evaluate --prices data/raw/prices.csv")
    st.stop()
results = pd.read_csv(path)
if results.empty:
    st.warning("Metrics file is empty")
    st.stop()
left, right = st.columns([2, 1])
with left:
    st.subheader("Average precision across folds")
    eligible = results.loc[results["model"].ne("linear_regression_abs_return")]
    if "pr_auc" in eligible:
        st.dataframe(eligible.pivot_table(
            index="model", columns="ablation", values="pr_auc", aggfunc="mean"
        ))
with right:
    st.subheader("Experiment coverage")
    st.metric("Evaluation folds", results["fold"].nunique())
    st.metric("Models", results["model"].nunique())
st.subheader("Full fold-level measurements")
st.dataframe(results, use_container_width=True)
st.caption("Consider class prevalence, source coverage, data revisions and uncertainty.")
