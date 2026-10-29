# Architecture and data contracts

This system studies one ticker on one market session at a time. It predicts whether the next session has an unusually large absolute closing return. The prediction cutoff is 4 p.m. America/New_York in the current research prototype.

## Inputs and outputs

Price CSV: date, ticker, open, high, low, close, volume. Use consistent adjusted-price conventions; audit historical revisions and corporate actions.

Financial text CSV: ticker, published_at, headline, plus optional source. Dates must carry a reliable timezone and actual publication time. News is scored with ProsusAI/finbert into positive, negative and neutral probabilities.

The resulting stock-day panel contains market features, session cutoff, and future-return target fields. Target fields must never enter the model feature matrix.

## Pipeline

1. Download or load and validate prices for the candidate companies and reference assets (SPY, QQQ, SMH, VIX, BTC).
2. Compute returns, momentum, realized volatility, relative volume, intraday range, overnight gap and rolling 60-session SPY beta.
3. Align news to the earliest observed session whose prediction cutoff is at or after publication; calculate article count, sentiment mean, disagreement and missingness.
4. Create expanding chronological train/test folds, separated by a one-session gap.
5. Fit each ticker's extreme-move percentile on training returns only, then apply the same fitted threshold to later observations.
6. Fit preprocessing and supervised models on training observations and report out-of-sample metrics per model and fold.
7. Compare market-only and market-plus-FinBERT experiments with identical folds; analyze market regimes separately using train-fitted PCA and clustering.
8. Read saved experiment results in the local Streamlit application.

Reusable code is in src/high_beta_analysis/ rather than notebooks. The main components are data/, features/, models/, regimes/ and evaluation/.

## Point-in-time caveats

The implementation uses observed equity sessions rather than a verified exchange calendar. It does not yet model early closes, vendor data latency or symbol-history changes. UTC crypto bars may end after the equity close and are lagged. This is a framework for historical experiments, not a verified point-in-time market feed.
