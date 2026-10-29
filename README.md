# Analysis of High-Beta Stocks

Research code for studying **unusually large next-session price moves** in public companies connected to AI infrastructure, GPU cloud services, data centers, power-intensive compute, and adjacent businesses.

The question is not "What will this stock close at tomorrow?" It is: **Can information already available by today's market close help identify when tomorrow's absolute return is unusually large for that particular stock?** We compare structured market features with financial-news features and test whether more complex models actually improve on simple baselines under chronological evaluation.

This is an empirical machine-learning project, **not** an investment recommendation service or a live trading system. "High-beta" in the repository name describes the intended study population; membership in the candidate universe does not imply that a stock's beta has already been measured or exceeds one.

## Companies and market context

The initial, configurable study universe is:

- **AI cloud / GPU infrastructure:** Nebius (\`NBIS\`), CoreWeave (\`CRWV\`).
- **AI-data-center and power-capacity development:** Applied Digital (\`APLD\`), TeraWulf (\`WULF\`), Cipher Mining (\`CIFR\`), Core Scientific (\`CORZ\`).
- **Compute / mining-to-AI exposure:** IREN (\`IREN\`), Hut 8 (\`HUT\`).

Further candidates for comparative or sector analysis include Vertiv (\`VRT\`), Super Micro Computer (\`SMCI\`), NVIDIA (\`NVDA\`), Dell (\`DELL\`), Vistra (\`VST\`), and Eaton (\`ETN\`). They represent different parts of the infrastructure supply chain, **not** a single homogeneous industry. Final inclusion will depend on historic trading coverage, corporate-action continuity, and measured rolling beta. Recent listings do not have the same history as established stocks.

Market context begins with \`SPY\` (broad U.S. market and beta denominator), \`QQQ\` (technology), \`SMH\` (semiconductors), \`^VIX\` (volatility environment), and \`BTC-USD\` (crypto exposure). These are inputs, not necessarily prediction targets.

## Research questions

1. Do return, volume, volatility, market-beta and cross-asset features predict extreme-move risk beyond a constant baseline?
2. Does financial text, summarized with **FinBERT**, add measurable information beyond the market-only model?
3. Do prediction quality, class balance and model explanations differ across market regimes?
4. Are predicted probabilities reliable, and where do models fail?
5. Are stronger algorithms genuinely better out of sample than linear/logistic regression and small decision trees?

## System pipeline

\`\`\`mermaid
flowchart TD
    Prices[Daily OHLCV equities] --> Validate[Validate dates, tickers, and coverage]
    Context[SPY / QQQ / SMH / VIX / BTC] --> Validate
    Validate --> Features[Stock-day features at the 4 pm ET cutoff]
    News[Timestamped headlines / filings] --> Sentiment[FinBERT: positive, negative, neutral]
    Sentiment --> NewsFeatures[As-of article count, sentiment, disagreement]
    Features --> Panel[Time-aligned stock-day panel]
    NewsFeatures --> Panel
    Panel --> Folds[Expanding walk-forward splits and one-session gap]
    Folds --> Labels[Stock-specific extreme threshold fitted on TRAIN only]
    Labels --> Models[Regression / Logistic / SVM / Tree / Forest / MLP]
    Models --> Scores[PR-AUC / Recall / F1 / ROC-AUC / Brier / MAE]
    Scores --> Ablation[Market-only vs market + FinBERT]
    Panel --> Regimes[Train-only PCA and K-means]
    Regimes --> Reliability[Regime analysis and failure inspection]
    Ablation --> Dashboard[Read-only Streamlit experiment viewer]
    Reliability --> Dashboard
\`\`\`

The architecture deliberately separates **data acquisition**, **features**, **labels**, **training**, **evaluation**, and **presentation**. Jupyter is for exploration; reusable calculations belong in \`src/high_beta_analysis/\`.

### 1. Stock-day observations

Each row corresponds to one ticker on one observed regular-session date. Market features include daily and absolute returns, rolling momentum (5/20 sessions), realized volatility (5/20), lagged volume ratio, daily range, overnight gap, rolling 60-session beta to SPY, relative market returns and cross-asset context.

A feature for date \`t\` must be known by the configured prediction cutoff on date \`t\`. The target is the **absolute closing-price return from session \`t\` to the next session**; it is never an input feature.

### 2. What counts as an extreme move?

A fixed 5% move does not mean the same thing for every ticker. For each training window and ticker, we estimate a threshold from its historical next-session absolute returns (90th percentile by default). A test-period move is labeled extreme only if its absolute return **exceeds that train-fitted threshold**.

For illustration, if a stock's training-only threshold is 7%, a later +9% day or -9% day is an extreme move, while a +2% day is not. This illustration is **not** a measured threshold for a real stock.

Estimating thresholds from the full dataset would leak future return distributions into training. We therefore re-estimate them independently inside each chronological fold.

### 3. Financial-language features (FinBERT)

\`ProsusAI/finbert\` returns three sentiment scores per article: positive, negative and neutral. We retain **all three**, not only the winning label, and aggregate articles by ticker and the first eligible session cutoff **at or after publication**. An article published at 4:30 p.m. ET on Thursday cannot affect Thursday's 4 p.m. estimate; its first eligible session is Friday (or the next open session).

The current NLP features include article count, mean class probabilities, mean positive-minus-negative sentiment, sentiment disagreement (standard deviation), maximum absolute sentiment, and a **missing-news indicator**. Missing coverage must not be interpreted as proof that no relevant news occurred.

The text pipeline accepts a licensed, user-supplied news CSV with \`ticker\`, \`published_at\` (timezone-aware), and \`headline\`. It does **not** secretly scrape or redistribute copyrighted news. Model weights download from Hugging Face when FinBERT is first invoked. Check the model card and publisher-data licenses before running.

### 4. Modeling and course concepts

- **Dummy classifier:** a no-signal reference.
- **Linear regression:** secondary target, next-session absolute return (MAE).
- **Logistic regression:** interpretable linear classifier and probability baseline.
- **Decision tree:** information-gain/Gini-style recursive feature splits, with depth and leaf-size controls.
- **Random forest:** ensemble of trees to capture nonlinear relationships.
- **RBF SVM:** margin-based nonlinear classifier; raw decision scores are **not** treated as calibrated probabilities.
- **MLP:** a neural baseline, with optional pedagogical PyTorch implementation.
- **PCA and K-means:** unsupervised market-regime exploration; fitted on training data, applied to later dates.

Preprocessing, imputation and scaling are fit **inside each training fold**, not on the complete panel. Models use the same chronological splits and target definitions for fair comparisons.

### 5. Evaluation and reliability

The primary task is intentionally imbalanced. Plain accuracy would be misleading. We record precision, recall, F1, balanced accuracy, **average precision (PR-AUC)**, ROC-AUC and Brier score when a genuine probability output is available. For regression we report MAE. A 0.5 classification threshold is an initial baseline, not a tuned operating point.

The evaluation uses expanding chronological training windows, later test windows and a one-session gap to prevent next-day outcomes from crossing the train/test boundary. It produces fold-level results for **market-only** and **market + FinBERT**. Differences should be assessed across folds, coverage periods, stocks and regimes rather than reported as a single best score.

Planned deeper research includes calibration plots, SHAP explanations, richer embeddings, concept-drift checks, longer-horizon labels and stress tests. These are **research extensions**, not implemented or validated performance claims.

## Quick start

Python **3.11+** is recommended. From the repository root:

\`\`\`bash
python -m venv .venv
source .venv/bin/activate                  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,market,regimes]"
python -m pytest -q
\`\`\`

Try the full classical-model pipeline **without any external data**:

\`\`\`bash
high-beta demo --demo-days 340
high-beta evaluate --prices data/raw/demo_prices.csv
\`\`\`

Those files are **synthetic fixtures** created to validate wiring, not evidence about market behavior. They are deliberately excluded from Git.

Download publicly accessible *prototype* price history (Yahoo Finance via \`yfinance\`) and evaluate it:

\`\`\`bash
high-beta download --output data/raw/prices.csv
high-beta evaluate --prices data/raw/prices.csv
\`\`\`

For the financial-news comparison, install the optional model dependencies and supply **timestamped** news rows:

\`\`\`bash
python -m pip install -e ".[nlp]"
high-beta finbert --news data/raw/news.csv --output data/processed/scored_news.csv
high-beta evaluate --prices data/raw/prices.csv --news data/processed/scored_news.csv
\`\`\`

The first FinBERT call downloads the model; CPU inference is supported but can be slow. The repo does not include any paid-news access or API keys. For local inspection of saved results:

\`\`\`bash
python -m pip install -e ".[app]"
streamlit run app/streamlit_app.py
\`\`\`

See [architecture and data contracts](docs/architecture.md), [experiment design](docs/experiments.md), and [research limitations](docs/limitations.md) for the methodology and constraints.

## Repository layout

\`\`\`text
configs/                     Research universe, windows, target definition
src/high_beta_analysis/
  data/                      OHLCV acquisition, validation and synthetic fixtures
  features/                  Market features, timestamp-aware FinBERT features
  evaluation/                Walk-forward splits, labels, metrics and ablations
  models/                    Classical baselines and PyTorch MLP
  regimes/                   PCA and clustering
app/                         Streamlit viewer for saved experiments
tests/                       Price alignment, leakage, news timestamps, splits
docs/                        Design, methodology, limitations and references
data/                        Local source and processed data (not committed)
reports/                     Local evaluation artifacts (not committed)
\`\`\`

## Current scope and limitations

This repository provides an **executable research framework**, not published benchmark results. Market downloading uses a convenience feed, with potential revisions, adjusted OHLC semantics and historical ticker/IPO coverage limitations. Observed U.S. equity sessions are not yet backed by an exchange-calendar service; early closes, delistings and point-in-time index membership need additional checks for publication-quality research. Corporate filings and financial-news rights must be checked independently. A daily price row is not a proof of exchange-timestamp availability at exactly 4:00 p.m.

Do not interpret classification scores as forecasts of a specific return, proof of causality, or permission to trade. A proper deployment would require licensed point-in-time data, audit trails, monitoring, uncertainty estimates and independent validation.

## References

- [Dogu Araci, *FinBERT* (2019)](https://arxiv.org/abs/1908.10063)
- [ProsusAI FinBERT model and licensing details](https://huggingface.co/ProsusAI/finbert)
- [Scikit-learn model-selection and evaluation documentation](https://scikit-learn.org/stable/model_selection.html)
