# Price and text providers

**Prices:** the real-market study script uses yfinance, a research convenience interface to Yahoo Finance. It records gaps, coverage, earliest and latest bars and suspicious daily changes. Price fields are not guaranteed to reproduce the information a trader had at a historical 4 p.m. cutoff. A point-in-time licensed archive would strengthen the study.

**News:** the optional recent-news script queries the GDELT DOC 2.0 ArticleList API for selected company names. That API provides a recent rolling window (roughly three months), not the same multi-year coverage as market bars. The query is capped and newest-first. Headlines may be syndicated, irrelevant or duplicated; the script retains URLs and domains for audit. In particular, GDELT seendate means *first seen by the monitoring system*, not necessarily the original publisher's publication timestamp. We use first-seen conservatively as an availability proxy and annotate it in metadata.

FinBERT is an inference model; its sentiment probabilities describe a piece of financial language, not realized returns. Scored articles remain local or in short-lived Actions artifacts; they are not committed.

**Valid ablation rule:** Do not run market-only over three years and market+news over 45 days, compare the two scores and claim an improvement. A meaningful model ablation must use the **same dates**, **same tickers**, **same folds**, **same label definitions** and **documented news coverage**. Until a suitable licensed historical text archive is available, the GDELT / FinBERT pilot demonstrates text ingestion and scoring only.

**Provider access:** data vendors can throttle, block CI IPs or change interfaces. If acquisition fails, the run must fail and record the problem instead of manufacturing results.
