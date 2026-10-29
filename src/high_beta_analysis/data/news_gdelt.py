"""Optional GDELT discovery for a recent, non-exhaustive financial headline pilot.

GDELT's seen date is an observed/ingestion timestamp, not a publisher-authored
publication date. Treat it conservatively as an earliest known availability
time, do not backdate articles, and record this distinction in provenance.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERIES = {
    "IREN": '"IREN Limited"',
    "NBIS": '"Nebius"',
    "CRWV": '"CoreWeave"',
    "WULF": '"TeraWulf"',
    "APLD": '"Applied Digital"',
    "CIFR": '"Cipher Mining"',
    "CORZ": '"Core Scientific"',
    "HUT": '"Hut 8"',
}


def parse_gdelt_articles(ticker: str, payload: dict) -> pd.DataFrame:
    rows = []
    for item in payload.get("articles", []):
        headline = item.get("title")
        seen = item.get("seendate")
        if not headline or not seen:
            continue
        timestamp = pd.to_datetime(seen, errors="coerce", utc=True)
        if pd.isna(timestamp):
            continue
        rows.append({
            "ticker": ticker,
            # First-seen is a conservative AS-OF availability proxy.
            "published_at": timestamp,
            "headline": headline,
            "source": item.get("domain", "GDELT"),
            "source_url": item.get("url", ""),
            "timestamp_semantics": "GDELT first-seen, not publisher publication",
        })
    return pd.DataFrame(rows, columns=[
        "ticker", "published_at", "headline", "source",
        "source_url", "timestamp_semantics",
    ]).drop_duplicates(["ticker", "published_at", "headline"])


def fetch_recent_gdelt(
    tickers: list[str], days: int = 45, maxrecords: int = 75
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch up to maxrecords newest articles per ticker; not full historical coverage."""
    if not 1 <= days <= 89:
        raise ValueError("GDELT DOC search is limited to recent months (use 1..89 days)")
    if not 1 <= maxrecords <= 250:
        raise ValueError("maxrecords must be 1..250")
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("pip install requests (or pip install -e '.[market]')") from exc
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    frames, errors = [], {}
    for ticker in dict.fromkeys(tickers):
        if ticker not in QUERIES:
            errors[ticker] = "Missing an unambiguous named-company search query"
            continue
        params = {
            "query": QUERIES[ticker],
            "mode": "artlist",
            "format": "json",
            "maxrecords": maxrecords,
            "sort": "datedesc",
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": now.strftime("%Y%m%d%H%M%S"),
        }
        try:
            response = requests.get(
                ENDPOINT, params=params,
                headers={"User-Agent": "high-beta-stock-research/0.1"},
                timeout=35,
            )
            response.raise_for_status()
            rows = parse_gdelt_articles(ticker, response.json())
            frames.append(rows)
        except Exception as exc:
            errors[ticker] = str(exc)[:400]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values(["ticker", "published_at"]).reset_index(drop=True)
    else:
        combined = parse_gdelt_articles("", {})
    return combined, errors
