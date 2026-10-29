from high_beta_analysis.data.news_gdelt import parse_gdelt_articles


def test_gdelt_first_seen_is_not_mislabeled_publisher_time():
    payload = {"articles": [{
        "title": "Company announces new data center contract",
        "seendate": "20261029T210000Z",
        "domain": "example.org",
        "url": "https://example.org/article",
    }]}
    rows = parse_gdelt_articles("NBIS", payload)
    assert len(rows) == 1
    assert rows.iloc[0].published_at.isoformat() == "2026-10-29T21:00:00+00:00"
    assert "not publisher publication" in rows.iloc[0].timestamp_semantics


def test_gdelt_does_not_invent_missing_timestamps():
    payload = {"articles": [{"title": "No timestamp", "domain": "example.org"}]}
    assert parse_gdelt_articles("NBIS", payload).empty
