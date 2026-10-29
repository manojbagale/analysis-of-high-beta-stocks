# Current limitations

This repository provides an executable research framework, not demonstrated prediction performance or a trading recommendation.

The candidate stocks belong to different parts of the AI-infrastructure supply chain; none is presumed to have beta above one without measuring it. Newly listed equities offer shorter histories. Historical ticker survivorship, delistings and business-model transitions need scrutiny.

Convenience price downloads can be revised and do not prove what was known at a past prediction cutoff. The code assumes a 4 p.m. ET close on observed sessions; holiday half-days and feed latency require a proper exchange calendar. News publisher timestamps, redistribution rights, syndication and historical coverage must be audited.

Financial sentiment is a linguistic classification, not a direct prediction of returns. A missing headline is not equivalent to neutral sentiment. The framework uses headline-level FinBERT scores but does not yet run document-level embeddings, full filings or attention-based fusion experiments.

Models may overfit; good historical metrics do not establish causality or future reliability. Additional chronological validation, calibration, fairness and failure analysis, real-world data quality checks, uncertainty estimates and transaction-cost-aware testing are necessary before any operational deployment.
