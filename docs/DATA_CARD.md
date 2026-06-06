# Data Card

## Smoke-mode data

Smoke mode uses synthetic ads impressions, synthetic products, synthetic uplift data, and synthetic logged bandit feedback. This is intended to validate the end-to-end framework without external downloads.

## Public-data-compatible adapters

The repository includes extensible adapter stubs for:

- Criteo-style CTR click logs
- Criteo Uplift Prediction Dataset
- Open Bandit Dataset
- Amazon ESCI Shopping Queries
- Amazon Reviews 2023

## Integration rule

Public datasets validate individual applied-science capabilities. A synthetic integration layer connects them into an Amazon Ads-style workflow. Datasets from different sources are not represented as one real production system.
