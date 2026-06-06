# Project Status

## Current version

This is a smoke-runnable, claim-safe core framework. It is designed to run without external data and to expose hooks for public-data validation.

## Implemented

- Synthetic ads integration data
- Data contract validation and leakage check
- Logistic Regression and Gradient-Boosting CTR baselines
- Optional lightweight PyTorch DNN path when torch is installed
- Empirical-Bayes calibration with hierarchical campaign/category calibration
- PHT-style lifecycle state features
- Mechanism-informed residual correction with champion/challenger selection
- Product-query relevance smoke benchmark
- ABC-style candidate expansion and bias-corrected reranking
- Synthetic uplift measurement with T-learner
- Synthetic logged-bandit OPE with IPS/SNIPS/DR
- Ranking policy comparison and release gate
- Drift diagnostics
- Reports, JSON/CSV outputs, and resume bullet generation

## Remaining recommended upgrades

1. Connect at least one real public dataset adapter before using stronger “public-data validated” resume wording.
2. Run optional DNN with torch installed and compare LR / GBDT / DNN under the same metrics.
3. Connect Amazon ESCI for real query-product relevance labels.
4. Connect Criteo Uplift local data for real incrementality validation.
5. Connect Open Bandit for real logged bandit OPE.
