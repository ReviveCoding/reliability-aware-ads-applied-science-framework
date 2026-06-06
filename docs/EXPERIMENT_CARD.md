# Experiment Card

## Core ablations

| ID | Vanilla baseline | Proposed upgrade | Primary metrics |
|---|---|---|---|
| E1 | Logistic Regression CTR | Gradient Boosting / DNN | AUC, PR-AUC, log loss |
| E2 | Raw pCTR | Empirical-Bayes calibrated pCTR | Brier, ECE, slice ECE |
| E3 | Raw campaign age | PHT-style lifecycle state | Cold-start log loss, sparse ECE |
| E4 | Direct pCTR | Expected pCTR + residual model | Log loss, Brier, residual bias |
| E5 | BM25 top-K | Dense/candidate-expanded reranking | NDCG@K, Recall@K, MRR |
| E6 | Top-K only | ABC-style expansion + reranking | Recall@K, Precision@K, NDCG |
| E7 | pCVR targeting | Uplift targeting | AUUC, Qini, uplift@K |
| E8 | Aggregate metric | IPS/SNIPS/DR + CI | Policy value, CI lower bound |
| E9 | bid x raw pCTR | Reliability-aware score | Conversions, iROAS, guardrails |
| E10 | Static model | Drift-triggered retrain | Post-drift log loss, ECE |
| E11 | Aggregate release | Slice-aware release | Risky-policy BLOCK/ITERATE reasons |
