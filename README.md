# Public-Data Ads Applied Science Framework

A local-runnable, public-data-compatible Amazon Ads-style applied science framework for reliability-aware ranking, measurement, and release decisions.

This repository is designed as an industry-style offline applied science framework, not a toy CTR notebook. It connects:

1. data contracts and validation
2. CTR/CVR supervised modeling
3. empirical-Bayes calibration
4. product-query relevance and candidate reranking
5. uplift and incrementality measurement
6. off-policy policy evaluation
7. reliability-aware ranking policy simulation
8. drift-triggered champion/challenger updates
9. PASS / ITERATE / BLOCK release decisions

## Claim boundary

This project validates ads-science capabilities using public datasets or synthetic fallback data and connects modules through a synthetic integration layer. It does not use proprietary Amazon Ads data, real advertiser data, real customer data, production traffic, or production deployment. All results are local-runnable, public-data-compatible, and production-style portfolio evidence only.

## Why this project exists

The goal is to show that an applied scientist can go beyond a vanilla CTR model. The framework compares vanilla baselines against reliability-aware upgrades:

- raw pCTR versus empirical-Bayes calibrated pCTR
- raw campaign age versus PHT-style lifecycle state
- direct pCTR prediction versus mechanism-informed residual bias correction
- BM25 top-K retrieval versus ABC-style candidate expansion and bias-corrected reranking
- pCVR targeting versus uplift targeting
- aggregate model metric release versus slice-aware release gates

## Quick start

```bash
python scripts/run_pipeline.py --mode smoke --track all
python -m pytest tests -q
```

## Release qualification safeguards

The offline PASS decision requires finite, valid empirical-Bayes calibration ECE,
worst-slice ECE with sample support, and both relevance and long-tail metrics for
the selected reliability policy. Missing or invalid evidence results in BLOCK,
rather than being interpreted as passing. Synthetic smoke qualification does not
constitute production release approval.

## Main generated artifacts

After running smoke mode:

```text
reports/full_pipeline_report.md
reports/calibration_report.md
reports/ranking_policy_report.md
reports/release_decision.md
reports/relevance_report.md
reports/uplift_report.md
reports/ope_report.md
outputs/metrics_summary.json
outputs/release_decision.json
outputs/policy_comparison.csv
outputs/slice_metrics.csv
```

## Repository layout

```text
src/common/                       Shared metrics, schema, reporting utilities
src/data_contracts/                Data validation and leakage checks
src/ctr_cvr/                       Synthetic ads data, feature building, CTR models
src/calibration/                   Beta-binomial and empirical-Bayes calibration
src/relevance/                     Product-query relevance, candidate expansion, reranking
src/uplift/                        Synthetic uplift data, T-learner, AUUC/Qini
src/ope/                           Synthetic logged bandit data and OPE estimators
src/policy/                        Ranking policies, guardrails, release gate
src/specialty/                     PHT-style, residual correction, ABC-style reliability layers
src/model_ops/                     Drift monitoring and champion/challenger simulation
scripts/                           Runnable pipelines
tests/                             Unit tests for the framework spine
```

## Public-data adapters

The repository is structured to support public datasets when available:

- Criteo Click Logs or Criteo-style samples for CTR/CVR prediction and calibration
- Criteo Uplift for incrementality analysis
- Open Bandit Dataset for off-policy evaluation
- Amazon ESCI Shopping Queries for query-product relevance
- Amazon Reviews 2023 for product metadata and graph-style relevance proxies

Smoke mode does not require external data.

## v3 structural cleanup

The project now uses a cleaner, less redundant pipeline architecture:

```text
scripts/run_pipeline.py          # unified CLI for all tracks
src/pipelines/                   # modular track orchestration
src/adapters/                    # public-data adapter interface
src/common/config.py             # config parser for smoke/sample/full modes
src/common/artifacts.py          # centralized report/output writing
docs/                            # detailed documentation
sample_outputs/                  # committed example outputs from a verified smoke run
```

Use:

```bash
python scripts/run_pipeline.py --mode smoke --track all
python scripts/run_pipeline.py --mode smoke --track ctr
python scripts/run_pipeline.py --mode smoke --track relevance
python scripts/run_pipeline.py --mode smoke --track uplift
python scripts/run_pipeline.py --mode smoke --track ope
```

<!-- PUBLIC_VALIDATION_SUMMARY_START -->
# Public Validation Summary

## Claim boundary

This repository uses local public/proxy datasets and synthetic fallback only. It does not claim use of proprietary Amazon Ads data, advertiser data, customer data, production traffic, or production deployment.

## Current status

| Module | Dataset | Status | Evidence |
|---|---|---|---|
| criteo_uplift | Criteo Uplift v2.1 local public-data sample | ITERATE | T-learner AUUC=31.323246684676043, Qini=1.3236038275331832, uplift@20=0.003468105315009738; pCVR baseline AUUC=54.45230542291612, Qini=24.452662565773274, uplift@20=0.006609127662921918. |
| esci_relevance | Amazon ESCI Shopping Queries local public-data sample | ITERATE | TF-IDF NDCG@10=0.7539334608935288, Recall@10=0.5468361105179388; reranker NDCG@10=0.7541430157510265, Recall@10=0.5424159058290577. |
| open_bandit_ope | Open Bandit local public-data sample | ITERATE | Logged value=0.0042, DR value=0.0024957770082094173, DR CI=[0.0011252045603265017, 0.004317053219629562], ESS=342.5569352840535. |
| criteo_click_ctr | Criteo Click Logs local public-data sample | ITERATE | LR AUC=0.7086663943114613, ECE=0.0015713278297538127; GBDT AUC=0.7002069626101098, ECE=0.004141469619541966; GBDT+EB AUC=0.6627800466601141, ECE=0.0005694217648938824. |
| GPU DeepCTR / DeepFM v2 | Criteo Click Logs public sample | PASS | LR+DeepFM ensemble: ROC-AUC 0.7205 to 0.7216, PR-AUC 0.0876 to 0.0880, log loss 0.12940 to 0.12917, ECE 0.00351 to 0.00212 (39.6% reduction). |
| GPU AutoInt attention CTR | Criteo Click Logs public sample | ITERATE | LR+AutoInt ensemble: ROC-AUC 0.7205 to 0.7208, PR-AUC 0.0876 to 0.0878, ECE 0.00351 to 0.00283; held at ITERATE versus DeepFM champion. |
| GPU residual-bias CTR v2 | Criteo Click Logs public sample | PASS_LR_BASELINE | Paper-inspired LR-offset residual logit correction: ROC-AUC 0.7205 to 0.7207, PR-AUC 0.0876 to 0.0877, log loss 0.12940 to 0.12931, ECE 0.00351 to 0.00229 (34.8% reduction); passed LR gate but not DeepFM champion upgrade. |
| ABC-style ESCI reranker v2 | Amazon ESCI public sample | PASS | NDCG@10 0.6498 to 0.6761, Recall@10 0.5727 to 0.5742, Precision@10 0.8471 to 0.8502, high-rank irrelevant 0.1529 to 0.1498. |
| ABC-style ESCI ablation | Amazon ESCI public sample | EVIDENCE_STRENGTHENED | Best variant expansion_plus_expected_gain with alpha=0.2, beta=0.8, gamma=0.0, delta=0.0: NDCG@10 0.6761, Recall@10 0.5742, high-rank irrelevant 0.1498. |

## Key conclusions

- GPU DeepFM v2 is the current CTR champion: a validation-selected LR+DeepFM ensemble improved ranking and calibration over the same-split LR baseline.
- Residual-bias CTR v2 validates the paper-inspired bias-correction specialty against LR, but DeepFM v2 remains the CTR champion.
- ABC-style ESCI reranker v2 is the strongest specialty result: it passed NDCG, recall, precision, and irrelevant-rate guardrails.
- Ablation shows learned expected-gain scoring is the primary driver; expansion alone is unsafe because it improves NDCG but hurts recall and irrelevant-rate guardrails.

### ESCI ABC ablation interpretation

The ablation strengthens the ABC-style ESCI result by separating the components.

- `expected_gain_only` passed all guardrails and was the main driver: NDCG@10 0.675228, Recall@10 0.575819, Precision@10 0.850489, high-rank irrelevant 0.149511.
- `expansion_only` improved NDCG@10 to 0.653443, but failed recall and increased high-rank irrelevant rate, so expansion alone is not safe.
- The promoted `expansion_plus_expected_gain` variant preserved all guardrails and achieved the best guardrail score.
- Explicit overreach penalty was evaluated, but not selected as the promoted scoring term; it is better framed as a release guardrail rather than the main performance driver.

## Limitations and interpretation

- Do not claim production Amazon Ads deployment.
- Do not claim all public-data challengers beat all baselines.
- Do not claim AutoInt beat DeepFM.
- Do not claim explicit ESCI overreach penalty was the promoted performance driver.

<!-- PUBLIC_VALIDATION_SUMMARY_END -->
