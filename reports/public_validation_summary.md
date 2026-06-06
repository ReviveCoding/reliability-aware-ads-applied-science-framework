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

## Resume-safe claim language

- Built a local-runnable ads applied-science framework covering CTR prediction, calibration, product-query relevance, uplift, OPE, and release-gated model validation on public/proxy datasets.
- Added GPU-trained DeepFM and AutoInt CTR experiments; promoted a validation-selected LR+DeepFM ensemble as the CTR champion under ranking and calibration gates.
- Adapted paper-inspired residual-bias correction to CTR; passed the LR baseline gate by improving ROC-AUC, PR-AUC, log loss, and ECE, while preserving DeepFM v2 as champion.
- Adapted ABC-style expansion/scoring to Amazon ESCI product-query ranking; ablation showed learned expected-gain scoring was the primary driver and expansion alone was unsafe.

## Do-not-overclaim boundaries

- Do not claim production Amazon Ads deployment.
- Do not claim all public-data challengers beat all baselines.
- Do not claim AutoInt beat DeepFM.
- Do not claim explicit ESCI overreach penalty was the promoted performance driver.
