# GPU Residual-Bias CTR v2 Public-Data Validation Report

## Claim boundary

This run uses a local public Criteo Click Logs sample and local GPU training only. It does not use proprietary Amazon Ads data, advertiser data, customer data, production traffic, or production deployment.

## Method

This v2 runner makes the residual correction safer than v1:

- Uses out-of-fold LR logits for residual training to reduce in-sample baseline leakage.
- Clips residual logits with `delta_clip`.
- Uses stronger residual L2 penalties.
- Restricts lambda selection to a conservative range.
- Applies validation constraints for PR-AUC, ROC-AUC, and log loss before choosing a correction.

Formula:

`corrected_logit = LR_baseline_logit + lambda * clipped_neural_residual_bias`

## Same-split comparison

| Metric | LR baseline | Safe residual-bias v2 | DeepFM v2 champion |
|---|---:|---:|---:|
| ROC-AUC | 0.7205262484121591 | 0.7206917465700768 | 0.721644151387245 |
| PR-AUC | 0.08757506612476876 | 0.08772190203774048 | 0.08800798140614195 |
| Log loss | 0.12939946618339213 | 0.1293115136202094 | 0.12916709604960983 |
| Brier | 0.02954358991944134 | 0.02956589061264065 | 0.029534284391458242 |
| ECE | 0.0035091918979077382 | 0.002288143587416676 | 0.002120649363918434 |

## Selected correction

- Lambda: 0.35
- Temperature: 1.0333850383758545
- Used temperature: True
- Constraint violations: 0
- Mean absolute validation delta: 0.037334
- Mean absolute test delta: 0.037258

## Release decision

- LR gate status: **PASS**
- LR gate checks: {'residual_auc_not_worse_than_lr': True, 'residual_pr_auc_not_worse_than_lr': True, 'residual_log_loss_not_worse_than_lr': True, 'residual_ece_improves_lr': True, 'gpu_residual_component_trained': True}
- Current champion upgrade status: **ITERATE**
- Champion checks: {'matches_or_beats_deepfm_v2_on_at_least_3_of_5_metrics': False, 'log_loss_not_worse_than_deepfm_v2': False, 'ece_not_worse_than_deepfm_v2': False}
- Overall status: **PASS_LR_BASELINE**
- Reason: Safe residual-bias correction passed the LR baseline gate but did not beat the current DeepFM v2 champion.
