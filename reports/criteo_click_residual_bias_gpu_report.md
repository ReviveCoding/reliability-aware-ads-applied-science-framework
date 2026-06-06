# GPU Residual-Bias CTR Public-Data Validation Report

## Claim boundary

This run uses a local public Criteo Click Logs sample and local GPU training only. It does not use proprietary Amazon Ads data, advertiser data, customer data, production traffic, or production deployment.

## Method

This experiment adapts hybrid bias-learning logic to CTR prediction:

`corrected_logit = LR_baseline_logit + lambda * neural_residual_bias`

The neural model is trained as a residual logit correction over a strong LR pCTR baseline rather than as a standalone CTR model. A residual L2 penalty is used to discourage unnecessary overcorrection.

## Environment

- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 4090 Laptop GPU`
- PyTorch: `2.12.0+cu126`
- Torch CUDA build: `12.6`

## Data summary

- Rows: 250000
- Train / validation / test rows: 148750 / 26250 / 75000
- Target column: `label`
- Positive rate: 0.031232
- Numeric features: 13
- Categorical features: 26
- Hash size per categorical field: 120000

## Same-split comparison

| Metric | LR baseline | Residual-bias correction | DeepFM v2 champion |
|---|---:|---:|---:|
| ROC-AUC | 0.720525878183342 | 0.7208991511041081 | 0.721644151387245 |
| PR-AUC | 0.08757376855823538 | 0.08682682060879761 | 0.08800798140614195 |
| Log loss | 0.12939947371895114 | 0.1294107404132424 | 0.12916709604960983 |
| Brier | 0.029543600357203794 | 0.029602320294017343 | 0.029534284391458242 |
| ECE | 0.0035082482130316213 | 0.0024265145561505642 | 0.002120649363918434 |

## Selected correction

- Lambda: 0.7250000000000001
- Temperature: 1.0385050773620605
- Used temperature: True
- Mean absolute validation delta: 0.168398
- Mean absolute test delta: 0.166934

## Release decision

- LR gate status: **ITERATE**
- LR gate checks: {'residual_auc_not_worse_than_lr': True, 'residual_pr_auc_not_worse_than_lr': False, 'residual_log_loss_not_worse_than_lr': True, 'residual_ece_improves_lr': True, 'gpu_residual_component_trained': True}
- Current champion upgrade status: **ITERATE**
- Champion checks: {'matches_or_beats_deepfm_v2_on_at_least_3_of_5_metrics': False, 'log_loss_not_worse_than_deepfm_v2': False, 'ece_not_worse_than_deepfm_v2': False}
- Overall status: **ITERATE**
- Reason: Residual-bias correction trained successfully but did not pass the LR baseline gate.

## Interpretation

This runner tests whether a GPU neural residual can correct systematic LR pCTR bias without damaging strong baseline memorization. Promotion is not based on neural complexity alone; the correction must preserve ranking quality and improve calibration/log-loss under validation-selected lambda control.
