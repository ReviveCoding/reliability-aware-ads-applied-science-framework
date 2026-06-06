# AutoInt Attention CTR Public-Data Validation Report

## Claim boundary

This run uses a local public Criteo Click Logs sample and local GPU training only. It does not use proprietary Amazon Ads data, advertiser data, customer data, production traffic, or production deployment.

## Environment

- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 4090 Laptop GPU`
- PyTorch: `2.12.0+cu126`
- Torch CUDA build: `12.6`

## Experiment purpose

This experiment tests whether an AutoInt-style self-attention feature-interaction block adds useful signal beyond a strong logistic-regression CTR baseline and the prior GPU DeepFM residual ensemble.

## Data summary

- Rows: 250000
- Train / validation / test rows: 148750 / 26250 / 75000
- Target column: `label`
- Positive rate: 0.031232
- Numeric features: 13
- Categorical features: 26
- Hash size per categorical field: 120000

## Same-split comparison

| Metric | LR baseline | Selected AutoInt | LR + AutoInt ensemble | LR + DeepFM v2 ensemble |
|---|---:|---:|---:|---:|
| ROC-AUC | 0.7205260074695956 | 0.616603104647185 | 0.7208420770997868 | 0.721644151387245 |
| PR-AUC | 0.08757344191868073 | 0.04902229909083366 | 0.08783979230323294 | 0.08800798140614195 |
| Log loss | 0.12939943933480536 | 0.17330559522170355 | 0.12925560621286797 | 0.12916709604960983 |
| Brier | 0.029543607895815695 | 0.03985188908714157 | 0.02953412410946339 | 0.029534284391458242 |
| ECE | 0.0035087737541662704 | 0.032595116603717264 | 0.002833650387175929 | 0.002120649363918434 |

## Release decision

- Status: **ITERATE**
- Checks: {'ensemble_auc_not_worse_than_lr': True, 'ensemble_pr_auc_not_worse_than_lr': True, 'ensemble_ece_improves_lr': True, 'autoint_component_trained_on_cuda': True, 'autoint_matches_or_beats_deepfm_on_2_metrics': False}
- Reason: AutoInt trained successfully, but the residual ensemble did not pass all ranking/calibration or DeepFM comparison checks.

## Interpretation

AutoInt maps numerical and categorical CTR fields into feature tokens and applies multi-head self-attention to model high-order feature interactions. The release gate promotes attention only if the validation-selected residual ensemble preserves LR ranking quality, improves calibration, and meaningfully compares with the prior DeepFM residual ensemble.
