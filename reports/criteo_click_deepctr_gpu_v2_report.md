# GPU DeepCTR v2 Public-Data Validation Report

## Claim boundary

This run uses a local public Criteo Click Logs sample and local GPU training only. It does not use proprietary Amazon Ads data, advertiser data, customer data, production traffic, or production deployment.

## Environment

- Device: `cuda`
- GPU: `NVIDIA GeForce RTX 4090 Laptop GPU`
- PyTorch: `2.12.0+cu126`
- Torch CUDA build: `12.6`

## What changed vs v1

- Added a DeepFM-style wide + embedding + interaction model.
- Increased default training budget to 20 epochs with early stopping.
- Added same-split logistic-regression baseline.
- Added validation-selected LR + DeepFM ensemble.
- Promotion now focuses on preserving LR ranking while improving calibration.

## Data summary

- Rows: 250000
- Train / validation / test rows: 148750 / 26250 / 75000
- Target column: `label`
- Positive rate: 0.031232
- Numeric features: 13
- Categorical features: 26
- Hash size per categorical field: 120000

## Same-split comparison

| Metric | LR baseline | Selected DeepFM | LR + DeepFM ensemble |
|---|---:|---:|---:|
| ROC-AUC | 0.7205277822172589 | 0.6196707824279484 | 0.721644151387245 |
| PR-AUC | 0.08757380384923485 | 0.0508846999990245 | 0.08800798140614195 |
| Log loss | 0.1293993536666876 | 0.174941548779456 | 0.12916709604960983 |
| Brier | 0.029543605163491485 | 0.04015433448871911 | 0.029534284391458242 |
| ECE | 0.0035088425819993175 | 0.03304477808289216 | 0.002120649363918434 |

## Release decision

- Status: **PASS**
- Checks: {'ensemble_auc_not_worse_than_lr': True, 'ensemble_pr_auc_not_worse_than_lr': True, 'ensemble_ece_improves_lr': True, 'gpu_deep_component_trained': True}
- Reason: GPU DeepFM residual ensemble improved calibration while preserving LR ranking metrics.

## Interpretation

The v2 runner tests whether a GPU-trained DeepFM component adds useful nonlinear/categorical-interaction signal beyond a strong linear CTR baseline. The final release gate promotes only if the ensemble preserves LR ranking quality and materially improves calibration.
