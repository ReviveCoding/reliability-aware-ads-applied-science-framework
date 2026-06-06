# Scientist Playbook

## AUC is not enough

A model can improve AUC while worsening calibration. In ads ranking, pCTR is used inside a decision score, so Brier score and ECE matter.

## When not to launch a high-AUC model

Do not launch when:

- sparse campaign ECE is high
- false high-rank irrelevant ads exceed the threshold
- OPE confidence intervals overlap or fall below baseline
- negative uplift segments are large
- long-tail exposure collapses

## Uplift is not pCVR

pCVR estimates conversion probability. Uplift estimates incremental effect caused by treatment. A high-pCVR audience may convert anyway, so it may not be the best advertising target.

## OPE interpretation

IPS is simple but high variance. SNIPS reduces variance through self-normalization. Doubly Robust combines a reward model with propensity weighting and is more stable when either component is reasonably specified.

## Slice failures

A release decision should consider worst-slice behavior, not only aggregate performance.
