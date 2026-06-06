# Release Decision Framework

The framework produces a PASS / ITERATE / BLOCK decision.

## PASS

- calibrated model quality is acceptable
- OPE lower confidence bound beats baseline
- no critical weak slice
- relevance and long-tail guardrails pass

## ITERATE

- aggregate performance improves
- one or more fixable reliability weaknesses remain
- OPE confidence interval overlaps the baseline
- sparse-slice calibration requires additional adjustment

## BLOCK

- OPE support is weak
- sparse-slice calibration fails badly
- false high-rank irrelevant rate is too high
- negative uplift segment is too large
- long-tail exposure collapses
