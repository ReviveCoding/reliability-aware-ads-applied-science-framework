# OPE correction audit — 2026-10-22

## Scope and provenance

Correction initially prepared and validated locally against public repository main `81fefd042036056c3ad68541f7872f144df425ca`. A fresh clone and remote-ref check established this base; the user's offline computer and any unpublished local changes were not inspected. Initial validation was local-only. Subsequent publication is limited to a separate correction branch and draft PR under user authorization; main merge and deployment are not authorized.

Raw data: [pinned official Open Bandit sample](https://raw.githubusercontent.com/st-tech/zr-obp/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/obd/bts/all/all.csv), 10,000 rows, SHA256 `0ad874e4dbf6902f0845dd478ad8dde5ef6903583d3ffaace78411bdad064106`. Re-executing the original runner on this download reproduced the historical metrics to floating-point precision; this is not a synthetic replacement or a different-size sample. The original local file hash is unavailable: matching historical metrics does not establish byte identity with that original file. Original committed output files remain unchanged.

## Before / after (pinned public sample reproducing historical metrics)

| Quantity | Historical | Corrected |
|---|---:|---:|
| Logged reward | 0.0042 | 0.0042 |
| IPS | 0.002459743 | 0.002459743 |
| SNIPS | 0.002435300 | 0.002435300 |
| SNIPS 95% exploratory interval | [0.001025365, 0.004158079] | [0.001038611, 0.004185765] |
| DR | 0.002495777 | 0.002755149 |
| DR 95% exploratory interval | [0.001125205, 0.004317053] | [0.000820700, 0.004794984] |
| ESS | 342.556935 | 342.556935 |
| Maximum importance weight | 277.777778 | 277.777778 |
| Decision | ITERATE | ITERATE |

All three checks still fail. ESS remains below 1,000, maximum weight exceeds 100, and the DR lower bound is below the logged value. Correcting the estimator is not evidence of a winning policy. Do not cite historical DR or SNIPS intervals as validated evidence. The corrected intervals are exploratory, not production qualification.

## Technical correction and caller trace

- `src/ope/estimators.py`: five-fold cross-fitted reward predictions; direct term `sum_a pi(a|x) q(x,a)`; logged residual evaluated only out-of-fold. Explicit complete normalized target distribution and action IDs required. SNIPS resamples paired reward/weight rows and recomputes the denominator. Invalid propensity values and zero-denominator bootstrap replicates fail closed. `snips` now returns paired numerator/weight rows, not misleading pseudo-observations.
- `src/ope/synthetic_logged_bandit.py`: persists full candidate probabilities; previously only the logged-action probability was available.
- `src/ope/open_bandit_adapter.py`: carries full policy and context columns and rejects absent candidate probabilities instead of inventing a target. This is a normalized-schema adapter, not raw OBD policy reconstruction; no current pipeline calls it.
- `src/pipelines/ope.py` calls shared `ope_summary`; `src/pipelines/all.py` consumes this output. Both covered by smoke execution, without overwriting historical artifacts.
- `scripts/run_open_bandit_local.py`: uses shared corrected helpers, counterfactual item affinity plus one-hot item IDs, categorical user context and position. Previously categorical strings were silently coerced to zero. Affinity suffixes are mapped explicitly to action IDs; missing mappings fail. Corrected artifacts have new filenames.
- `scripts/summarize_public_validations.py`: prefers corrected provenance-tagged metrics; historical OPE results are visibly withdrawn if no corrected file exists, preventing accidental re-publication through summary regeneration.
- `README.md`: replaces the current OPE row with corrected evidence and explains superseded numbers. DeepFM, ESCI and other module figures are untouched; this work does not independently revalidate those modules.

## Interpretation limits

The public target policy is softmax over 80 explicitly mapped item-affinity columns. Evaluation is per-position marginal reward averaged over observed positions, **not full-slate value**. Propensity correctness and support remain assumptions of the dataset/policy interface. Five-fold random cross-fitting prevents training on a row's own outcome, but does not establish time-aware or repeated-user independence. Intervals use 300 IID row bootstrap replicates, seed 42; DR holds fitted out-of-fold scores fixed rather than refitting nuisance models. These intervals do not incorporate temporal/user dependence, model selection uncertainty, or all finite-sample nuisance uncertainty. The baseline comparison retains the existing heuristic of comparing a DR lower bound to the observed logged mean, not a paired lift-confidence interval.

## Verification and reproduction

An isolated Python 3.12 environment was installed from `requirements.txt`; exact package versions are recorded in `validation/ope_correction/requirements-resolved.txt` and public output provenance. No dependency pins in the project were changed.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/run_pipeline.py --mode smoke --track ope
.venv/bin/python scripts/validate_ope_known_value.py
mkdir -p data/open_bandit_verified
curl -fL 'https://raw.githubusercontent.com/st-tech/zr-obp/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/obd/bts/all/all.csv' -o data/open_bandit_verified/all.csv
OPEN_BANDIT_DIR=data/open_bandit_verified/all.csv \
OPEN_BANDIT_SOURCE_URL='https://raw.githubusercontent.com/st-tech/zr-obp/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/obd/bts/all/all.csv' \
.venv/bin/python scripts/run_open_bandit_local.py
```

Numerical regression tests cover action-sensitive exact oracle rewards (target value 0.9), no in-fold prediction using a spy model, behavior/target importance-weight identity, SNIPS constant-reward and independently computed paired-bootstrap intervals, malformed policy/propensity rejection, public action mapping/counterfactual features, adapter fail-closed behavior and historical-summary withdrawal. `validation/ope_correction/synthetic_known_value.json` records a separate 20,000-row synthetic experiment with analytically known conditional reward expectation; it is not public-data evidence. The complete integration smoke writes only under `validation/ope_correction/synthetic_smoke` and reports synthetic PASS, not real-policy promotion.

### Final check results

- Existing suite before added regressions: **17 passed**.
- Complete suite with new regression coverage: **34 passed**.
- Synthetic full integration smoke: **PASS**; OPE CLI smoke completed.
- Synthetic known-value simulation: target conditional expectation 0.276882504; absolute errors IPS 0.001263722, SNIPS 0.002535472, DR 0.003197978. This is a correctness sanity check, not evidence that DR must outperform IPS on every sample.
- Independent reviewer reran the public pipeline and matched all nine corrected estimator/CI values within 1e-14.
- `git diff --check` and Python compile check passed. No configured lint/typecheck tool exists in this repository.
