# Public Open Bandit reproduction evidence

- Repository baseline: `81fefd042036056c3ad68541f7872f144df425ca`.
- Public upstream: `st-tech/zr-obp`, pinned commit `8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4`.
- Source URL: https://raw.githubusercontent.com/st-tech/zr-obp/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/obd/bts/all/all.csv
- Full CSV SHA-256: `0ad874e4dbf6902f0845dd478ad8dde5ef6903583d3ffaace78411bdad064106`.
- 10,000 feedback rows, 90 columns, 80 actions, positions 1/2/3; 42 clicks.
- Retrieved from the public upstream, not the user's offline computer. No synthetic substitution.
- Original local sample has no recorded hash, so byte identity cannot be established. Original runner estimates and intervals reproduce the checked-in metrics to floating-point precision.

## Exact baseline reproduction

Executed using Python 3.12.14, numpy 2.3.5, pandas 2.2.3, sklearn 1.8.0:

```python
import json, subprocess
from pathlib import Path
scope = {'__name__': 'baseline', '__file__': str(Path('scripts/run_open_bandit_local.py').resolve())}
exec(subprocess.check_output([
    'git', 'show', '81fefd042036056c3ad68541f7872f144df425ca:scripts/run_open_bandit_local.py'
], text=True), scope)
df, path = scope['load_open_bandit']('data/open_bandit_verified')
metrics = scope['summarize'](scope['add_candidate_policy_probability'](df))
print(json.dumps(metrics, indent=2))
```

Output is `historical_runner_reproduced.json`. This executes functions only, not the historical `main`, so historical artifacts are not overwritten.

## Corrected rerun

```bash
mkdir -p data/open_bandit_verified
curl -fL https://raw.githubusercontent.com/st-tech/zr-obp/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/obd/bts/all/all.csv -o data/open_bandit_verified/all.csv
sha256sum data/open_bandit_verified/all.csv
OPEN_BANDIT_DIR=data/open_bandit_verified/all.csv \
OPEN_BANDIT_SOURCE_URL=https://raw.githubusercontent.com/st-tech/zr-obp/8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4/obd/bts/all/all.csv \
python scripts/run_open_bandit_local.py
```

Final run environment and timestamp are recorded in `outputs/open_bandit_ope_metrics_corrected.json`. The corrected reward model also fixes categorical user features previously silently coerced to zero, uses one-hot action identity, and changes observed affinity for every counterfactual action. Differences in DR therefore reflect both the estimator repair and reward-feature repair; this is not an ablation isolating one change.

SNIPS point estimate is unchanged; its bootstrap now recomputes the ratio. IPS, logged value, ESS, and max weight are unchanged to numerical tolerance. The corrected candidate remains **ITERATE**, failing all three historical promotion checks. DR intervals condition on fitted nuisance scores and assume IID rows; temporal/user dependence and uncertainty from fitting are not fully modeled. This is a per-position marginal evaluation, not a slate-value or production claim.
