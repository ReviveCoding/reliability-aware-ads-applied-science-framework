"""Synthetic known-conditional-value check, not public-data performance evidence."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ope.synthetic_logged_bandit import generate_logged_bandit
from src.ope.estimators import ope_summary


def run(n=20000, seed=123):
    df = generate_logged_bandit(n=n, seed=seed)
    actions = np.arange(5)
    reward_means = 1 / (1 + np.exp(-(-1.5 + .4 * df.x1.to_numpy()[:, None] + .2 * df.x2.to_numpy()[:, None] + .25 * actions - .08 * (actions - 2)**2)))
    probs = df[[f"candidate_prob_action_{a}" for a in actions]].to_numpy()
    truth = float(np.mean(np.sum(probs * reward_means, axis=1)))
    metrics = ope_summary(df)
    return {"data_kind": "synthetic only", "n": n, "seed": seed,
            "known_target_value_on_sampled_contexts": truth,
            "absolute_errors": {k: abs(metrics[k]["value"] - truth) for k in ("ips", "snips", "doubly_robust")},
            "metrics": metrics}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
