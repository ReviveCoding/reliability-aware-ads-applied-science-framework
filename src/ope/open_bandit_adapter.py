import pandas as pd


def load_open_bandit(path, nrows=None):
    """Load a simple Open Bandit-style CSV into the OPE schema.

    Required normalized columns: action, reward, pscore. If the candidate policy
    probability is absent, a conservative fallback is created for smoke-style tests.
    """
    df = pd.read_csv(path, nrows=nrows)
    lower = {c.lower(): c for c in df.columns}
    action_col = lower.get("action") or lower.get("item_id")
    reward_col = lower.get("reward") or lower.get("click")
    pscore_col = lower.get("pscore") or lower.get("propensity_score") or lower.get("propensity")
    if action_col is None or reward_col is None or pscore_col is None:
        raise ValueError("Open Bandit adapter requires action, reward, and propensity/pscore columns.")
    out = pd.DataFrame({
        "action": df[action_col],
        "reward": df[reward_col].astype(float),
        "pscore": df[pscore_col].astype(float).clip(1e-6, 1.0),
    })
    if lower.get("candidate_prob"):
        out["candidate_prob"] = df[lower.get("candidate_prob")].astype(float).clip(0, 1)
    else:
        # Conservative default: small probability for observed action under candidate policy.
        out["candidate_prob"] = 0.5 * out["pscore"] + 0.5 / max(out["action"].nunique(), 1)
    return out
