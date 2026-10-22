import pandas as pd


def load_open_bandit(path, nrows=None):
    """Load a simple Open Bandit-style CSV into the OPE schema.

    Required normalized columns: action, reward, pscore, candidate_prob.
    Full candidate_prob_action_<id> columns and x1/x2 are needed for generic DR.
    This is a normalized CSV adapter, not a raw OBD policy reconstruction.
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
        "pscore": df[pscore_col].astype(float),
    })
    if lower.get("candidate_prob"):
        out["candidate_prob"] = df[lower.get("candidate_prob")].astype(float)
    else:
        raise ValueError("Explicit candidate_prob required; no invented target policy")
    for c in df.columns:
        if c.startswith("candidate_prob_action_") or c in {"x1", "x2"}:
            out[c] = df[c].astype(float)
    from src.ope.estimators import importance_weights
    importance_weights(out.reward, out.pscore, out.candidate_prob)
    return out
