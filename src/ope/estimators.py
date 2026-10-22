"""Finite-action OPE. CIs are iid row bootstraps, not deployment guarantees."""
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from src.common.metrics import bootstrap_ci


def importance_weights(rewards, pscore, candidate_prob):
    rewards, pscore, candidate_prob = map(lambda x: np.asarray(x, dtype=float), (rewards, pscore, candidate_prob))
    if rewards.ndim != 1 or rewards.size == 0 or pscore.shape != rewards.shape or candidate_prob.shape != rewards.shape:
        raise ValueError("Expected nonempty, aligned one-dimensional OPE inputs")
    if not all(np.isfinite(x).all() for x in (rewards, pscore, candidate_prob)):
        raise ValueError("OPE inputs must be finite")
    if np.any((pscore <= 0) | (pscore > 1)) or np.any((candidate_prob < 0) | (candidate_prob > 1)):
        raise ValueError("Invalid propensity or candidate probability; no silent clipping")
    return candidate_prob / pscore


def bootstrap_snips_ci(rewards, weights, n_boot=300, seed=42, alpha=.05):
    """Paired row bootstrap of the ratio, recomputing each denominator.

    A zero-weight resample makes the bootstrap undefined: fail closed rather
    than discard it or invent a denominator and understate uncertainty.
    """
    rewards, weights = np.asarray(rewards, float), np.asarray(weights, float)
    if rewards.ndim != 1 or rewards.size == 0 or weights.shape != rewards.shape or not np.isfinite(rewards).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid SNIPS bootstrap inputs")
    if n_boot < 1 or not 0 < alpha < 1:
        raise ValueError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(len(rewards), size=len(rewards))
        denominator = weights[idx].sum()
        if denominator <= 0:
            raise ValueError("SNIPS bootstrap resample has zero total weight")
        values.append(np.sum(weights[idx] * rewards[idx]) / denominator)
    return tuple(float(v) for v in np.quantile(values, [alpha / 2, 1 - alpha / 2]))


def cross_fitted_dr(rewards, actions, pscore, policy_probs, action_values, feature_builder, *, n_splits=5, seed=42, model=None):
    """Cross-fit q(x,a), then integrate over the full fixed target policy.

    feature_builder(row_indices, action) must create counterfactual features for
    that action (including action-dependent affinity), never outcomes. Target
    policy must be specified independently of evaluation rewards. Returned score
    bootstrap conditions on fitted nuisances; it does not refit forests.
    """
    rewards = np.asarray(rewards, float)
    actions, action_values = np.asarray(actions), np.asarray(action_values)
    probs = np.asarray(policy_probs, float)
    n = len(rewards)
    if action_values.ndim != 1 or len(np.unique(action_values)) != len(action_values) or len(action_values) == 0:
        raise ValueError("Action universe must be nonempty and unique")
    if actions.shape != (n,) or probs.shape != (n, len(action_values)) or not np.isfinite(probs).all() or (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-10, rtol=0):
        raise ValueError("Full target distribution must be finite and sum to one")
    index = {a: j for j, a in enumerate(action_values)}
    if any(a not in index for a in actions):
        raise ValueError("Logged action outside explicit action universe")
    logged_cols = np.array([index[a] for a in actions])
    weights = importance_weights(rewards, pscore, probs[np.arange(n), logged_cols])
    if not 2 <= n_splits <= n:
        raise ValueError("Cross-fitting requires 2 <= n_splits <= number of rows")
    if model is None:
        model = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=seed)
    q_logged, q_eval = np.empty(n), np.zeros(n)
    for train, test in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(rewards):
        # Build training features using each row's actual action.
        X_train = np.stack([np.asarray(feature_builder(np.array([i]), actions[i]))[0] for i in train])
        fitted = clone(model).fit(X_train, rewards[train])
        for j, action in enumerate(action_values):
            q = np.clip(fitted.predict(feature_builder(test, action)), 0, 1)
            q_eval[test] += probs[test, j] * q
            match = actions[test] == action
            q_logged[test[match]] = q[match]
    return q_eval + weights * (rewards - q_logged)


def ips(df):
    weights = importance_weights(df.reward, df.pscore, df.candidate_prob)
    values = weights * np.asarray(df.reward)
    return float(values.mean()), values


def snips(df):
    """Return point estimate and paired [weighted reward, weight] rows (not scores)."""
    weights = importance_weights(df.reward, df.pscore, df.candidate_prob)
    if weights.sum() <= 0:
        raise ValueError("SNIPS requires positive total weight")
    numerator = weights * np.asarray(df.reward)
    return float(numerator.sum() / weights.sum()), np.column_stack([numerator, weights])


def doubly_robust(df):
    # Explicit integer action IDs encoded in full target-probability columns.
    columns = [c for c in df.columns if c.startswith("candidate_prob_action_")]
    if not columns:
        raise ValueError("DR requires full candidate_prob_action_<id> distribution")
    actions = np.array([int(c.removeprefix("candidate_prob_action_")) for c in columns])
    probs = df[columns].to_numpy(float)
    mapping = {a: i for i, a in enumerate(actions)}
    if any(a not in mapping for a in df.action):
        raise ValueError("Logged action outside target action universe")
    logged = probs[np.arange(len(df)), [mapping[a] for a in df.action]]
    if not np.allclose(logged, df.candidate_prob, atol=1e-10, rtol=0):
        raise ValueError("Logged candidate probability disagrees with full policy")
    context = df[["x1", "x2"]].to_numpy(float)
    def features(rows, action):
        # One-hot action identity avoids treating arbitrary IDs as ordinal.
        return np.column_stack([context[rows], np.tile(actions == action, (len(rows), 1))])
    values = cross_fitted_dr(df.reward, df.action, df.pscore, probs, actions, features)
    return float(values.mean()), values


def overlap_diagnostics(df):
    w = importance_weights(df.reward, df.pscore, df.candidate_prob)
    return {"effective_sample_size": float(w.sum()**2 / max((w**2).sum(), 1e-9)), "max_importance_weight": float(w.max()), "mean_importance_weight": float(w.mean()), "low_support_rate": float((df.candidate_prob < 0.05).mean())}


def ope_summary(df):
    out = {}
    for name, fn in [("ips", ips), ("snips", snips), ("doubly_robust", doubly_robust)]:
        value, vals = fn(df)
        if name == "snips":
            lo, hi = bootstrap_snips_ci(df.reward, vals[:, 1], n_boot=300)
        else:
            lo, hi = bootstrap_ci(vals, n_boot=300)
        out[name] = {"value": value, "ci_low": lo, "ci_high": hi, "ci_width": hi-lo}
    out["diagnostics"] = overlap_diagnostics(df)
    out["inference"] = {"method": "iid row percentile bootstrap; DR fixed cross-fitted scores", "n_boot": 300, "n_splits": 5, "seed": 42, "limitation": "Not cluster/time-aware; no nuisance refitting or production coverage guarantee"}
    return out
