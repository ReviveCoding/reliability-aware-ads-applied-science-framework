import numpy as np
import pandas as pd


def generate_logged_bandit(n=2500, n_actions=5, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # behavior policy is soft random with context effect
    logits = np.vstack([0.2*a*x1 - 0.15*a*x2 for a in range(n_actions)]).T
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    action = np.array([rng.choice(n_actions, p=p) for p in probs])
    pscore = probs[np.arange(n), action]
    reward_prob = 1 / (1 + np.exp(-(-1.5 + 0.4*x1 + 0.2*x2 + 0.25*action - 0.08*(action-2)**2)))
    reward = rng.binomial(1, reward_prob)
    # candidate policy prefers action with highest simple estimated score
    cand_scores = np.vstack([0.25*a*x1 + 0.10*a*x2 - 0.05*(a-3)**2 for a in range(n_actions)]).T
    cand_action = cand_scores.argmax(axis=1)
    cand_prob_logged_action = (cand_action == action).astype(float) * 0.85 + 0.15 / n_actions
    df = pd.DataFrame({"x1": x1, "x2": x2, "action": action, "pscore": pscore, "reward": reward, "candidate_prob": cand_prob_logged_action})

    for a in range(n_actions):
        df[f"candidate_prob_action_{a}"] = (cand_action == a).astype(float) * 0.85 + 0.15 / n_actions
    return df
