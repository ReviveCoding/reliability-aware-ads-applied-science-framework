from src.ope.synthetic_logged_bandit import generate_logged_bandit
from src.ope.estimators import ope_summary

def run_ope_track(config):
    df = generate_logged_bandit(n=max(600, int(config.get("n_rows", 600))), n_actions=5, seed=int(config.get("random_seed", 42)))
    return {"logged_feedback": df, "metrics": ope_summary(df)}
