from src.uplift.synthetic_uplift_generator import generate_synthetic_uplift
from src.uplift.t_learner import fit_t_learner, predict_uplift, uplift_report_metrics
from sklearn.model_selection import train_test_split

def run_uplift_track(config):
    df = generate_synthetic_uplift(n=max(600, int(config.get("n_rows", 600))), seed=int(config.get("random_seed", 42)))
    train, test = train_test_split(df, test_size=0.35, random_state=int(config.get("random_seed", 42)), stratify=df["treatment"])
    pred = predict_uplift(fit_t_learner(train), test)
    return {"train": train, "test_pred": pred, "metrics": uplift_report_metrics(pred)}
