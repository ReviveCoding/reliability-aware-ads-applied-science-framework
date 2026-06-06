from src.uplift.synthetic_uplift_generator import generate_synthetic_uplift
from src.uplift.t_learner import fit_t_learner, predict_uplift, uplift_report_metrics
from src.ope.synthetic_logged_bandit import generate_logged_bandit
from src.ope.estimators import ope_summary


def test_uplift_pipeline_smoke():
    df = generate_synthetic_uplift(n=300, seed=10)
    models = fit_t_learner(df)
    pred = predict_uplift(models, df)
    metrics = uplift_report_metrics(pred)
    assert "auuc" in metrics
    assert "negative_uplift_rate" in metrics


def test_ope_summary_smoke():
    df = generate_logged_bandit(n=300, seed=11)
    metrics = ope_summary(df)
    assert "ips" in metrics
    assert "snips" in metrics
    assert "doubly_robust" in metrics
