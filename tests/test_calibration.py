from src.ctr_cvr.synthetic_ads_generator import generate_synthetic_ads
from src.ctr_cvr.feature_builder import build_features
from src.common.splits import temporal_split
from src.calibration.empirical_bayes import fit_beta_binomial_prior, group_posterior_table, apply_empirical_bayes


def test_empirical_bayes_outputs_probability():
    df = build_features(generate_synthetic_ads(n_rows=300, seed=2))
    train, _, test = temporal_split(df)
    test = test.copy()
    test["pctr_raw"] = 0.05
    prior = fit_beta_binomial_prior(train)
    post, prior = group_posterior_table(train, prior=prior)
    out = apply_empirical_bayes(test, posterior_table=post, prior=prior)
    assert "pctr_eb" in out.columns
    assert ((out["pctr_eb"] > 0) & (out["pctr_eb"] < 1)).all()
