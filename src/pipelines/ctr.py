from src.common.splits import temporal_split
from src.common.metrics import classification_report_dict
from src.data_contracts.validators import validate_ads_schema, leakage_check
from src.ctr_cvr.synthetic_ads_generator import generate_synthetic_ads
from src.ctr_cvr.feature_builder import build_features
from src.ctr_cvr.lr_baseline import train_lr, predict_proba as lr_predict
from src.ctr_cvr.gbdt_baseline import train_gbdt, predict_proba as gbdt_predict
from src.calibration.empirical_bayes import fit_beta_binomial_prior, group_posterior_table, apply_empirical_bayes
from src.specialty.mechanism_residual_correction import fit_residual_model, apply_residual_model
from src.specialty.slice_reliability_metrics import compute_slice_metrics, worst_slice_summary

def run_ctr_track(config):
    ads = generate_synthetic_ads(n_rows=int(config.get("n_rows", 900)), n_products=int(config.get("n_products", 90)), n_queries=int(config.get("n_queries", 36)), seed=int(config.get("random_seed", 42)))
    schema_result = validate_ads_schema(ads); ads = build_features(ads); train, val, test = temporal_split(ads, time_col="timestamp"); leak_result = leakage_check(train, test)
    lr = train_lr(train, "click"); test = test.copy(); test["pctr_lr"] = lr_predict(lr, test)
    gbdt = train_gbdt(train, "click"); test["pctr_gbdt"] = gbdt_predict(gbdt, test); test["pctr_raw"] = test["pctr_gbdt"]
    ctr_metrics = {"lr": classification_report_dict(test.click, test.pctr_lr), "gbdt": classification_report_dict(test.click, test.pctr_gbdt)}
    prior = fit_beta_binomial_prior(train, "click"); posterior, prior = group_posterior_table(train, group_col="campaign_id", target_col="click", prior=prior)
    test_cal = apply_empirical_bayes(test, raw_prob_col="pctr_raw", group_col="campaign_id", posterior_table=posterior, prior=prior)
    calibration_metrics = {"raw": classification_report_dict(test_cal.click, test_cal.pctr_raw), "eb": classification_report_dict(test_cal.click, test_cal.pctr_eb), "prior": prior}
    residual_features = [c for c in ["bid", "position", "relevance", "campaign_age", "lifecycle_state_index", "expected_pctr"] if c in train.columns]
    residual_model = fit_residual_model(train, residual_features, expected_col="expected_pctr", target_col="click")
    test_cal = apply_residual_model(residual_model, test_cal, residual_features); residual_metrics = classification_report_dict(test_cal.click, test_cal.pctr_residual_corrected)
    slice_df = compute_slice_metrics(test_cal, slice_col="category", target_col="click", prob_col="pctr_eb"); slice_summary = worst_slice_summary(slice_df)
    return {"schema": schema_result, "leakage": leak_result, "train": train, "test": test_cal, "ctr_metrics": ctr_metrics, "calibration_metrics": calibration_metrics, "residual_metrics": residual_metrics, "slice_df": slice_df, "slice_summary": slice_summary}
