import numpy as np
from sklearn.ensemble import RandomForestClassifier
from src.common.metrics import auuc_score
FEATURES = ["x1", "x2", "x3"]

def fit_t_learner(df):
    treated = df[df["treatment"] == 1]; control = df[df["treatment"] == 0]
    mt = RandomForestClassifier(n_estimators=35, max_depth=5, random_state=42).fit(treated[FEATURES], treated["conversion"])
    mc = RandomForestClassifier(n_estimators=35, max_depth=5, random_state=43).fit(control[FEATURES], control["conversion"])
    return mt, mc

def predict_uplift(models, df):
    mt, mc = models
    pt = mt.predict_proba(df[FEATURES])[:, 1]; pc = mc.predict_proba(df[FEATURES])[:, 1]
    out = df.copy(); out["pred_uplift"] = pt - pc; out["p_treated"] = pt; out["p_control"] = pc
    return out

def qini_proxy(df):
    ordered = df.sort_values("pred_uplift", ascending=False).reset_index(drop=True)
    treated = (ordered.treatment == 1).astype(float); control = (ordered.treatment == 0).astype(float)
    cum_t = treated.cumsum().replace(0, np.nan); cum_c = control.cumsum().replace(0, np.nan)
    gain = (ordered.conversion * treated).cumsum() - (ordered.conversion * control).cumsum() * (cum_t / cum_c)
    gain = gain.fillna(0).values; baseline = np.linspace(0, gain[-1] if len(gain) else 0, len(gain))
    return float(np.trapezoid(gain - baseline) / max(len(gain), 1))

def uplift_report_metrics(df):
    top = df.sort_values("pred_uplift", ascending=False).head(max(1, len(df)//5))
    return {"auuc": auuc_score(df.treatment, df.conversion, df.pred_uplift), "qini_proxy": qini_proxy(df), "uplift_at_20pct": float(top[top.treatment == 1].conversion.mean() - top[top.treatment == 0].conversion.mean()), "negative_uplift_rate": float((df.pred_uplift < 0).mean())}
