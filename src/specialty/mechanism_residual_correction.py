import numpy as np
from sklearn.tree import DecisionTreeRegressor


def fit_residual_model(train_df, feature_cols, expected_col="expected_pctr", target_col="click"):
    residual = train_df[target_col].values - train_df[expected_col].values
    model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=15, random_state=42)
    model.fit(train_df[feature_cols], residual)
    return model


def apply_residual_model(model, df, feature_cols, expected_col="expected_pctr"):
    out = df.copy()
    residual_pred = model.predict(out[feature_cols])
    out["pctr_residual_corrected"] = np.clip(out[expected_col].values + residual_pred, 1e-6, 1 - 1e-6)
    out["residual_pred"] = residual_pred
    return out
