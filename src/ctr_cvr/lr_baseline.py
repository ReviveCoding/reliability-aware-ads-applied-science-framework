import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from .feature_builder import feature_columns


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_lr_pipeline():
    cat_cols, num_cols = feature_columns()
    pre = ColumnTransformer([
        ("cat", _one_hot_encoder(), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    model = LogisticRegression(max_iter=500, n_jobs=None)
    return Pipeline([("pre", pre), ("model", model)])


def train_lr(train_df, target_col="click"):
    pipe = make_lr_pipeline()
    cat_cols, num_cols = feature_columns()
    cols = cat_cols + num_cols
    pipe.fit(train_df[cols], train_df[target_col])
    return pipe


def predict_proba(model, df):
    cat_cols, num_cols = feature_columns()
    cols = cat_cols + num_cols
    return model.predict_proba(df[cols])[:, 1]
