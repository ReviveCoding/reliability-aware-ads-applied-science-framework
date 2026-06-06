from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from .feature_builder import feature_columns


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def train_gbdt(train_df, target_col="click"):
    cat_cols, num_cols = feature_columns()
    cols = cat_cols + num_cols
    pre = ColumnTransformer([
        ("cat", _one_hot_encoder(), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(train_df[cols], train_df[target_col])
    return pipe


def predict_proba(model, df):
    cat_cols, num_cols = feature_columns()
    cols = cat_cols + num_cols
    return model.predict_proba(df[cols])[:, 1]
