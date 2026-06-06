import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_synthetic_uplift(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.binomial(1, 0.4, size=n)
    treatment = rng.binomial(1, 0.5, size=n)
    base = sigmoid(-2.4 + 0.6*x1 - 0.4*x2 + 0.3*x3)
    uplift = 0.03 + 0.08*(x1 > 0) - 0.04*(x2 > 1.0)
    prob = np.clip(base + treatment * uplift, 0.001, 0.8)
    conversion = rng.binomial(1, prob)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "treatment": treatment, "conversion": conversion, "true_uplift": uplift})
