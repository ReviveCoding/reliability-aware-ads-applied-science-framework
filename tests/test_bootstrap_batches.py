import math

import numpy as np
import pytest

from src.common.metrics import bootstrap_ci


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize("n,draws", [(1, 30), (31, 257), (5001, 257)])
@pytest.mark.parametrize("seed", [0, 42])
def test_bootstrap_preserves_dtype_reductions_and_seeded_draws(
    dtype, n, draws, seed, monkeypatch
):
    values = np.arange(n).astype(dtype) / 7 if dtype != np.int64 else np.arange(n)
    values = values.astype(dtype)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=n, replace=True).mean() for _ in range(draws)]
    quantile = np.quantile

    def verify_draws(actual, probability):
        np.testing.assert_array_equal(actual, means)
        assert np.asarray(actual).dtype == np.asarray(means).dtype
        return quantile(actual, probability)

    monkeypatch.setattr(np, "quantile", verify_draws)
    result = bootstrap_ci(values, draws, seed)
    assert result == (float(quantile(means, 0.025)), float(quantile(means, 0.975)))


def test_empty_bootstrap_keeps_nan_interval():
    assert all(math.isnan(value) for value in bootstrap_ci([]))
