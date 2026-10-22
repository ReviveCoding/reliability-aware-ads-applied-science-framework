"""Independent mathematical regression tests for the OPE corrections."""
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from src.ope.estimators import (
    bootstrap_snips_ci, cross_fitted_dr, doubly_robust,
    importance_weights, ips, snips,
)
from src.ope.synthetic_logged_bandit import generate_logged_bandit


class OracleWithLeakageGuard(RegressorMixin, BaseEstimator):
    """Oracle reward model that fails if a prediction row was in its training set."""
    fits = []
    predictions = []

    def fit(self, X, y):
        self.training_ids_ = set(X[:, 0])
        type(self).fits.append(self.training_ids_)
        return self

    def predict(self, X):
        assert self.training_ids_.isdisjoint(X[:, 0]), "evaluation row leaked into fit"
        type(self).predictions.extend(X[:, 0])
        return X[:, 1]  # deterministic reward equals binary action


def test_dr_integrates_actions_and_excludes_each_evaluation_row_from_fit():
    n = 20
    actions = np.arange(n) % 2
    target = np.tile([0.1, 0.9], (n, 1))
    def features(rows, action):
        return np.column_stack([rows, np.full(len(rows), action)])
    OracleWithLeakageGuard.fits = []
    OracleWithLeakageGuard.predictions = []
    scores = cross_fitted_dr(actions.astype(float), actions, np.full(n, .5),
                             target, [0, 1], features, n_splits=5,
                             model=OracleWithLeakageGuard())
    # Perfect q makes every residual zero. The correct direct term is .9;
    # the old q_eval=q_logged implementation would produce alternating 0/1.
    np.testing.assert_allclose(scores, .9)
    assert len(OracleWithLeakageGuard.fits) == 5
    ids, counts = np.unique(OracleWithLeakageGuard.predictions, return_counts=True)
    np.testing.assert_array_equal(ids, np.arange(n))
    np.testing.assert_array_equal(counts, np.full(n, 2))


def test_snips_constant_reward_has_degenerate_ratio_bootstrap_interval():
    lo, hi = bootstrap_snips_ci(np.full(20, .4), np.arange(1, 21), n_boot=300)
    assert lo == pytest.approx(.4)
    assert hi == pytest.approx(.4)


def test_snips_bootstrap_matches_independent_paired_ratio_reference():
    rewards = np.array([0., 1., .2, .7])
    weights = np.array([.2, 5., 1., 2.])
    rng = np.random.default_rng(17)
    ratios = []
    for _ in range(250):
        rows = rng.integers(4, size=4)
        ratios.append(np.dot(weights[rows], rewards[rows]) / weights[rows].sum())
    expected = np.quantile(ratios, [.025, .975])
    np.testing.assert_allclose(bootstrap_snips_ci(rewards, weights, n_boot=250, seed=17), expected)


def test_zero_weight_snips_fails_closed():
    with pytest.raises(ValueError, match="zero total weight"):
        bootstrap_snips_ci([0, 1], [0, 0])
    frame = pd.DataFrame({'reward': [0., 1.], 'pscore': [.5, .5], 'candidate_prob': [0., 0.]})
    with pytest.raises(ValueError, match="positive total weight"):
        snips(frame)


def test_behavior_target_identity_for_ips_and_snips():
    frame = pd.DataFrame({'reward': [0., 1., .2], 'pscore': [.2, .8, .4],
                          'candidate_prob': [.2, .8, .4]})
    assert ips(frame)[0] == pytest.approx(.4)
    assert snips(frame)[0] == pytest.approx(.4)


@pytest.mark.parametrize('pscore,candidate', [(0, .5), (-.1, .5), (1.1, .5), (.5, -.1), (.5, 1.1), (np.nan, .5)])
def test_invalid_probabilities_are_not_silently_clipped(pscore, candidate):
    with pytest.raises(ValueError):
        importance_weights([1.], [pscore], [candidate])


def test_generator_exposes_normalized_full_target_consistent_with_logged_probability():
    frame = generate_logged_bandit(n=100, n_actions=7)
    probs = frame[[f'candidate_prob_action_{a}' for a in range(7)]].to_numpy()
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    np.testing.assert_allclose(probs[np.arange(100), frame.action], frame.candidate_prob)


def test_dr_rejects_missing_inconsistent_and_unnormalized_policies():
    frame = generate_logged_bandit(n=20)
    with pytest.raises(ValueError, match='full candidate'):
        doubly_robust(frame.drop(columns=[c for c in frame if c.startswith('candidate_prob_action_')]))
    inconsistent = frame.copy()
    inconsistent['candidate_prob'] = 0
    with pytest.raises(ValueError, match='disagrees'):
        doubly_robust(inconsistent)
    invalid = frame.copy()
    for c in frame:
        if c.startswith('candidate_prob'):
            invalid[c] *= .5
    with pytest.raises(ValueError, match='sum to one'):
        doubly_robust(invalid)


def test_public_runner_counterfactual_affinity_and_action_mapping(monkeypatch):
    from scripts import run_open_bandit_local as runner
    # Non-contiguous IDs and reversed column order catch index/ID confusion.
    frame = pd.DataFrame({
        'item_id_int': [3, 9], 'reward': [0., 1.], 'pscore': [.5, .5],
        'position': [1, 2], 'user_feature_0': ['a', 'b'],
        'user-item_affinity_9': [2., 4.], 'user-item_affinity_3': [1., 3.],
    })
    enriched = runner.add_candidate_policy_probability(frame)
    np.testing.assert_allclose(enriched.candidate_prob, [1 / (1 + np.e), np.e / (1 + np.e)])
    np.testing.assert_allclose(enriched.observed_affinity, [1, 4])
    def inspect_crossfit(rewards, actions, pscore, probs, action_values, features, **kwargs):
        np.testing.assert_array_equal(action_values, [9, 3])
        rows = np.array([0, 1])
        f9, f3 = features(rows, 9), features(rows, 3)
        # Final features: counterfactual affinity followed by one-hot action.
        np.testing.assert_allclose(f9[:, -3], [2, 4])
        np.testing.assert_allclose(f3[:, -3], [1, 3])
        np.testing.assert_allclose(f9[:, -2:], [[1, 0], [1, 0]])
        np.testing.assert_allclose(f3[:, -2:], [[0, 1], [0, 1]])
        return np.array([.1, .2])
    monkeypatch.setattr(runner, 'cross_fitted_dr', inspect_crossfit)
    np.testing.assert_allclose(runner.dr_values(enriched), [.1, .2])


def test_public_runner_rejects_missing_affinity_instead_of_inventing_target():
    from scripts import run_open_bandit_local as runner
    with pytest.raises(ValueError, match='affinity'):
        runner.candidate_policy(pd.DataFrame({'item_id_int': [3]}))
    with pytest.raises(ValueError, match='mapped'):
        runner.candidate_policy(pd.DataFrame({'item_id_int': [3], 'user-item_affinity_9': [1.]}))


def test_normalized_adapter_requires_explicit_policy_and_preserves_full_distribution(tmp_path):
    from src.ope.open_bandit_adapter import load_open_bandit
    frame = generate_logged_bandit(n=10)
    path = tmp_path / 'feedback.csv'
    frame.drop(columns=['candidate_prob']).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Explicit candidate_prob'):
        load_open_bandit(path)
    frame.to_csv(path, index=False)
    restored = load_open_bandit(path)
    for name in ['x1', 'x2', 'candidate_prob', 'candidate_prob_action_0']:
        np.testing.assert_allclose(restored[name], frame[name])


def test_summary_withdraws_legacy_ope_estimates_but_accepts_corrected_provenance():
    from scripts.summarize_public_validations import one_line_summary
    metrics = {'doubly_robust': {'value': .91, 'ci_low': .89, 'ci_high': .94},
               'diagnostics': {'effective_sample_size': 30},
               'release_decision': {'status': 'PASS'}}
    legacy = one_line_summary('open_bandit_ope', metrics)
    assert legacy['status'] == 'ITERATE'
    assert 'withdrawn' in legacy['primary_result']
    assert 'DR value=0.91' not in legacy['primary_result']
    corrected = one_line_summary('open_bandit_ope', {**metrics, 'estimator_version': 'crossfit-action-integrated-v1'})
    assert 'DR value=0.91' in corrected['primary_result']
