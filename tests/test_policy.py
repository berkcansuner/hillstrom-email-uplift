import numpy as np
import pytest

from hillstrom.policy import assign_policy, bootstrap_policy_value, policy_value_ipw


def test_assign_policy_picks_larger_net_uplift():
    policy = assign_policy(np.array([0.5, 0.2, 0.1]), np.array([0.3, 0.4, 0.05]), cost=0.0)
    assert policy.tolist() == ["mens", "womens", "mens"]


def test_assign_policy_high_cost_sends_nothing():
    policy = assign_policy(np.array([0.5, 0.2]), np.array([0.3, 0.4]), cost=1.0)
    assert policy.tolist() == ["control", "control"]


def test_assign_policy_sends_only_when_net_gain_is_positive():
    policy = assign_policy(np.array([0.3, -0.1, 0.31]), np.array([0.1, -0.2, 0.0]), cost=0.3)
    assert policy.tolist() == ["control", "control", "mens"]


def _randomized(n: int = 60_000, seed: int = 0):
    """Three equal arms; true mean outcome 1.0 (control), 1.5 (mens), 1.2 (womens)."""
    rng = np.random.default_rng(seed)
    arm = rng.choice(["control", "mens", "womens"], n)
    mean = np.select([arm == "mens", arm == "womens"], [1.5, 1.2], 1.0)
    y = rng.exponential(mean)
    return arm, y


def test_ipw_value_of_fixed_policies_is_close_to_truth():
    arm, y = _randomized()
    n = len(arm)
    for action, truth in [("control", 1.0), ("mens", 1.5), ("womens", 1.2)]:
        value = policy_value_ipw(np.full(n, action), arm, y)
        assert value == pytest.approx(truth, abs=0.05)


def test_ipw_value_of_mixed_policy_is_close_to_truth():
    arm, y = _randomized()
    rng = np.random.default_rng(1)
    policy = rng.choice(["control", "mens"], len(arm))  # half control, half mens
    assert policy_value_ipw(policy, arm, y) == pytest.approx(0.5 * 1.0 + 0.5 * 1.5, abs=0.05)


def test_ipw_subtracts_cost_for_emailed_customers_only():
    arm, y = _randomized()
    policy = np.where(np.arange(len(arm)) % 4 == 0, "control", "mens")  # 75% emailed
    gross = policy_value_ipw(policy, arm, y)
    assert policy_value_ipw(policy, arm, y, cost=0.2) == pytest.approx(gross - 0.2 * 0.75)


def test_bootstrap_policy_value_reproducible_and_centered():
    arm, y = _randomized(n=6000)
    policy = np.full(len(arm), "mens")
    a = bootstrap_policy_value(policy, arm, y, n_boot=200, seed=3)
    b = bootstrap_policy_value(policy, arm, y, n_boot=200, seed=3)
    assert np.array_equal(a, b)
    low, high = np.percentile(a, [2.5, 97.5])
    assert low < policy_value_ipw(policy, arm, y) < high
