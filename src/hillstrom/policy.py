"""Targeting policies (who gets which email) and their value, estimated from the experiment."""

import numpy as np

from hillstrom.paths import SEED


def assign_policy(uplift_mens: np.ndarray, uplift_womens: np.ndarray, cost: float) -> np.ndarray:
    """For each customer, the action with the highest estimated net gain.

    Net gain is the estimated effect minus the cost per email for "mens" and "womens",
    and 0 for "control" (no email). A customer is emailed only if the net gain is
    strictly positive; ties between the two emails go to "mens".
    """
    net_mens = np.asarray(uplift_mens, dtype=float) - cost
    net_womens = np.asarray(uplift_womens, dtype=float) - cost
    send_mens = (net_mens > 0) & (net_mens >= net_womens)
    send_womens = (net_womens > 0) & ~send_mens
    return np.select([send_mens, send_womens], ["mens", "womens"], "control")


def policy_value_ipw(
    policy: np.ndarray, arm: np.ndarray, y: np.ndarray, cost: float = 0.0, p: float = 1 / 3
) -> float:
    """Inverse-propensity-weighted estimate of mean outcome per customer under `policy`, net of cost.

    In the experiment each customer had probability p of receiving any given action. The
    customers whose actual arm matches the policy's choice are a random sample of what
    would happen under the policy; dividing by p re-weights them to the full population.
    The cost is charged for every customer the policy emails.
    """
    policy, arm, y = np.asarray(policy), np.asarray(arm), np.asarray(y, dtype=float)
    gross = np.mean((arm == policy) * y / p)
    return float(gross - cost * np.mean(policy != "control"))


def bootstrap_policy_value(
    policy: np.ndarray,
    arm: np.ndarray,
    y: np.ndarray,
    cost: float = 0.0,
    p: float = 1 / 3,
    n_boot: int = 1000,
    seed: int = SEED,
) -> np.ndarray:
    """Bootstrap draws of `policy_value_ipw`, resampling customers.

    The same seed gives the same resamples, so draws for two policies evaluated on the
    same customers are paired and their difference can be summarised directly.
    """
    policy, arm, y = np.asarray(policy), np.asarray(arm), np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        draws[b] = policy_value_ipw(policy[idx], arm[idx], y[idx], cost=cost, p=p)
    return draws
