import numpy as np
import pandas as pd
import pytest

from hillstrom.uplift import TLearner, make_features, split_train_test


def _customers(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "recency": rng.integers(1, 13, n),
            "history": rng.uniform(30, 1000, n),
            "history_segment": "1) $0 - $100",
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "purchase_group": "both",
            "zip_code": rng.choice(["Rural", "Suburban", "Urban"], n),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["Multichannel", "Phone", "Web"], n),
            "arm": rng.choice(["mens", "womens", "control"], n),
            "visit": rng.integers(0, 2, n),
            "spend": 0.0,
        }
    )


def test_split_is_stratified_by_arm_and_disjoint():
    df = _customers()
    train, test = split_train_test(df, test_size=0.3, seed=1)
    assert len(test) == pytest.approx(0.3 * len(df), abs=2)
    assert set(train["customer_id"]).isdisjoint(test["customer_id"])
    shares = df["arm"].value_counts(normalize=True)
    for part in (train, test):
        assert np.allclose(part["arm"].value_counts(normalize=True)[shares.index], shares, atol=0.01)


def test_split_is_reproducible():
    df = _customers()
    a, _ = split_train_test(df, seed=5)
    b, _ = split_train_test(df, seed=5)
    assert a["customer_id"].tolist() == b["customer_id"].tolist()


def test_make_features_uses_only_pretreatment_columns():
    X = make_features(_customers())
    assert list(X.columns) == [
        "recency", "history", "mens", "womens", "newbie",
        "zip_code=Rural", "zip_code=Suburban", "zip_code=Urban",
        "channel=Multichannel", "channel=Phone", "channel=Web",
    ]
    assert X.dtypes.eq(float).all()


def test_make_features_columns_fixed_even_if_a_level_is_missing():
    df = _customers()
    subset = df[(df["zip_code"] == "Urban") & (df["channel"] == "Web")]
    assert list(make_features(subset).columns) == list(make_features(df).columns)


def _heterogeneous(n: int = 8000, seed: int = 0):
    """Effect only for x > 0: +0.2 on a binary outcome, +3 on a continuous one."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    treated = rng.integers(0, 2, n).astype(bool)
    effect_on = (x > 0) & treated
    y_bin = rng.binomial(1, 0.2 + 0.2 * effect_on)
    y_cont = 5 + 3 * effect_on + rng.normal(0, 1, n)
    X = pd.DataFrame({"x": x, "noise": rng.normal(size=n)})
    return X, treated, y_bin, y_cont, x


def test_tlearner_binary_recovers_where_the_effect_is():
    X, treated, y_bin, _, x = _heterogeneous()
    uplift = TLearner("binary", seed=0).fit(X, y_bin, treated).predict_uplift(X)
    assert uplift[x > 0].mean() == pytest.approx(0.2, abs=0.05)
    assert uplift[x <= 0].mean() == pytest.approx(0.0, abs=0.05)


def test_tlearner_continuous_recovers_where_the_effect_is():
    X, treated, _, y_cont, x = _heterogeneous()
    uplift = TLearner("continuous", seed=0).fit(X, y_cont, treated).predict_uplift(X)
    assert uplift[x > 0].mean() == pytest.approx(3.0, abs=0.3)
    assert uplift[x <= 0].mean() == pytest.approx(0.0, abs=0.3)


def test_tlearner_rejects_unknown_outcome_type():
    with pytest.raises(ValueError):
        TLearner("count")
