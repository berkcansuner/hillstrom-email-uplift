import numpy as np
import pandas as pd
import pytest

from hillstrom.segments import effects_by_segment, heterogeneity_test


def _synthetic(effect_a: float, effect_b: float, n: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Two segments, 50/50 treated vs control, binary outcome with base rate 0.2."""
    rng = np.random.default_rng(seed)
    segment = rng.choice(["A", "B"], n)
    arm = rng.choice(["mens", "control"], n)
    effect = np.where(segment == "A", effect_a, effect_b) * (arm == "mens")
    visit = rng.binomial(1, 0.2 + effect)
    spend = visit * rng.uniform(10, 100, n)
    return pd.DataFrame({"segment": segment, "arm": arm, "visit": visit, "spend": spend})


def test_effects_by_segment_one_row_per_level():
    df = _synthetic(0.2, 0.0)
    table = effects_by_segment(df, "mens", "control", "visit", by="segment")
    assert table["level"].tolist() == ["A", "B"]
    assert table.loc[table["level"] == "A", "estimate"].item() == pytest.approx(0.2, abs=0.05)
    assert table.loc[table["level"] == "B", "estimate"].item() == pytest.approx(0.0, abs=0.05)
    assert (table["ci_low"] < table["estimate"]).all() and (table["estimate"] < table["ci_high"]).all()
    assert (table["n_t"] + table["n_c"]).sum() == len(df)


def test_effects_by_segment_spend_uses_bootstrap():
    df = _synthetic(0.2, 0.0)
    table = effects_by_segment(df, "mens", "control", "spend", by="segment", n_boot=500)
    assert table["ci_method"].unique().tolist() == ["bootstrap"]
    assert np.isnan(table["p_value"]).all()


def test_heterogeneity_detected_when_effects_differ():
    assert heterogeneity_test(_synthetic(0.2, 0.0), "mens", "control", "visit", by="segment") < 0.001


def test_heterogeneity_not_detected_when_effects_equal():
    assert heterogeneity_test(_synthetic(0.1, 0.1), "mens", "control", "visit", by="segment") > 0.05
