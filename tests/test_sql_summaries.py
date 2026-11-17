import pytest

from hillstrom.data import run_sql_file


def test_arm_summary_counts_sum_to_total(synthetic_con):
    summary = run_sql_file(synthetic_con, "02_arm_summary.sql")
    assert summary["n"].sum() == 6
    assert summary["share"].sum() == pytest.approx(1.0)


def test_arm_summary_values(synthetic_con):
    summary = run_sql_file(synthetic_con, "02_arm_summary.sql").set_index("arm")
    assert summary.index.tolist() == ["control", "mens", "womens"]
    assert summary.loc["womens", "n"] == 3
    assert summary.loc["womens", "visit_rate"] == pytest.approx(1 / 3)
    assert summary.loc["mens", "conversion_rate"] == pytest.approx(0.5)
    assert summary.loc["mens", "spend_per_customer"] == pytest.approx(60.0)
    assert summary.loc["mens", "spend_per_converter"] == pytest.approx(120.0)
    assert summary.loc["control", "conversions"] == 0


def test_segment_summary_counts_per_dimension(synthetic_con):
    summary = run_sql_file(synthetic_con, "03_segment_summary.sql")
    assert set(summary["dimension"]) == {
        "history_segment", "channel", "zip_code", "newbie", "purchase_group",
    }
    assert (summary.groupby("dimension")["n"].sum() == 6).all()


def test_history_segment_order_preserved(synthetic_con):
    summary = run_sql_file(synthetic_con, "03_segment_summary.sql")
    levels = summary.loc[summary["dimension"] == "history_segment", "level"]
    assert levels.drop_duplicates().tolist() == [
        "1) $0 - $100", "2) $100 - $200", "3) $200 - $350", "5) $500 - $750", "7) $1,000 +",
    ]


def test_newbie_levels_are_labeled(synthetic_con):
    summary = run_sql_file(synthetic_con, "03_segment_summary.sql")
    levels = set(summary.loc[summary["dimension"] == "newbie", "level"])
    assert levels == {"new", "existing"}
