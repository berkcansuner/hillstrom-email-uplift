import duckdb
import pandas as pd
import pytest

from hillstrom.data import load_customers, run_sql_file
from hillstrom.paths import RAW_CSV

EXPECTED_COLUMNS = [
    "customer_id", "recency", "history_segment", "history", "mens", "womens",
    "purchase_group", "zip_code", "newbie", "channel", "segment", "arm",
    "visit", "conversion", "spend",
]


def test_load_customers_has_expected_columns(synthetic_csv):
    df = load_customers(synthetic_csv)
    assert list(df.columns) == EXPECTED_COLUMNS


def test_load_customers_maps_all_arms(synthetic_csv):
    df = load_customers(synthetic_csv)
    assert df["arm"].notna().all()
    assert set(df["arm"]) == {"mens", "womens", "control"}


def test_zip_code_typo_recoded(synthetic_csv):
    df = load_customers(synthetic_csv)
    assert "Surburban" not in set(df["zip_code"])
    assert set(df["zip_code"]) == {"Suburban", "Rural", "Urban"}


def test_purchase_group(synthetic_csv):
    df = load_customers(synthetic_csv)
    assert df["purchase_group"].tolist() == [
        "mens_only", "both", "womens_only", "mens_only", "mens_only", "womens_only",
    ]


def test_customer_id_follows_file_order(synthetic_csv):
    df = load_customers(synthetic_csv)
    assert df["customer_id"].tolist() == [1, 2, 3, 4, 5, 6]
    assert df["history"].tolist() == [142.44, 329.08, 180.65, 675.83, 45.34, 1050.00]


@pytest.mark.data
def test_real_file_row_count_and_order():
    df = load_customers()
    raw = pd.read_csv(RAW_CSV)
    assert len(df) == 64_000
    assert df["arm"].notna().all()
    assert (df["history"].to_numpy() == raw["history"].to_numpy()).all()


def test_run_sql_file_reads_utf8_regardless_of_locale(tmp_path, monkeypatch):
    # U+2212 (minus sign) is not in Windows code pages such as cp1252/cp1254, so
    # reading with the locale default would garble it.
    (tmp_path / "unicode.sql").write_bytes("-- a − b\nSELECT '−' AS s;".encode("utf-8"))
    monkeypatch.setattr("hillstrom.data.SQL_DIR", tmp_path)
    con = duckdb.connect()
    assert run_sql_file(con, "unicode.sql")["s"].iloc[0] == "−"
