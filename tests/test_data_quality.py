from hillstrom.data import connect, run_sql_file
from tests.conftest import SYNTHETIC_CSV

VIOLATION_COLUMNS = [
    "rows_with_missing", "unmapped_arm", "duplicate_ids", "invalid_amounts",
    "spend_without_conversion", "conversion_without_spend", "conversion_without_visit",
    "no_purchase_history",
]


def test_clean_data_has_no_violations(synthetic_con):
    checks = run_sql_file(synthetic_con, "04_data_quality.sql").iloc[0]
    assert checks["n_rows"] == 6
    assert (checks[VIOLATION_COLUMNS] == 0).all()


def test_logic_violations_are_counted(tmp_path):
    # Add a row with spend but no conversion or visit, and an unknown arm label.
    bad_row = "3,1) $0 - $100,50.0,1,0,Urban,0,Web,Other E-Mail,0,0,25.0\n"
    path = tmp_path / "bad.csv"
    path.write_text(SYNTHETIC_CSV + bad_row)
    with connect(path) as con:
        checks = run_sql_file(con, "04_data_quality.sql").iloc[0]
    assert checks["spend_without_conversion"] == 1
    assert checks["unmapped_arm"] == 1
