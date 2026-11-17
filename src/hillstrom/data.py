"""Run the project's DuckDB SQL files and load the cleaned customer table."""

from pathlib import Path

import duckdb
import pandas as pd

from hillstrom.paths import RAW_CSV, SQL_DIR


def run_sql_file(con: duckdb.DuckDBPyConnection, name: str, **params) -> pd.DataFrame:
    """Run one statement from `sql/<name>` with named parameters ($param) and return the result."""
    sql = (SQL_DIR / name).read_text()
    return con.execute(sql, params).df()


def connect(csv_path: Path = RAW_CSV) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with the cleaned `customers` table loaded."""
    con = duckdb.connect()
    run_sql_file(con, "01_load_clean.sql", csv_path=str(csv_path))
    return con


def load_customers(csv_path: Path = RAW_CSV) -> pd.DataFrame:
    """Return the cleaned `customers` table as a DataFrame, ordered by customer_id."""
    with connect(csv_path) as con:
        return con.execute("SELECT * FROM customers ORDER BY customer_id").df()
