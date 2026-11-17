"""Project paths and constants, resolved relative to the repo root."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = (
    REPO_ROOT
    / "data"
    / "raw"
    / "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
SQL_DIR = REPO_ROOT / "sql"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"

# Default seed for every random operation (bootstrap, train/test split, models).
SEED = 20081
