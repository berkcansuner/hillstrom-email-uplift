from hillstrom.paths import REPO_ROOT, SQL_DIR


def test_repo_root_contains_pyproject():
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_sql_dir_is_inside_repo():
    assert SQL_DIR.parent == REPO_ROOT
