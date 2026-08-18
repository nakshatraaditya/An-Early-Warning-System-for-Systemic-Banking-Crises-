from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import make_raw

CRISES = {"USA": [1893, 1907, 1930, 1984, 2008], "GBR": [1890, 1931, 1974, 2007]}


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "financial_crisis_ews.train", *args],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("data") / "jst.xlsx"
    make_raw(crisis_years=CRISES, seed=7).to_excel(path, index=False)
    return str(path)


class TestHelp:
    def test_help_renders(self):
        """The budget help string contains a literal % that argparse must not expand."""
        result = run_cli("--help")
        assert result.returncode == 0
        assert "Alert budget" in result.stdout
        assert "20%" in result.stdout


class TestTrainingRun:
    def test_writes_metrics_csv_and_reports_summary(self, dataset, tmp_path):
        reports = tmp_path / "reports"
        result = run_cli("--raw-file", dataset, "--reports-dir", str(reports))

        assert result.returncode == 0, result.stderr
        assert "DONE. Metrics:" in result.stdout

        csv_path = reports / "rolling_metrics.csv"
        assert csv_path.exists()

        out = pd.read_csv(csv_path)
        assert not out.empty
        assert {"cutoff_year", "pr_auc", "brier", "alert_rate"} <= set(out.columns)

    def test_creates_missing_reports_directory(self, dataset, tmp_path):
        nested = tmp_path / "a" / "b" / "reports"
        assert run_cli("--raw-file", dataset, "--reports-dir", str(nested)).returncode == 0
        assert (nested / "rolling_metrics.csv").exists()

    def test_missing_required_column_fails_clearly(self, tmp_path):
        bad = make_raw(crisis_years=CRISES).drop(columns=["ltrate"])
        path = tmp_path / "bad.xlsx"
        bad.to_excel(path, index=False)

        result = run_cli("--raw-file", str(path), "--reports-dir", str(tmp_path / "r"))
        assert result.returncode != 0
        assert "ltrate" in result.stderr

    def test_missing_file_fails(self, tmp_path):
        result = run_cli("--raw-file", str(tmp_path / "nope.xlsx"),
                         "--reports-dir", str(tmp_path / "r"))
        assert result.returncode != 0

    def test_horizon_flag_is_honoured(self, dataset, tmp_path):
        reports = tmp_path / "reports"
        assert run_cli("--raw-file", dataset, "--horizon", "1",
                       "--reports-dir", str(reports)).returncode == 0
        assert Path(reports / "rolling_metrics.csv").exists()
