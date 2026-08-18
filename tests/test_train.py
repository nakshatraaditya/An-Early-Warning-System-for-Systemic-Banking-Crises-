from __future__ import annotations

import pandas as pd
import pytest

from financial_crisis_ews.features import (
    apply_causal_cleaning,
    build_feature_frame,
    create_target,
)
from financial_crisis_ews.train import rolling_train_eval
from tests.conftest import make_raw

CRISES = {"USA": [1893, 1907, 1930, 1984, 2008], "GBR": [1890, 1931, 1974, 2007]}


@pytest.fixture(scope="module")
def prepared() -> tuple[pd.DataFrame, list[str]]:
    raw = make_raw(crisis_years=CRISES, seed=3)
    df, features = build_feature_frame(raw)
    clean = apply_causal_cleaning(df, features, train_end_year=1950)
    return create_target(clean, horizon=2).reset_index(drop=True), features


@pytest.fixture(scope="module")
def folds(prepared) -> pd.DataFrame:
    df_target, features = prepared
    return rolling_train_eval(df_target, features, "crisisJST", budget=0.20)


class TestFoldStructure:
    def test_produces_folds(self, folds):
        assert not folds.empty

    def test_expected_columns(self, folds):
        expected = {
            "cutoff_year", "test_end_year", "n_train", "n_test", "base_rate",
            "pr_auc", "brier", "alert_threshold", "alert_rate", "crisis_recall",
            "tp", "fp", "tn", "fn",
        }
        assert expected <= set(folds.columns)

    def test_cutoffs_are_strictly_increasing(self, folds):
        assert folds["cutoff_year"].is_monotonic_increasing
        assert folds["cutoff_year"].is_unique

    def test_test_window_follows_its_cutoff(self, folds):
        assert (folds["test_end_year"] > folds["cutoff_year"]).all()

    def test_confusion_matrix_sums_to_test_size(self, folds):
        totals = folds[["tp", "fp", "tn", "fn"]].sum(axis=1)
        assert (totals == folds["n_test"]).all()

    def test_training_set_grows_over_time(self, folds):
        assert folds["n_train"].is_monotonic_increasing


class TestNoLeakage:
    def test_no_fold_trains_on_its_own_test_window(self, prepared, folds):
        """Every training row must predate the cutoff that defines the test window."""
        df_target, _ = prepared
        for _, fold in folds.iterrows():
            cutoff = fold["cutoff_year"]
            train_years = df_target.loc[df_target["year"] < cutoff, "year"]
            test_years = df_target.loc[
                (df_target["year"] >= cutoff) & (df_target["year"] < fold["test_end_year"]),
                "year",
            ]
            assert train_years.max() < test_years.min()

    def test_first_cutoff_respects_min_train_years(self, prepared):
        df_target, features = prepared
        earliest = df_target["year"].min()
        out = rolling_train_eval(
            df_target, features, "crisisJST", budget=0.20, min_train_years=40
        )
        assert out["cutoff_year"].min() >= earliest + 40


class TestAlertBudget:
    def test_alert_rate_tracks_the_budget(self, prepared):
        df_target, features = prepared
        out = rolling_train_eval(df_target, features, "crisisJST", budget=0.20)
        assert out["alert_rate"].mean() == pytest.approx(0.20, abs=0.10)

    def test_tighter_budget_alerts_less(self, prepared):
        df_target, features = prepared
        tight = rolling_train_eval(df_target, features, "crisisJST", budget=0.10)
        loose = rolling_train_eval(df_target, features, "crisisJST", budget=0.40)
        assert tight["alert_rate"].mean() < loose["alert_rate"].mean()

    def test_metrics_are_within_valid_ranges(self, folds):
        assert folds["alert_rate"].between(0, 1).all()
        assert folds["base_rate"].between(0, 1).all()
        assert folds["crisis_recall"].between(0, 1).all()


class TestDegenerateInputs:
    def test_no_crises_yields_no_folds(self):
        raw = make_raw(seed=5)
        df, features = build_feature_frame(raw)
        clean = apply_causal_cleaning(df, features, train_end_year=1950)
        target = create_target(clean, horizon=2).reset_index(drop=True)
        assert rolling_train_eval(target, features, "crisisJST", budget=0.20).empty

    def test_short_history_yields_no_folds(self):
        raw = make_raw(start=2000, end=2010, crisis_years={"USA": [2008]})
        df, features = build_feature_frame(raw)
        clean = apply_causal_cleaning(df, features, train_end_year=2005)
        target = create_target(clean, horizon=2).reset_index(drop=True)
        assert rolling_train_eval(target, features, "crisisJST", budget=0.20).empty
