from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from financial_crisis_ews.evaluation import (
    budget_threshold_topk,
    compute_binary_metrics,
    compute_prob_metrics,
    event_level_recall,
)


class TestBudgetThreshold:
    def test_twenty_percent_budget_alerts_about_a_fifth(self):
        probs = np.linspace(0, 1, 100)
        thr = budget_threshold_topk(probs, 0.20)
        assert (probs >= thr).mean() == pytest.approx(0.20, abs=0.02)

    def test_tighter_budget_raises_the_threshold(self):
        probs = np.linspace(0, 1, 100)
        assert budget_threshold_topk(probs, 0.10) > budget_threshold_topk(probs, 0.50)

    def test_empty_input_returns_one(self):
        assert budget_threshold_topk(np.array([]), 0.2) == 1.0

    def test_constant_probabilities(self):
        assert budget_threshold_topk(np.full(10, 0.3), 0.2) == pytest.approx(0.3)


class TestEventLevelRecall:
    def _frame(self, crisis_year: int = 2008):
        years = list(range(2000, 2011))
        return pd.DataFrame({
            "country": ["USA"] * len(years),
            "year": years,
            "crisisJST": [1 if y == crisis_year else 0 for y in years],
        })

    def test_alert_in_the_window_counts_as_captured(self):
        df = self._frame()
        alerts = (df["year"] == 2007).astype(int).to_numpy()
        assert event_level_recall(df, alerts, "crisisJST") == 1.0

    def test_alert_two_years_before_also_counts(self):
        df = self._frame()
        alerts = (df["year"] == 2006).astype(int).to_numpy()
        assert event_level_recall(df, alerts, "crisisJST") == 1.0

    def test_alert_on_the_crisis_year_is_too_late(self):
        df = self._frame()
        alerts = (df["year"] == 2008).astype(int).to_numpy()
        assert event_level_recall(df, alerts, "crisisJST") == 0.0

    def test_alert_long_before_does_not_count(self):
        df = self._frame()
        alerts = (df["year"] == 2001).astype(int).to_numpy()
        assert event_level_recall(df, alerts, "crisisJST") == 0.0

    def test_no_crises_returns_nan(self):
        df = self._frame(crisis_year=0)
        alerts = np.zeros(len(df), dtype=int)
        assert np.isnan(event_level_recall(df, alerts, "crisisJST"))

    def test_partial_capture_across_countries(self):
        years = list(range(2000, 2011))
        df = pd.DataFrame({
            "country": ["USA"] * len(years) + ["GBR"] * len(years),
            "year": years * 2,
            "crisisJST": [1 if y == 2008 else 0 for y in years] * 2,
        })
        alerts = ((df["country"] == "USA") & (df["year"] == 2007)).astype(int).to_numpy()
        assert event_level_recall(df, alerts, "crisisJST") == 0.5


class TestBinaryMetrics:
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        out = compute_binary_metrics(y_true, y_prob, thr=0.5)
        assert out["precision"] == 1.0
        assert out["recall"] == 1.0
        assert out["f1"] == 1.0

    def test_alert_rate_reflects_threshold(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        assert compute_binary_metrics(y_true, y_prob, thr=0.5)["alert_rate"] == 0.5

    def test_no_alerts_does_not_divide_by_zero(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.1, 0.2])
        out = compute_binary_metrics(y_true, y_prob, thr=0.9)
        assert out["precision"] == 0.0
        assert out["recall"] == 0.0


class TestProbMetrics:
    def test_returns_finite_values_for_mixed_labels(self):
        out = compute_prob_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
        assert np.isfinite(out["pr_auc"])
        assert np.isfinite(out["brier"])

    def test_single_class_yields_nan(self):
        out = compute_prob_metrics(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4]))
        assert np.isnan(out["pr_auc"])
        assert np.isnan(out["brier"])

    def test_better_ranking_scores_higher_pr_auc(self):
        y_true = np.array([0, 0, 1, 1])
        good = compute_prob_metrics(y_true, np.array([0.1, 0.2, 0.8, 0.9]))["pr_auc"]
        bad = compute_prob_metrics(y_true, np.array([0.9, 0.8, 0.2, 0.1]))["pr_auc"]
        assert good > bad
