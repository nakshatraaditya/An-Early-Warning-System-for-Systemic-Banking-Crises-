from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from financial_crisis_ews.features import (
    apply_causal_cleaning,
    build_feature_frame,
    create_target,
    detect_equity_column,
    engineer_macro_features,
)
from tests.conftest import make_raw


class TestDetectEquityColumn:
    def test_prefers_eq_tr(self, raw):
        assert detect_equity_column(raw) == "eq_tr"

    def test_falls_back_to_later_candidate(self, raw):
        df = raw.rename(columns={"eq_tr": "eq_dp"})
        assert detect_equity_column(df) == "eq_dp"

    def test_returns_none_when_absent(self, raw):
        assert detect_equity_column(raw.drop(columns=["eq_tr"])) is None


class TestMacroFeatures:
    def test_creates_every_declared_feature(self, raw):
        df, features = engineer_macro_features(raw)
        for feature in features:
            assert feature in df.columns

    def test_yield_curve_is_long_minus_short(self, raw):
        df, _ = engineer_macro_features(raw)
        expected = raw["ltrate"] - raw["stir"]
        assert np.allclose(df["yield_curve"], expected)

    def test_sovereign_spread_is_zero_for_the_us_benchmark(self, raw):
        df, _ = engineer_macro_features(raw)
        usa = df[df["country"] == "USA"]
        assert np.allclose(usa["sovereign_spread"], 0.0)

    def test_credit_growth_is_computed_within_country(self, raw):
        """Each country's first year has no predecessor, so growth is undefined."""
        df, _ = engineer_macro_features(raw)
        first_rows = df.sort_values("year").groupby("country").head(1)
        assert first_rows["credit_growth"].isna().all()


class TestBuildFeatureFrame:
    def test_keeps_only_keys_and_features(self, raw):
        df, features = build_feature_frame(raw)
        assert set(df.columns) == {"country", "year", "crisisJST", *features}

    def test_replaces_infinities_with_nan(self):
        df_raw = make_raw(countries=("USA",), start=1900, end=1910)
        df_raw.loc[df_raw.index[0], "cpi"] = 0.0
        df_raw.loc[df_raw.index[0], "hp"] = np.inf
        df, _ = build_feature_frame(df_raw)
        assert not np.isinf(df.select_dtypes("number")).any().any()

    def test_row_count_is_preserved(self, raw):
        df, _ = build_feature_frame(raw)
        assert len(df) == len(raw)


class TestCausalCleaning:
    def test_excludes_both_world_wars(self, raw):
        df, features = build_feature_frame(raw)
        clean = apply_causal_cleaning(df, features, train_end_year=1950)
        assert clean[clean["year"].between(1914, 1918)].empty
        assert clean[clean["year"].between(1939, 1945)].empty

    def test_leaves_no_missing_feature_values(self, raw):
        df, features = build_feature_frame(raw)
        clean = apply_causal_cleaning(df, features, train_end_year=1950)
        assert not clean[features].isna().any().any()

    def test_adds_a_missing_flag_per_feature(self, raw):
        df, features = build_feature_frame(raw)
        clean = apply_causal_cleaning(df, features, train_end_year=1950)
        for feature in features:
            assert f"{feature}_missing" in clean.columns

    def test_missing_flags_are_binary(self, raw):
        df, features = build_feature_frame(raw)
        clean = apply_causal_cleaning(df, features, train_end_year=1950)
        flags = clean[[f"{f}_missing" for f in features]]
        assert set(np.unique(flags.to_numpy())) <= {0, 1}

    def test_features_stay_numeric(self, raw):
        df, features = build_feature_frame(raw)
        clean = apply_causal_cleaning(df, features, train_end_year=1950)
        assert all(pd.api.types.is_numeric_dtype(clean[f]) for f in features)

    def test_imputation_uses_only_pre_train_end_data(self):
        """A post-train-end outlier must not move the value used to fill gaps."""
        base = make_raw(countries=("USA",), start=1960, end=2000, seed=1)
        df, features = build_feature_frame(base)
        df.loc[df["year"] == 1975, "yield_curve"] = np.nan

        spiked = df.copy()
        spiked.loc[spiked["year"] > 1990, "yield_curve"] = 999.0

        filled_plain = apply_causal_cleaning(df, features, train_end_year=1990)
        filled_spiked = apply_causal_cleaning(spiked, features, train_end_year=1990)

        value_plain = filled_plain.loc[filled_plain["year"] == 1975, "yield_curve"].iloc[0]
        value_spiked = filled_spiked.loc[filled_spiked["year"] == 1975, "yield_curve"].iloc[0]
        assert value_plain == pytest.approx(value_spiked)


class TestCreateTarget:
    def test_flags_the_two_years_before_a_crisis(self):
        raw = make_raw(countries=("USA",), start=2000, end=2010,
                       crisis_years={"USA": [2008]})
        df, _ = build_feature_frame(raw)
        out = create_target(df, horizon=2)
        flagged = set(out.loc[out["target"] == 1, "year"])
        assert flagged == {2006, 2007}

    def test_horizon_one_flags_only_the_prior_year(self):
        raw = make_raw(countries=("USA",), start=2000, end=2010,
                       crisis_years={"USA": [2008]})
        df, _ = build_feature_frame(raw)
        out = create_target(df, horizon=1)
        assert set(out.loc[out["target"] == 1, "year"]) == {2007}

    def test_target_does_not_leak_across_countries(self):
        raw = make_raw(countries=("USA", "GBR"), start=2000, end=2010,
                       crisis_years={"USA": [2008]})
        df, _ = build_feature_frame(raw)
        out = create_target(df, horizon=2)
        assert out[out["country"] == "GBR"]["target"].sum() == 0

    def test_crisis_year_itself_is_not_flagged(self):
        raw = make_raw(countries=("USA",), start=2000, end=2010,
                       crisis_years={"USA": [2008]})
        df, _ = build_feature_frame(raw)
        out = create_target(df, horizon=2)
        assert out.loc[out["year"] == 2008, "target"].iloc[0] == 0

    def test_no_crises_yields_all_zero(self, raw):
        df, _ = build_feature_frame(raw)
        assert create_target(df, horizon=2)["target"].sum() == 0
