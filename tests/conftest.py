from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

RAW_COLUMNS = [
    "hp", "cpi", "real_credit", "debtgdp", "gdp",
    "ltrate", "stir", "money", "ca", "eq_tr",
]


def make_raw(
    countries: tuple[str, ...] = ("USA", "GBR"),
    start: int = 1870,
    end: int = 2020,
    crisis_years: dict[str, list[int]] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """A JST-shaped frame: one row per country-year with the columns train.py requires."""
    rng = np.random.default_rng(seed)
    crisis_years = crisis_years or {}
    rows = []
    for country in countries:
        crises = set(crisis_years.get(country, []))
        for year in range(start, end + 1):
            rows.append({
                "country": country,
                "year": year,
                "crisisJST": int(year in crises),
                "hp": 100 + rng.normal(0, 5),
                "cpi": 50 + 0.5 * (year - start) + rng.normal(0, 2),
                "real_credit": 100 + 0.8 * (year - start) + rng.normal(0, 10),
                "debtgdp": 0.5 + rng.normal(0, 0.05),
                "gdp": 1000 + 10 * (year - start) + rng.normal(0, 50),
                "ltrate": 4 + rng.normal(0, 1),
                "stir": 3 + rng.normal(0, 1),
                "money": 200 + rng.normal(0, 20),
                "ca": rng.normal(0, 2),
                "eq_tr": 100 + rng.normal(0, 8),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def raw_with_crises() -> pd.DataFrame:
    return make_raw(crisis_years={"USA": [1930, 2008], "GBR": [1931, 2008]})


@pytest.fixture
def raw_xlsx(tmp_path, raw) -> str:
    path = tmp_path / "jst.xlsx"
    raw.to_excel(path, index=False)
    return str(path)
