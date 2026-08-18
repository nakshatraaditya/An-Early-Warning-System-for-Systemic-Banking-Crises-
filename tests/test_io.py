from __future__ import annotations

import pytest

from financial_crisis_ews.io import load_data, require_columns


class TestLoadData:
    def test_reads_excel(self, raw_xlsx, raw):
        assert len(load_data(raw_xlsx)) == len(raw)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path / "nope.xlsx"))


class TestRequireColumns:
    def test_passes_when_present(self, raw):
        require_columns(raw, ["country", "year", "crisisJST"])

    def test_raises_listing_every_missing_column(self, raw):
        with pytest.raises(KeyError) as exc:
            require_columns(raw, ["country", "nope", "also_missing"])
        assert "nope" in str(exc.value)
        assert "also_missing" in str(exc.value)

    def test_empty_requirement_is_a_noop(self, raw):
        require_columns(raw, [])
