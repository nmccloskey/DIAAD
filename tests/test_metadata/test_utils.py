from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import diaad.metadata.utils as metadata_utils


def test_small_metadata_helpers():
    assert metadata_utils.normalize_to_list(None) == []
    assert metadata_utils.normalize_to_list(("a", "b")) == ["a", "b"]
    assert metadata_utils.present_cols(pd.DataFrame(columns=["a", "c"]), ["c", "b", "a"]) == ["c", "a"]

    with pytest.raises(ValueError, match="missing required columns"):
        metadata_utils.validate_columns(pd.DataFrame(columns=["a"]), ["a", "b"])


def test_load_metadata_from_transcript_tables_combines_sources(monkeypatch, tmp_path):
    table_a = tmp_path / "a.xlsx"
    table_b = tmp_path / "b.xlsx"
    frames = {
        table_a: pd.DataFrame({"sample_id": ["S1"]}),
        table_b: pd.DataFrame({"sample_id": ["S2"]}),
    }

    monkeypatch.setattr(
        metadata_utils,
        "extract_transcript_data",
        lambda path, kind="joined": frames[Path(path)].copy(),
    )

    df = metadata_utils.load_metadata_from_transcript_tables(
        transcript_tables=[table_a, table_b],
        include_source_file=True,
    )

    assert list(df["sample_id"]) == ["S1", "S2"]
    assert list(df["file"]) == ["a.xlsx", "b.xlsx"]


def test_load_metadata_from_transcript_tables_requires_a_match():
    with pytest.raises(FileNotFoundError, match="No transcript tables found"):
        metadata_utils.load_metadata_from_transcript_tables(transcript_tables=[])


def test_load_metadata_from_transcript_tables_uses_configured_filename(
    monkeypatch,
    tmp_path,
):
    table = tmp_path / "site_metadata.xlsx"
    calls = {}

    def fake_find_transcript_table(**kwargs):
        calls.update(kwargs)
        return table

    monkeypatch.setattr(metadata_utils, "find_transcript_table", fake_find_transcript_table)
    monkeypatch.setattr(
        metadata_utils,
        "extract_transcript_data",
        lambda path, kind="joined": pd.DataFrame({"sample_id": ["S1"]}),
    )

    df = metadata_utils.load_metadata_from_transcript_tables(
        directories=tmp_path,
        transcript_table_filename="site_metadata.xlsx",
    )

    assert calls["filename"] == "site_metadata.xlsx"
    assert list(df["sample_id"]) == ["S1"]


def test_load_metadata_from_transcript_tables_errors_on_duplicate_configured_source(
    tmp_path,
):
    first = tmp_path / "site_a" / "site_metadata.xlsx"
    second = tmp_path / "site_b" / "site_metadata.xlsx"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"")
    second.write_bytes(b"")

    with pytest.raises(RuntimeError, match="multiple transcript table files"):
        metadata_utils.load_metadata_from_transcript_tables(
            directories=tmp_path,
            transcript_table_filename="site_metadata.xlsx",
            combine=False,
        )
