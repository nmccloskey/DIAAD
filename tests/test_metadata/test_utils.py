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
