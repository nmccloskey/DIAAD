from __future__ import annotations

import pandas as pd
import pytest

from diaad.core.config import AdvancedConfig
from diaad.metadata import blinding


def test_choose_join_keys_requires_exact_configured_columns():
    df = pd.DataFrame(columns=["expanded_sample_id", "expanded_utterance_id"])
    metadata_df = pd.DataFrame(
        columns=["expanded_sample_id", "expanded_utterance_id", "speaker"]
    )

    assert blinding._choose_join_keys(
        df,
        metadata_df,
        sample_id_field="expanded_sample_id",
        utterance_id_field="expanded_utterance_id",
    ) == ["expanded_sample_id", "expanded_utterance_id"]


def test_choose_join_keys_does_not_fallback_when_configured_key_is_missing():
    df = pd.DataFrame(columns=["sample_id"])
    metadata_df = pd.DataFrame(columns=["sample_id", "speaker"])

    with pytest.raises(ValueError, match="missing_from_input_df=.*utterance_id"):
        blinding._choose_join_keys(df, metadata_df)


def test_generate_integer_blind_codebook_is_deterministic():
    df = pd.DataFrame({"sample_id": ["S2", "S1", "S1"], "speaker": ["A", "B", None]})
    codebook = blinding.generate_integer_blind_codebook(df, ["sample_id", "speaker"], seed=1)

    assert set(codebook["column"]) == {"sample_id", "speaker"}
    assert list(codebook[codebook["column"] == "sample_id"]["blind_code"]) == [1, 2]


def test_validate_blind_codebook_compatibility_rejects_duplicates():
    df = pd.DataFrame({"sample_id": ["S1"]})
    codebook = pd.DataFrame(
        {
            "column": ["sample_id", "sample_id"],
            "raw_value": ["S1", "S1"],
            "blind_code": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="duplicate mappings"):
        blinding.validate_blind_codebook_compatibility(df, codebook, ["sample_id"])


def test_blind_analysis_dataframe_recovers_metadata_columns():
    config = AdvancedConfig(
        sample_id_column="expanded_sample_id",
        utterance_id_column="expanded_utterance_id",
        id_columns=["expanded_sample_id"],
        blind_columns=["speaker"],
    )
    df = pd.DataFrame({"expanded_sample_id": ["S1", "S2"], "score": [1, 2]})
    metadata_df = pd.DataFrame(
        {"expanded_sample_id": ["S1", "S2"], "speaker": ["A", "B"]}
    )

    blinded_df, diagnostics_df, codebook_df = blinding.blind_analysis_dataframe(
        df,
        config,
        metadata_df=metadata_df,
        seed=1,
    )

    assert "speaker" not in blinded_df.columns
    assert "speaker_blinded" in blinded_df.columns
    assert list(diagnostics_df.columns) == [
        "expanded_sample_id",
        "speaker",
        "speaker_blinded",
    ]
    assert not codebook_df.empty


def test_blind_analysis_dataframe_uses_metadata_source_filename(monkeypatch, tmp_path):
    calls = {}

    def fake_load_metadata_from_transcript_tables(**kwargs):
        calls.update(kwargs)
        return pd.DataFrame({"sample_id": ["S1"], "speaker": ["A"]})

    monkeypatch.setattr(
        blinding,
        "load_metadata_from_transcript_tables",
        fake_load_metadata_from_transcript_tables,
    )
    config = AdvancedConfig(
        blind_columns=["speaker"],
        metadata_source="site_metadata.xlsx",
        id_columns=["sample_id"],
    )
    df = pd.DataFrame({"sample_id": ["S1"], "score": [1]})

    blinding.blind_analysis_dataframe(
        df,
        config,
        directories=tmp_path,
        discover_existing_codebook=False,
    )

    assert calls["transcript_table_filename"] == "site_metadata.xlsx"


def test_blind_file_identifiers_replaces_original_values():
    config = AdvancedConfig(blind_columns=["sample_id"])
    df = pd.DataFrame({"sample_id": ["S1", "S2"], "value": [10, 20]})

    blinded_df, codebook_df = blinding.blind_file_identifiers(df, config, seed=1)

    assert set(blinded_df["sample_id"]) == {1, 2}
    assert set(codebook_df["raw_value"]) == {"S1", "S2"}
