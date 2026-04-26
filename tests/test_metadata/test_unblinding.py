from __future__ import annotations

import pandas as pd
import pytest

from diaad.core.config import AdvancedConfig
from diaad.metadata import unblinding


@pytest.fixture
def codebook_df():
    return pd.DataFrame(
        {
            "column": ["sample_id", "sample_id", "speaker", "speaker"],
            "raw_value": ["S1", "S2", "A", "B"],
            "blind_code": [1, 2, 10, 20],
        }
    )


def test_generate_blind_decode_dict(codebook_df):
    decode = unblinding.generate_blind_decode_dict(codebook_df, target_cols=["sample_id"])
    assert decode == {"sample_id": {1: "S1", 2: "S2"}}


def test_unblind_dataframe_handles_suffixed_and_in_place_columns(codebook_df):
    df = pd.DataFrame({"sample_id_blinded": [1, 2], "speaker": [10, 20]})

    out = unblinding.unblind_dataframe(
        df,
        codebook_df,
        target_cols=["sample_id", "speaker"],
    )

    assert list(out["sample_id"]) == ["S1", "S2"]
    assert list(out["speaker"]) == ["A", "B"]
    assert "sample_id_blinded" not in out.columns


def test_validate_decode_codebook_rejects_duplicate_blind_codes():
    codebook = pd.DataFrame(
        {
            "column": ["sample_id", "sample_id"],
            "raw_value": ["S1", "S2"],
            "blind_code": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="duplicate \\(column, blind_code\\)"):
        unblinding.validate_decode_codebook(codebook)


def test_maybe_unblind_dataframe_returns_original_without_codebook(monkeypatch):
    config = AdvancedConfig()
    df = pd.DataFrame({"sample_id": [1]})
    monkeypatch.setattr(unblinding, "_load_blind_codebook", lambda **kwargs: None)

    out, codebook = unblinding.maybe_unblind_dataframe(df, config)

    assert out.equals(df)
    assert codebook is None
