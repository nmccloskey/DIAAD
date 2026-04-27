from __future__ import annotations

import pandas as pd
import pytest

from diaad.blinding import decode as decode_module
from diaad.blinding import encode as encode_module
from diaad.core.config import AdvancedConfig


def test_encode_blinding_generates_codebook_and_skips_missing_cols(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    source_path = input_dir / "analysis.xlsx"
    pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "score": [10, 20],
        }
    ).to_excel(source_path, index=False)

    blinded_path, codebook_path, diagnostics_path = encode_module.encode_blinding(
        input_dir,
        output_dir,
        blinding_config=AdvancedConfig(
            blind_cols=["sample_id", "site", "test"],
        ),
        seed=1,
    )

    blinded_df = pd.read_excel(blinded_path)
    codebook_df = pd.read_excel(codebook_path)
    diagnostics_df = pd.read_excel(diagnostics_path)

    assert "sample_id" not in blinded_df.columns
    assert "sample_id_blinded" in blinded_df.columns
    assert list(blinded_df["score"]) == [10, 20]
    assert set(codebook_df["column"]) == {"sample_id"}
    assert list(diagnostics_df.columns) == ["sample_id", "sample_id_blinded"]


def test_decode_blinding_requires_codebook(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    pd.DataFrame({"sample_id_blinded": [1]}).to_excel(
        input_dir / "analysis_blinded.xlsx",
        index=False,
    )
    with pytest.raises(FileNotFoundError, match="blind codebook"):
        decode_module.decode_blinding(
            input_dir,
            output_dir,
            blinding_config=AdvancedConfig(),
        )


def test_encode_blinding_uses_codebook_columns_when_codebook_exists(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    target_path = input_dir / "analysis.xlsx"
    codebook_path = input_dir / "blind_codebook.xlsx"

    pd.DataFrame(
        {
            "participant": ["P1", "P2"],
            "score": [10, 20],
        }
    ).to_excel(target_path, index=False)
    pd.DataFrame(
        {
            "column": ["participant", "participant"],
            "raw_value": ["P1", "P2"],
            "blind_code": [101, 102],
        }
    ).to_excel(codebook_path, index=False)

    blinded_path, _, _ = encode_module.encode_blinding(
        input_dir,
        output_dir,
        blinding_config=AdvancedConfig(blind_cols=["sample_id"]),
    )

    blinded_df = pd.read_excel(blinded_path)

    assert "participant" not in blinded_df.columns
    assert list(blinded_df["participant_blinded"]) == [101, 102]


def test_decode_blinding_restores_suffixed_columns(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    target_path = input_dir / "analysis_blinded.xlsx"
    codebook_path = input_dir / "blind_codebook.xlsx"

    pd.DataFrame(
        {
            "sample_id_blinded": [1, 2],
            "score": [10, 20],
        }
    ).to_excel(target_path, index=False)
    pd.DataFrame(
        {
            "column": ["sample_id", "sample_id"],
            "raw_value": ["S1", "S2"],
            "blind_code": [1, 2],
        }
    ).to_excel(codebook_path, index=False)

    decoded_path = decode_module.decode_blinding(
        input_dir,
        output_dir,
        blinding_config=AdvancedConfig(),
    )

    decoded_df = pd.read_excel(decoded_path)

    assert list(decoded_df["sample_id"]) == ["S1", "S2"]
    assert "sample_id_blinded" not in decoded_df.columns
    assert list(decoded_df["score"]) == [10, 20]
