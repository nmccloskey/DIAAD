from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Import target module
try:
    from diaad.coding.corelex import corelex as clx
    from diaad.coding.corelex import utils as clx_utils
except Exception as e:
    pytest.skip(f"Could not import diaad.coding.corelex: {e}", allow_module_level=True)


def _make_placeholder_files(tmp_path, *, unblind=False, fallback=False):
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    if unblind:
        p = input_dir / "unblind_utterance_data.xlsx"
        p.write_bytes(b"stub")
        paths.append(p)

    if fallback:
        u = input_dir / "TU_P01_transcript_tables.xlsx"
        u.write_bytes(b"stub")
        paths.append(u)

    return input_dir, paths


@pytest.fixture
def dummy_tiers():
    class DummyTier:
        def __init__(self, name, partition=False):
            self.name = name
            self.partition = partition

    return {"file": DummyTier("file", partition=False)}


def test_run_corelex_unblind_mode(tmp_path, monkeypatch, dummy_tiers):
    input_dir, _ = _make_placeholder_files(tmp_path, unblind=True)
    output_dir = tmp_path / "out"

    unblind_df = pd.DataFrame({
        "sample_id": ["S1"],
        "narrative": ["Sandwich"],
        "utterance": ["put peanut butter on bread then put two together"],
        "speaking_time": [120],
        "c2_cu": [1],
    })

    monkeypatch.setattr(clx_utils, "read_excel_safely", lambda path: unblind_df.copy())

    # Stub norms + percentile calculations
    monkeypatch.setattr(clx, "preload_corelex_norms",
                        lambda present: {scene: {"accuracy": object(), "efficiency": object()}
                                         for scene in present})
    monkeypatch.setattr(clx, "get_percentiles",
                        lambda score, df, col, **k: {"control_percentile": 60.0, "pwa_percentile": 80.0})

    clx.run_corelex(dummy_tiers, Path(input_dir), Path(output_dir))

    corelex_dir = output_dir / "core_lex"
    files = list(corelex_dir.glob("core_lex_data_*.xlsx"))
    assert files, "Expected a core_lex_data_<timestamp>.xlsx file."

    df = pd.read_excel(files[0], sheet_name="summary")
    detail_df = pd.read_excel(files[0], sheet_name="details")
    row = df.iloc[0]
    assert row["sample_id"] == "S1"
    assert row["narrative"] == "Sandwich"
    assert row["num_tokens"] == 9
    assert row["num_base_forms_produced"] == 8
    assert row["num_core_token_matches"] == 9
    assert row["speaking_time"] == 120
    assert pytest.approx(row["core_tokens_per_min"], 1e-6) == 4.5
    assert row["accuracy_control_percentile"] == 60.0
    assert row["accuracy_pwa_percentile"] == 80.0
    assert row["efficiency_control_percentile"] == 60.0
    assert row["efficiency_pwa_percentile"] == 80.0
    assert list(detail_df.columns) == clx.DETAIL_COLUMNS
    assert len(detail_df) == 25
    put = detail_df[detail_df["base_form"] == "put"].iloc[0]
    assert put["num_tokens_matched"] == 2
    assert put["score"] == 1


def test_run_corelex_transcript_table_mode(tmp_path, monkeypatch, dummy_tiers):
    input_dir, _ = _make_placeholder_files(tmp_path, fallback=True)
    output_dir = tmp_path / "out2"

    utts = pd.DataFrame({
        "sample_id": ["S2", "S2"],
        "narrative": ["Sandwich", "Sandwich"],
        "speaker": ["INV", "PAR"],
        "utterance": ["bread butter", "put"],
        "speaking_time": [np.nan, 60]
    })

    monkeypatch.setattr(clx, "preload_corelex_norms",
                        lambda present: {scene: {"accuracy": object(), "efficiency": object()}
                                         for scene in present})
    monkeypatch.setattr(clx, "get_percentiles",
                        lambda score, df, col, **k: {"control_percentile": 55.0, "pwa_percentile": 65.0})

    monkeypatch.setattr(clx_utils, "extract_transcript_data", lambda path: utts.copy())

    clx.run_corelex(dummy_tiers, Path(input_dir), Path(output_dir), exclude_participants={"INV"})

    corelex_dir = output_dir / "core_lex"
    files = list(corelex_dir.glob("core_lex_data_*.xlsx"))
    assert files, "Expected a core_lex_data_<timestamp>.xlsx file."

    df = pd.read_excel(files[0], sheet_name="summary")
    detail_df = pd.read_excel(files[0], sheet_name="details")
    row = df.iloc[0]
    assert row["sample_id"] == "S2"
    assert row["narrative"] == "Sandwich"
    assert row["num_base_forms_produced"] == 1
    assert row["num_core_token_matches"] == 1
    assert row["speaking_time"] == 60
    assert pytest.approx(row["core_tokens_per_min"], 1e-6) == 1.0
    assert row["accuracy_control_percentile"] == 55.0
    assert row["accuracy_pwa_percentile"] == 65.0
    assert row["efficiency_control_percentile"] == 55.0
    assert row["efficiency_pwa_percentile"] == 65.0
    assert len(detail_df) == 25
