from __future__ import annotations

import pandas as pd

from diaad.coding.compl_utts import analysis, files, rates, rel_evaluation
from diaad.coding.utils.transcript import drop_excluded_speaker_rows


def test_cu_file_helpers_accept_custom_sample_id(monkeypatch):
    monkeypatch.setattr(files.random, "shuffle", lambda seq: None)
    monkeypatch.setattr(files.random, "sample", lambda seq, k: list(seq)[:k])

    utt_df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S1", "S2"],
            "expanded_utterance_id": ["U1", "U2", "U3"],
            "speaker": ["PAR", "INV", "PAR"],
            "utterance": ["one", "two", "three"],
            "narrative": ["A", "A", "B"],
        }
    )

    base = files._prepare_cu_base_dataframe(
        uttdf=utt_df,
        metadata_fields=[],
        stimulus_field="narrative",
        sample_id_field="expanded_sample_id",
    )
    primary, reliability = files._build_cu_assignments(
        base,
        mode="two",
        coder_ids=[1, 2],
        frac=1.0,
        cu_paradigms=[],
        exclude_speakers=[],
        sample_id_field="expanded_sample_id",
    )

    assert "expanded_sample_id" in primary.columns
    assert set(primary["id"]) == {1, 2}
    assert reliability is not None
    assert "expanded_sample_id" in reliability.columns


def test_cu_analysis_and_rates_accept_custom_sample_id():
    cu_df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S1", "S2"],
            "sv": [1, 0, 1],
            "rel": [1, 1, 1],
        }
    )

    pair = {"coder_prefix": None, "paradigm": None, "sv_col": "sv", "rel_col": "rel"}
    summary_long, summary_wide, cu_col = analysis._summarize_pair(
        cu_df,
        pair,
        sample_id_field="expanded_sample_id",
    )

    assert cu_col == "cu"
    assert "expanded_sample_id" in summary_long.columns
    assert "expanded_sample_id" in summary_wide.columns

    speaking = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S2"],
            "speaking_time": [120, 60],
            "speaking_minutes": [2, 1],
        }
    )
    merged = rates.merge_speaking_time(
        summary_long,
        speaking,
        sample_id_field="expanded_sample_id",
    )
    final = rates.finalize_cu_rates_columns(
        merged,
        sample_id_field="expanded_sample_id",
    )

    assert final.columns[0] == "expanded_sample_id"


def test_cu_analysis_drops_excluded_speakers_before_summary():
    cu_df = pd.DataFrame(
        {
            "sample_id": ["S1", "S1", "S2"],
            "speaker": ["PAR", "INV", "PAR"],
            "sv": [1, 1, 0],
            "rel": [1, 1, 1],
        }
    )

    filtered = drop_excluded_speaker_rows(cu_df, ["INV"])
    pair = {"coder_prefix": None, "paradigm": None, "sv_col": "sv", "rel_col": "rel"}
    summary_long, _, _ = analysis._summarize_pair(filtered, pair)

    s1 = summary_long.loc[summary_long["sample_id"] == "S1"].iloc[0]
    assert set(filtered["speaker"]) == {"PAR"}
    assert s1["no_utt"] == 1
    assert s1["cu"] == 1


def test_cu_reliability_accepts_custom_ids():
    rel_df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S1", "S2", "S2"],
            "expanded_utterance_id": ["U1", "U2", "U3", "U4"],
            "c2_sv": [1, 0, 1, 0],
            "c2_rel": [1, 1, 1, 0],
            "c2_cu": [1, 0, 1, 0],
            "c3_sv": [1, 1, 1, 0],
            "c3_rel": [1, 1, 0, 0],
            "c3_cu": [1, 1, 0, 0],
            "agmt_sv": [1, 0, 1, 1],
            "agmt_rel": [1, 1, 0, 1],
            "agmt_cu": [1, 0, 0, 1],
        }
    )

    summary, stats = rel_evaluation.summarize_cu_reliability(
        rel_df,
        sample_id_field="expanded_sample_id",
    )

    assert "expanded_sample_id" in summary.columns
    assert stats["sample_totals"]["cu"]["paired_samples"] == 2
