from __future__ import annotations

import pandas as pd

from diaad.coding.word_counts import analysis, files, rates, rel_evaluation


def test_word_count_file_helpers_accept_custom_ids(monkeypatch):
    monkeypatch.setattr(files.random, "sample", lambda seq, k: list(seq)[:k])

    df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S1", "S2"],
            "expanded_utterance_id": ["U1", "U2", "U3"],
            "speaker": ["PAR", "INV", "PAR"],
            "utterance": ["hello world", "ignored", "another one"],
            "comment": ["", "", ""],
            "cu": [1, None, 1],
        }
    )

    prepared = files._prepare_wc_df(
        df,
        source_type="cu",
        exclude_speakers=["INV"],
        sample_id_field="expanded_sample_id",
        utterance_id_field="expanded_utterance_id",
    )
    primary, reliability = files._assign_wc_coders(
        prepared,
        num_coders=1,
        frac=1.0,
        sample_id_field="expanded_sample_id",
    )

    assert list(prepared.columns) == [
        "expanded_sample_id",
        "expanded_utterance_id",
        "speaker",
        "utterance",
        "comment",
        "id",
        "word_count",
        "wc_comment",
    ]
    assert list(prepared["word_count"]) == [2, "NA", 2]
    assert set(primary["id"]) == {1}
    assert set(reliability["expanded_sample_id"]) == {"S1", "S2"}


def test_word_count_analysis_and_rates_accept_custom_sample_id():
    wc_df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S1", "S2"],
            "word_count": [2, None, 5],
        }
    )

    summary = analysis._summarize_word_counts(
        wc_df,
        word_count_field="word_count",
        sample_id_field="expanded_sample_id",
    )

    assert "expanded_sample_id" in summary.columns
    assert dict(zip(summary["expanded_sample_id"], summary["total_words"])) == {
        "S1": 2.0,
        "S2": 5.0,
    }

    speaking = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S2"],
            "speaking_time": [120, 60],
            "speaking_minutes": [2, 1],
        }
    )
    merged = rates.merge_speaking_time(
        summary,
        speaking,
        sample_id_field="expanded_sample_id",
    )
    final = rates.finalize_word_count_rates_columns(
        merged,
        sample_id_field="expanded_sample_id",
    )

    assert final.columns[0] == "expanded_sample_id"


def test_word_count_analysis_drops_excluded_speakers_before_summary():
    wc_df = pd.DataFrame(
        {
            "sample_id": ["S1", "S1", "S2"],
            "speaker": ["PAR", "INV", "PAR"],
            "word_count": [2, 99, 5],
        }
    )

    filtered = analysis._drop_excluded_speaker_rows(wc_df, ["INV"])
    summary = analysis._summarize_word_counts(filtered, word_count_field="word_count")

    s1 = summary.loc[summary["sample_id"] == "S1"].iloc[0]
    assert set(filtered["speaker"]) == {"PAR"}
    assert s1["no_utt_coded"] == 1
    assert s1["total_words"] == 2


def test_word_count_reliability_output_uses_custom_utterance_id(monkeypatch, tmp_path):
    calls = {}

    def fake_icc(df, target_col, col_org, col_rel):
        calls["target_col"] = target_col
        return 0.5

    monkeypatch.setattr(rel_evaluation, "calculate_icc_from_pingouin", fake_icc)

    merged = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S1"],
            "expanded_utterance_id": ["U1", "U2"],
            "word_count_org": [2, 3],
            "word_count_rel": [2, 4],
            "agmt": [1, 1],
            "abs_diff": [0, 1],
            "perc_diff": [0.0, 33.3],
            "perc_sim": [100.0, 66.7],
        }
    )

    rel_evaluation._write_word_rel_outputs(
        wc_merged=merged,
        out_dir=tmp_path,
        rel_name="word_count_reliability.xlsx",
        utterance_id_field="expanded_utterance_id",
    )

    assert calls["target_col"] == "expanded_utterance_id"
