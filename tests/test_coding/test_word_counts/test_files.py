from __future__ import annotations

import pandas as pd

from diaad.coding.word_counts import files


def test_count_words_handles_annotations_and_fillers():
    assert files.count_words("um I can't [=! laugh] go 2 times") == 5
    assert files.count_words(None) == 0


def test_cu_neutrality_and_word_count_preparation():
    df = pd.DataFrame(
        {
            "sample_id": ["S1", "S1"],
            "utterance_id": ["U1", "U2"],
            "speaker": ["PAR", "INV"],
            "utterance": ["hello world", "ignored"],
            "comment": ["", ""],
            "c1_cu": [1, None],
        }
    )

    prepared = files._prepare_wc_df(df, source_type="cu", exclude_speakers=["INV"])

    assert list(prepared["word_count"]) == [2, "NA"]
    assert files._get_cu_columns(df) == ["c1_cu"]
    assert files._is_neutral_value("NA") is True


def test_assign_wc_coders_for_one_coder(monkeypatch):
    monkeypatch.setattr(files.random, "sample", lambda seq, k: list(seq)[:k])
    wc_df = pd.DataFrame({"sample_id": ["S1", "S1", "S2"], "coder_id": ["", "", ""]})

    primary, reliability = files._assign_wc_coders(wc_df, num_coders=1, frac=0.5)

    assert set(primary["coder_id"]) == {1}
    assert set(reliability["coder_id"]) == {1}


def test_assign_wc_coders_supports_strict_string_inference(monkeypatch):
    monkeypatch.setattr(files.random, "sample", lambda seq, k: list(seq)[:k])

    with pd.option_context("future.infer_string", True):
        wc_df = pd.DataFrame(
            {"sample_id": ["S1", "S1", "S2"], "coder_id": ["", "", ""]}
        )
        primary, reliability = files._assign_wc_coders(
            wc_df,
            num_coders=2,
            frac=1.0,
        )

    assert set(primary["coder_id"]) == {1, 2}
    assert set(reliability["coder_id"]) == {1, 2}
