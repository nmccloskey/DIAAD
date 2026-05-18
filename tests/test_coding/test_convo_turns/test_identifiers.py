from __future__ import annotations

import pandas as pd

from diaad.coding.convo_turns import analysis, files, rel_evaluation, rel_reselection


def test_turn_template_helpers_accept_custom_sample_id(monkeypatch):
    sample_df = pd.DataFrame({"expanded_sample_id": ["S2", "S1", "S1"]})
    monkeypatch.setattr(
        files,
        "extract_transcript_data",
        lambda *args, **kwargs: sample_df,
    )

    base = files._build_turns_template_base(
        "transcript_tables.xlsx",
        sample_id_field="expanded_sample_id",
    )
    expanded = files.expand_by_coder(base, [""], insert_after="expanded_sample_id")
    sorted_df = files._sort_turns_template(
        expanded,
        sample_id_field="expanded_sample_id",
    )

    assert list(base.columns) == ["expanded_sample_id", "session", "bin", "turns"]
    assert list(sorted_df.columns) == [
        "expanded_sample_id",
        "coder_id",
        "session",
        "bin",
        "turns",
    ]


def test_turn_analysis_accepts_custom_sample_id():
    df = pd.DataFrame(
        {
            "expanded_sample_id": ["G1"],
            "session": ["A"],
            "bin": [1],
            "turns": ["0.1..1"],
        }
    )

    result = analysis._analyze_convo_turns_file(
        df,
        sample_id_field="expanded_sample_id",
    )

    assert "group_level" in result
    assert list(result["group_level"]["group"]) == ["G1"]


def test_turn_reliability_helpers_accept_custom_sample_id(tmp_path):
    org = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S2"],
            "session": ["A", "A"],
            "bin": ["bin_1", "bin_1"],
            "turns": ["0.1", "1.0"],
        }
    )
    rel = pd.DataFrame(
        {
            "expanded_sample_id": ["S1"],
            "session": ["A"],
            "bin": ["bin_1"],
            "turns": ["0.1"],
        }
    )

    org_norm = rel_evaluation._normalize_turn_file(
        org,
        label="org",
        sample_id_field="expanded_sample_id",
    )
    rel_norm = rel_evaluation._normalize_turn_file(
        rel,
        label="rel",
        sample_id_field="expanded_sample_id",
    )
    merged = pd.merge(
        org_norm,
        rel_norm,
        on=rel_evaluation._turn_key_cols("expanded_sample_id"),
        how="outer",
        suffixes=("_main", "_rel"),
    )
    merged["turns_main"] = merged["turns_main"].fillna("")
    merged["turns_rel"] = merged["turns_rel"].fillna("")

    counts = rel_evaluation._build_counts_sheet(
        merged,
        sample_id_field="expanded_sample_id",
    )
    sequences = rel_evaluation._build_sequences_sheet(
        merged,
        out_dir=tmp_path,
        sample_id_field="expanded_sample_id",
    )
    samples = rel_evaluation._build_sample_sheet(
        counts,
        sequences,
        sample_id_field="expanded_sample_id",
    )

    assert "expanded_sample_id" in counts.columns
    assert "expanded_sample_id" in samples.columns


def test_turn_reselection_accepts_custom_sample_id():
    df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S2"],
            "turns": ["0.1", "1.0"],
        }
    )

    out = rel_reselection._build_turns_reliability_frame(
        df,
        ["S2"],
        sample_id_field="expanded_sample_id",
    )

    assert list(out["expanded_sample_id"]) == ["S2"]
    assert list(out["turns"]) == [""]
