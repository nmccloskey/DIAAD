from __future__ import annotations

import pandas as pd

from diaad.coding.templates import samples, times, utterances, utils


def _write_transcript_table(path, sample_df, utt_df):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sample_df.to_excel(writer, sheet_name="samples", index=False)
        utt_df.to_excel(writer, sheet_name="utterances", index=False)


def test_template_utils_accept_custom_sample_id(monkeypatch):
    df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S2", "S3", "S4"],
            "story": ["A", "B", "C", "D"],
        }
    )

    stim_df = utils.prepare_stimulus_lookup(
        df,
        "story",
        sample_id_field="expanded_sample_id",
    )

    monkeypatch.setattr(utils, "assign_coders", lambda coders: [tuple(coders), tuple(reversed(coders))])
    monkeypatch.setattr(utils.random, "sample", lambda seq, k: list(seq)[:k])

    assigned, segments, assignments = utils.assign_template_coders(
        stim_df,
        coder_ids=["1", "2"],
        sample_id_field="expanded_sample_id",
    )
    rel_df = utils.build_reliability_subset(
        assigned,
        frac=0.5,
        coder_ids=["1", "2"],
        segments=segments,
        assignments=assignments,
        sample_id_field="expanded_sample_id",
    )

    assert list(stim_df.columns) == ["expanded_sample_id", "stimulus"]
    assert assigned["coder_id"].notna().all()
    assert rel_df is not None
    assert "expanded_sample_id" in rel_df.columns


def test_sample_template_accepts_custom_sample_id(tmp_path):
    path = tmp_path / "transcript_tables.xlsx"
    sample_df = pd.DataFrame(
        {"expanded_sample_id": ["S1", "S2"], "narrative": ["A", "B"]}
    )
    utt_df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S2"],
            "expanded_utterance_id": ["S1-U1", "S2-U1"],
            "utterance": ["one", "two"],
        }
    )
    _write_transcript_table(path, sample_df, utt_df)

    df, _ = samples.build_sample_coding_template(
        path,
        coder_ids=["1"],
        sample_id_field="expanded_sample_id",
    )
    balanced = samples.add_balanced_bins(
        df,
        num_bins=2,
        sample_id_col="expanded_sample_id",
    )

    assert list(df.columns) == ["expanded_sample_id", "coder_id", "stimulus", "bin"]
    assert list(balanced["expanded_sample_id"]) == ["S1", "S2"]


def test_utterance_template_accepts_custom_identifiers(tmp_path):
    path = tmp_path / "transcript_tables.xlsx"
    sample_df = pd.DataFrame(
        {"expanded_sample_id": ["S1"], "narrative": ["StoryA"]}
    )
    utt_df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1"],
            "expanded_utterance_id": ["S1-U1"],
            "utterance": ["hello"],
        }
    )
    _write_transcript_table(path, sample_df, utt_df)

    df, _ = utterances.build_utterance_coding_template(
        path,
        coder_ids=["1"],
        sample_id_field="expanded_sample_id",
        utterance_id_field="expanded_utterance_id",
    )

    assert list(df.columns) == [
        "expanded_sample_id",
        "expanded_utterance_id",
        "coder_id",
        "stimulus",
        "utterance",
    ]


def test_speaking_time_template_accepts_custom_sample_id(tmp_path):
    path = tmp_path / "transcript_tables.xlsx"
    sample_df = pd.DataFrame({"expanded_sample_id": ["S2", "S1", "S1"]})
    utt_df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1"],
            "expanded_utterance_id": ["S1-U1"],
            "utterance": ["one"],
        }
    )
    _write_transcript_table(path, sample_df, utt_df)

    df = times.build_speaking_time_template(
        path,
        sample_id_field="expanded_sample_id",
    )

    assert list(df.columns) == ["expanded_sample_id", "speaking_time"]
    assert list(df["expanded_sample_id"]) == ["S1", "S2"]
