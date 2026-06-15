from __future__ import annotations

import pandas as pd

from diaad.coding.convo_turns import analysis


def test_extract_turn_counts_and_stats():
    assert analysis.extract_turn_counts("0.1..12.0") == {"0": 2, "1": 2, "2": 1}

    turns, mark1, mark2 = analysis.extract_turn_stats("0.1..12.0")
    assert turns["1"] == 2
    assert mark1["1"] == 0
    assert mark2["1"] == 1


def test_compute_group_and_transition_metrics():
    df = pd.DataFrame(
        {
            "group": ["G1", "G1", "G1", "G1"],
            "speaker": ["0", "1", "1", "0"],
            "sequence_position": [1, 2, 3, 4],
            "mark1": [0, 0, 0, 0],
            "mark2": [0, 0, 0, 0],
            "source": ["dct_coding"] * 4,
        }
    )

    metrics = analysis.compute_transition_metrics(df)

    assert "G1" in metrics["transition_matrices"]
    assert set(metrics["speaker_ratios"].columns) == {
        "group",
        "participant_to_participant",
        "participant_to_clinician",
        "clinician_to_participant",
    }


def test_analyze_convo_turns_file_returns_expected_levels():
    df = pd.DataFrame(
        {
            "group": ["G1"],
            "session": ["A"],
            "bin": [1],
            "turns": ["0.1..1"],
        }
    )

    result = analysis._analyze_convo_turns_file(df)

    assert "speaker_level" in result
    assert "group_level" in result
    assert "session_level" in result
    assert "participation_level" in result


def test_bin_level_turns_are_sorted_by_grouping_and_speaker():
    df = pd.DataFrame(
        {
            "group": ["G1", "G1"],
            "session": ["A", "A"],
            "bin": ["bin_2", "bin_1"],
            "turns": ["3.2.1.0", "3.2.1.0"],
        }
    )

    result = analysis._analyze_convo_turns_file(df)

    assert result["bin_level"][["bin", "speaker"]].to_dict("records") == [
        {"bin": "bin_1", "speaker": "0"},
        {"bin": "bin_1", "speaker": "1"},
        {"bin": "bin_1", "speaker": "2"},
        {"bin": "bin_1", "speaker": "3"},
        {"bin": "bin_2", "speaker": "0"},
        {"bin": "bin_2", "speaker": "1"},
        {"bin": "bin_2", "speaker": "2"},
        {"bin": "bin_2", "speaker": "3"},
    ]


def test_analyze_digital_convo_turns_uses_configured_primary_file(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    primary = pd.DataFrame(
        {
            "sample_id": ["S001"],
            "turns": ["0.1"],
        }
    )
    reliability = pd.DataFrame(
        {
            "sample_id": ["S002"],
            "turns": ["1.0"],
        }
    )
    primary.to_excel(input_dir / "conversation_turns.xlsx", index=False)
    reliability.to_excel(
        input_dir / "conversation_turns_reliability.xlsx",
        index=False,
    )

    analysis.analyze_digital_convo_turns(
        input_dir=input_dir,
        output_dir=output_dir,
        dct_coding_filename="conversation_turns.xlsx",
    )

    assert (output_dir / "conversation_turns_analysis.xlsx").exists()
    assert not (output_dir / "conversation_turns_reliability_analysis.xlsx").exists()


def test_dct_events_map_zero_to_first_excluded_speaker():
    df = pd.DataFrame(
        {
            "sample_id": ["S001"],
            "turns": ["0.1..0"],
        }
    )

    events, has_session, has_bin = analysis._events_from_dct_coding(
        df,
        exclude_speakers=["INV", "INV2"],
    )

    assert has_session is False
    assert has_bin is False
    assert list(events["speaker"]) == ["INV", "1", "INV"]
    assert list(events["mark2"]) == [0, 1, 0]


def test_transcript_rows_map_excluded_speakers_and_keep_tokens():
    df = pd.DataFrame(
        {
            "sample_id": ["S001", "S001", "S001", "S001"],
            "position": [1, 2, 3, 4],
            "position_sub": [0, 0, 0, 0],
            "speaker": ["INV", "CHI", "INV2", "MOT"],
        }
    )

    events, has_session, has_bin = analysis._events_from_transcript_rows(
        df,
        exclude_speakers=["INV", "INV2"],
    )

    assert has_session is False
    assert has_bin is False
    assert list(events["speaker"]) == ["INV", "CHI", "INV", "MOT"]
    assert list(events["sequence_position"]) == [1, 2, 3, 4]


def test_analyze_digital_convo_turns_from_transcript_table(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    table_dir = input_dir / "transcript_tables"
    table_dir.mkdir(parents=True)
    transcript_table = table_dir / "transcript_tables.xlsx"

    samples = pd.DataFrame(
        {
            "sample_id": ["S001"],
            "session": ["visit_1"],
        }
    )
    utterances = pd.DataFrame(
        {
            "sample_id": ["S001", "S001", "S001"],
            "utterance_id": ["U001", "U002", "U003"],
            "position": [1, 2, 3],
            "position_sub": [0, 0, 0],
            "speaker": ["INV", "CHI", "INV2"],
            "utterance": ["hello", "hi", "again"],
            "comment": ["", "", ""],
        }
    )
    with pd.ExcelWriter(transcript_table, engine="openpyxl") as writer:
        samples.to_excel(writer, sheet_name="samples", index=False)
        utterances.to_excel(writer, sheet_name="utterances", index=False)

    analysis.analyze_digital_convo_turns(
        input_dir=input_dir,
        output_dir=output_dir,
        use_transcript_tables=True,
        exclude_speakers=["INV", "INV2"],
    )

    out_file = output_dir / "transcript_tables_turns_analysis.xlsx"
    assert out_file.exists()
    with pd.ExcelFile(out_file, engine="openpyxl") as xls:
        speakers = pd.read_excel(xls, sheet_name="speaker_level_turns")
        matrix = pd.read_excel(xls, sheet_name="speaker_matrix_S001", index_col=0)

    assert set(speakers["speaker"]) == {"INV", "CHI"}
    assert "INV2" not in set(speakers["speaker"])
    assert {"INV", "CHI"} <= set(matrix.columns.astype(str))
