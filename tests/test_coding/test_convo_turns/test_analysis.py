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
            "group": ["G1", "G1"],
            "turns": ["0.1", "1.0"],
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
