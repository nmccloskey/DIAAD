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
