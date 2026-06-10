from __future__ import annotations

import pandas as pd

from diaad.coding.utils import transcript


def test_resolve_stim_cols():
    assert transcript.resolve_stim_cols("story") == ["story"]
    assert transcript.resolve_stim_cols("") == transcript.DEFAULT_STIM_COLS


def test_drop_excluded_speaker_rows_filters_case_insensitively():
    df = pd.DataFrame(
        {
            "speaker": [" PWA ", "Clinician", "partner", None],
            "utterance": ["a", "b", "c", "d"],
        }
    )

    result = transcript.drop_excluded_speaker_rows(df, ["pwa", "PARTNER"])

    assert result["utterance"].tolist() == ["b", "d"]


def test_drop_excluded_speaker_rows_without_speaker_column_returns_input():
    df = pd.DataFrame({"utterance": ["a"]})

    assert transcript.drop_excluded_speaker_rows(df, ["pwa"]) is df
