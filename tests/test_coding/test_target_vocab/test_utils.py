from __future__ import annotations

import pandas as pd

from diaad.coding.target_vocab import utils
from ...helpers import sample_target_vocab_resource


def _resources():
    resource = sample_target_vocab_resource("StoryA")
    resource["_reverse_variant_lookup"] = {"cat": "cat", "cats": "cat", "dog": "dog", "dogs": "dog"}
    return {"StoryA": resource}


def test_generate_token_columns_and_reformat():
    cols = utils.generate_token_columns({"StoryA"}, resources=_resources())
    assert cols == ["Sto_cat", "Sto_dog"]

    reformatted = utils.reformat("He's got 2 dogs [*] [=! laugh] xxx")
    assert "two" in reformatted
    assert "xxx" not in reformatted


def test_id_core_words_and_percentiles():
    stats = utils.id_core_words("StoryA", "cat cats dog", resources=_resources())
    assert stats["num_base_forms_produced"] == 2
    assert stats["num_core_token_matches"] == 3

    norm_df = pd.DataFrame({"group": [0, 0, 1, 1], "score": [1, 2, 1, 3]})
    percentiles = utils.get_percentiles(2, norm_df, "score", group_col="group")
    assert set(percentiles) == {"control_percentile", "pwa_percentile"}


def test_prepare_target_vocab_inputs_for_unblind_mode(monkeypatch):
    df = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "speaker": ["PAR", "PAR", "INV"],
            "story": ["StoryA", "Other", "StoryA"],
            "word_count": [10, 20, 5],
        }
    )
    monkeypatch.setattr(utils, "find_target_vocab_inputs", lambda *args, **kwargs: ("unblind", df))

    utt_df, present = utils.prepare_target_vocab_inputs(
        "input",
        "output",
        exclude_speakers=["INV"],
        stimulus_field="story",
        resources=_resources(),
    )

    assert list(utt_df["narrative"]) == ["StoryA"]
    assert list(utt_df["speaker"]) == ["PAR"]
    assert present == {"StoryA"}


def test_prepare_target_vocab_inputs_accepts_custom_sample_id(monkeypatch):
    df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S2"],
            "story": ["StoryA", "Other"],
            "word_count": [10, 20],
        }
    )
    monkeypatch.setattr(utils, "find_target_vocab_inputs", lambda *args, **kwargs: ("unblind", df))

    utt_df, present = utils.prepare_target_vocab_inputs(
        "input",
        "output",
        exclude_speakers=[],
        stimulus_field="story",
        resources=_resources(),
        sample_id_field="expanded_sample_id",
    )

    assert list(utt_df["expanded_sample_id"]) == ["S1"]
    assert present == {"StoryA"}
