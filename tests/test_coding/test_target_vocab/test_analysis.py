from __future__ import annotations

import pandas as pd

from diaad.coding.target_vocab import analysis
from ...helpers import sample_target_vocab_resource


def _resources():
    resource = sample_target_vocab_resource("StoryA")
    resource["_reverse_variant_lookup"] = {"cat": "cat", "cats": "cat", "dog": "dog", "dogs": "dog"}
    return {"StoryA": resource}


def test_compute_target_vocabulary_coverage_for_text():
    norm_df = pd.DataFrame({"group": [0, 1], "score": [1, 1]})
    norm_lookup = {
        "StoryA": {
            "accuracy": norm_df,
            "efficiency": norm_df,
        }
    }

    summary, detail_rows = analysis.compute_target_vocabulary_coverage_for_text(
        text="cat dogs dog",
        speaking_time=120,
        narrative="StoryA",
        norm_lookup=norm_lookup,
        sample_id="S1",
        resources=_resources(),
    )

    assert summary["num_base_forms_produced"] == 2
    assert summary["num_core_token_matches"] == 3
    assert summary["core_tokens_per_min"] == 1.5
    assert len(detail_rows) == 2


def test_extract_target_vocab_inputs_from_sample_df_and_ordered_columns():
    sample_df = pd.DataFrame(
        {
            "sample_id": ["S1", "S1"],
            "narrative": ["StoryA", "StoryA"],
            "speaking_time": [60, 60],
            "utterance": ["hello", "world"],
            "extra": [1, 2],
        }
    )

    extracted = analysis.extract_target_vocab_inputs_from_sample_df(sample_df)
    ordered = analysis._ordered_summary_columns(
        pd.DataFrame([{"narrative": "StoryA", "sample_id": "S1", "extra": 1}])
    )

    assert extracted["text"] == "hello world"
    assert ordered[:2] == ["sample_id", "narrative"]


def test_target_vocab_analysis_accepts_custom_sample_id():
    sample_df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1", "S1"],
            "narrative": ["StoryA", "StoryA"],
            "speaking_time": [60, 60],
            "utterance": ["cat", "dogs"],
        }
    )

    extracted = analysis.extract_target_vocab_inputs_from_sample_df(
        sample_df,
        sample_id_field="expanded_sample_id",
    )
    row, details = analysis._compute_target_vocab_for_sample(
        sample_df,
        norm_lookup={},
        resources=_resources(),
        sample_id_field="expanded_sample_id",
    )
    ordered = analysis._ordered_summary_columns(
        pd.DataFrame([{"narrative": "StoryA", "expanded_sample_id": "S1"}]),
        sample_id_field="expanded_sample_id",
    )

    assert extracted["expanded_sample_id"] == "S1"
    assert row["expanded_sample_id"] == "S1"
    assert details[0]["expanded_sample_id"] == "S1"
    assert ordered[:2] == ["expanded_sample_id", "narrative"]
