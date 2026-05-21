from __future__ import annotations

import pandas as pd
import random
import pytest

from diaad.metadata.discovery import MultipleFilesFoundError
from diaad.coding.powers import analysis, files, rates, rel_evaluation, rel_reselection
from diaad.coding.utils import reselection_utils


def test_powers_generation_helpers_accept_custom_sample_id(monkeypatch):
    df = pd.DataFrame(
        {
            "expanded_sample_id": ["S2", "S1", "S1"],
            "expanded_utterance_id": ["S2-1", "S1-1", "S1-2"],
            "speaker": ["P", "P", "INV"],
            "utterance": ["two", "one", "prompt"],
        }
    )

    monkeypatch.setattr(files.random, "shuffle", lambda values: None)

    prepared = files._prepare_powers_dataframe(
        df,
        metadata_fields={},
        exclude_participants=["INV"],
        sample_id_field="expanded_sample_id",
    )
    assigned, primary_map, segments = files._assign_primary_coders(
        prepared,
        ["1", "2"],
        sample_id_field="expanded_sample_id",
    )
    section_e = files._build_section_e_dataframe(
        assigned,
        sample_id_field="expanded_sample_id",
    )

    assert "expanded_sample_id" in assigned.columns
    assert primary_map == {"S2": "1", "S1": "2"}
    assert segments == [["S2"], ["S1"]]
    assert list(section_e["expanded_sample_id"]) == ["S2", "S1"]
    assert list(assigned.loc[assigned["speaker"] == "INV", "content_words"]) == ["NA"]


def test_powers_analysis_groups_by_custom_sample_id():
    df = pd.DataFrame(
        {
            "expanded_sample_id": ["A", "A", "B"],
            "speaker": ["P", "P", "P"],
            "turn_type": ["T", "T", "ST"],
            "speech_units": [1, 2, 3],
            "content_words": [2, 3, 4],
        }
    )

    labeled = analysis.add_turn_labels(df)
    sheets = analysis.compute_level_summaries(
        labeled,
        sample_id_field="expanded_sample_id",
    )

    assert "expanded_sample_id" in sheets["Dialogs"].columns
    assert dict(zip(sheets["Dialogs"]["expanded_sample_id"], sheets["Dialogs"]["speech_units_sum"])) == {
        "A": 3,
        "B": 3,
    }


def test_powers_analysis_requires_one_exact_coding_file(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    for root in [input_dir / "site_a", output_dir / "site_b"]:
        root.mkdir(parents=True)
        pd.DataFrame(
            {
                "sample_id": ["S1"],
                "speaker": ["P"],
                "turn_type": ["T"],
                "speech_units": [1],
                "content_words": [2],
            }
        ).to_excel(root / "powers_coding.xlsx", index=False)

    with pytest.raises(MultipleFilesFoundError):
        analysis.analyze_powers_coding(
            input_dir=input_dir,
            output_dir=output_dir,
        )


def test_powers_reliability_merges_on_custom_identifiers():
    org = pd.DataFrame(
        {
            "expanded_sample_id": ["A", "A"],
            "expanded_utterance_id": ["1", "2"],
            "speech_units": [1, 3],
            "turn_type": ["T", "ST"],
        }
    )
    rel = pd.DataFrame(
        {
            "expanded_sample_id": ["A", "A"],
            "expanded_utterance_id": ["1", "2"],
            "speech_units": [2, 3],
            "turn_type": ["T", "MT"],
        }
    )

    merged = rel_evaluation._merge_powers_reliability(
        org,
        rel,
        sample_id_field="expanded_sample_id",
        utterance_id_field="expanded_utterance_id",
    )

    assert len(merged) == 2
    assert {"expanded_sample_id", "expanded_utterance_id"}.issubset(merged.columns)
    assert {"speech_units_org", "speech_units_rel"}.issubset(merged.columns)


def test_powers_reselection_uses_custom_sample_id():
    df_org = pd.DataFrame(
        {
            "expanded_sample_id": ["A", "B"],
            "utterance": ["one", "two"],
            "comment": ["", ""],
            "speech_units": [1, 2],
        }
    )

    out = rel_reselection._build_powers_reliability_frame(
        df_org=df_org,
        rel_template=pd.DataFrame(columns=["expanded_sample_id", "utterance", "comment", "speech_units"]),
        re_ids=["B"],
        automate_powers=False,
        sample_id_field="expanded_sample_id",
    )

    assert list(out["expanded_sample_id"]) == ["B"]
    assert out.iloc[0]["speech_units"] == ""


def test_reselection_utils_accept_custom_sample_id():
    df = pd.DataFrame({"expanded_sample_id": ["A", "B", "C"]})
    rel = pd.DataFrame({"expanded_sample_id": ["B"]})

    used = reselection_utils.collect_used_ids(
        [rel],
        sample_id_col="expanded_sample_id",
    )
    selected = reselection_utils.select_new_samples(
        df,
        used,
        frac=1,
        rng=random.Random(1),
        sample_id_col="expanded_sample_id",
    )

    assert used == {"B"}
    assert set(selected) == {"A", "C"}


def test_powers_rates_uses_custom_sample_id():
    df = pd.DataFrame(
        {
            "expanded_sample_id": ["A"],
            "source_file": ["powers_analysis.xlsx"],
            "speech_units_sum": [30],
            "prop_repairs": [0.5],
        }
    )

    numerators = rates.infer_powers_rate_numerators(
        df,
        sample_id_field="expanded_sample_id",
    )
    out = rates.finalize_powers_rates_columns(
        df.assign(speaking_time=60, speaking_minutes=1, speech_units_sum_per_min=30),
        numerator_cols=numerators,
        sample_id_field="expanded_sample_id",
    )

    assert numerators == ["speech_units_sum"]
    assert list(out.columns) == [
        "expanded_sample_id",
        "source_file",
        "speech_units_sum",
        "speaking_time",
        "speaking_minutes",
        "speech_units_sum_per_min",
    ]
