from __future__ import annotations

from types import SimpleNamespace

import pytest

from diaad.core.config_overrides import (
    apply_config_overrides,
    build_cli_config_overrides,
    parse_config_overrides,
)


def test_parse_config_overrides_maps_bare_keys_to_sections() -> None:
    overrides = parse_config_overrides(
        [
            "input_dir=data/input/siteA",
            "powers_coding_file=siteA_powers_coding.xlsx",
            "sample_id_field=expanded_sample_id",
            "automate_powers=false",
            "auto_tabularize=true",
        ]
    )

    assert overrides == {
        "project.input_dir": "data/input/siteA",
        "advanced.powers_coding_file": "siteA_powers_coding.xlsx",
        "advanced.sample_id_field": "expanded_sample_id",
        "project.automate_powers": False,
        "project.auto_tabularize": True,
    }


def test_parse_config_overrides_accepts_explicit_sections() -> None:
    overrides = parse_config_overrides(
        [
            "project.output_dir=data/output/siteA",
            "advanced.powers_reliability_file=siteA_powers_rel.xlsx",
            "advanced.utterance_id_field=expanded_utterance_id",
        ]
    )

    assert overrides == {
        "project.output_dir": "data/output/siteA",
        "advanced.powers_reliability_file": "siteA_powers_rel.xlsx",
        "advanced.utterance_id_field": "expanded_utterance_id",
    }


def test_parse_config_overrides_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown DIAAD config override key"):
        parse_config_overrides(["just_c2_powers=true"])
    with pytest.raises(ValueError, match="Unknown DIAAD config override key"):
        parse_config_overrides(["id" + "_cols=sample_id"])


def test_build_cli_config_overrides_prefers_direct_input_output_flags() -> None:
    args = SimpleNamespace(
        input_dir="direct/input",
        output_dir="direct/output",
        set_values=["input_dir=set/input", "powers_coding_file=custom.xlsx"],
    )

    assert build_cli_config_overrides(args) == {
        "project.input_dir": "direct/input",
        "project.output_dir": "direct/output",
        "advanced.powers_coding_file": "custom.xlsx",
    }


def test_apply_config_overrides_returns_copied_config_dicts() -> None:
    project = {"input_dir": "input", "output_dir": "output"}
    advanced = {"powers_coding_file": "powers_coding.xlsx"}

    new_project, new_advanced = apply_config_overrides(
        project,
        advanced,
        {
            "project.input_dir": "input/siteA",
            "advanced.powers_coding_file": "siteA_powers.xlsx",
        },
    )

    assert project["input_dir"] == "input"
    assert advanced["powers_coding_file"] == "powers_coding.xlsx"
    assert new_project["input_dir"] == "input/siteA"
    assert new_advanced["powers_coding_file"] == "siteA_powers.xlsx"
