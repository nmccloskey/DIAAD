from __future__ import annotations

import pytest

from diaad.core.config import AdvancedConfig, ConfigManager
from ..helpers import write_yaml


def _make_config_dir(tmp_path, *, project=None, advanced=None):
    config_dir = tmp_path / "config"
    write_yaml(config_dir / "project.yaml", project or {})
    write_yaml(config_dir / "advanced.yaml", advanced or {})
    return config_dir


def test_config_manager_normalizes_values_from_yaml(tmp_path):
    config_dir = _make_config_dir(
        tmp_path,
        project={
            "input_dir": "inputs",
            "output_dir": "outputs",
            "random_seed": "7",
            "reliability_fraction": "0.5",
            "shuffle_samples": "false",
            "exclude_participants": ["INV"],
            "auto_tabularize": "true",
            "num_bins": 3.0,
            "num_coders": "2",
            "metadata_fields": {"group": r"group\d+"},
        },
        advanced={
            "sample_id_field": "expanded_sample_id",
            "utterance_id_field": "expanded_utterance_id",
            "auto_blind": "true",
            "blind_cols": ["sample_id", "speaker"],
            "codebook_filename": "custom_codebook.xlsx",
            "powers_coding_file": "custom_powers.xlsx",
            "powers_reliability_file": "custom_powers_rel.xlsx",
        },
    )

    config = ConfigManager(config_dir)

    assert config.input_dir == "inputs"
    assert config.output_dir == "outputs"
    assert config.random_seed == 7
    assert config.reliability_fraction == 0.5
    assert config.shuffle_samples is False
    assert config.exclude_participants == ["INV"]
    assert config.auto_tabularize is True
    assert config.num_bins == 3
    assert config.num_coders == 2
    assert config.metadata_fields_config == {"tiers": {"group": r"group\d+"}}
    assert config.sample_id_field == "expanded_sample_id"
    assert config.utterance_id_field == "expanded_utterance_id"
    assert config.auto_blind is True
    assert config.blind_cols == ["sample_id", "speaker"]
    assert config.coding_blind_cols == ["sample_id", "speaker"]
    assert config.analysis_blind_cols == ["sample_id", "speaker"]
    assert config.codebook_filename == "custom_codebook.xlsx"
    assert config.powers_coding_file == "custom_powers.xlsx"
    assert config.powers_reliability_file == "custom_powers_rel.xlsx"


def test_config_manager_applies_overrides_and_records_diff(tmp_path):
    config_dir = _make_config_dir(
        tmp_path,
        project={"input_dir": "input", "output_dir": "output"},
        advanced={"powers_coding_file": "powers_coding.xlsx"},
    )

    config = ConfigManager(
        config_dir,
        config_overrides={
            "project.input_dir": "input/siteA",
            "advanced.powers_coding_file": "siteA_powers.xlsx",
        },
    )

    assert config.input_dir == "input/siteA"
    assert config.powers_coding_file == "siteA_powers.xlsx"
    assert config.override_diff["project.input_dir"] == {
        "source": "cli",
        "old": "input",
        "new": "input/siteA",
    }
    assert config.override_diff["advanced.powers_coding_file"] == {
        "source": "cli",
        "old": "powers_coding.xlsx",
        "new": "siteA_powers.xlsx",
    }


def test_config_manager_rejects_invalid_reliability_fraction(tmp_path):
    config_dir = _make_config_dir(tmp_path, project={"reliability_fraction": 2})

    with pytest.raises(ValueError, match="reliability_fraction"):
        ConfigManager(config_dir)


def test_config_manager_defaults_do_not_auto_tabularize(tmp_path):
    config_dir = _make_config_dir(tmp_path)

    config = ConfigManager(config_dir)

    assert config.auto_tabularize is False
    assert config.to_dict()["project"]["auto_tabularize"] is False


def test_advanced_config_blinding_helpers():
    advanced = AdvancedConfig(auto_blind=True, blind_cols=["sample_id"])

    assert advanced.should_blind("coding") is True
    assert advanced.should_blind("analysis") is True
    assert advanced.blinded_suffix == "_blinded"

    with pytest.raises(ValueError, match="Unknown blinding mode"):
        advanced.get_blind_cols("bad")


def test_advanced_config_defaults_do_not_auto_blind():
    advanced = AdvancedConfig()

    assert advanced.sample_id_field == "sample_id"
    assert advanced.utterance_id_field == "utterance_id"
    assert advanced.blind_cols == ["sample_id"]
    assert advanced.should_blind("coding") is False
    assert advanced.should_blind("analysis") is False


def test_advanced_config_accepts_legacy_blind_cols():
    advanced = AdvancedConfig(coding_blind_cols=["sample_id", "speaker"])

    assert advanced.blind_cols == ["sample_id", "speaker"]
    assert advanced.analysis_blind_cols == ["sample_id", "speaker"]
