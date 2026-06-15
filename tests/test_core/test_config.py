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
            "exclude_speakers": ["INV"],
            "auto_tabularize": "true",
            "num_bins": 3.0,
            "num_coders": "2",
            "metadata_fields": {"group": r"group\d+"},
        },
        advanced={
            "sample_id_column": "expanded_sample_id",
            "utterance_id_column": "expanded_utterance_id",
            "auto_blind": "true",
            "blind_columns": ["sample_id", "speaker"],
            "id_columns": ["expanded_sample_id"],
            "codebook_filename": "custom_codebook.xlsx",
            "powers_coding_filename": "custom_powers.xlsx",
            "powers_reliability_filename": "custom_powers_rel.xlsx",
            "spacy_model_name": "en_core_web_trf",
            "dct_coding_filename": "site_turns.xlsx",
            "dct_coding_reliability": "site_turns_reliability.xlsx",
        },
    )

    config = ConfigManager(config_dir)

    assert config.input_dir == "inputs"
    assert config.output_dir == "outputs"
    assert config.random_seed == 7
    assert config.reliability_fraction == 0.5
    assert config.shuffle_samples is False
    assert config.exclude_speakers == ["INV"]
    assert config.auto_tabularize is True
    assert config.num_bins == 3
    assert config.num_coders == 2
    assert config.metadata_fields_config == {"tiers": {"group": r"group\d+"}}
    assert config.sample_id_column == "expanded_sample_id"
    assert config.utterance_id_column == "expanded_utterance_id"
    assert config.auto_blind is True
    assert config.blind_columns == ["sample_id", "speaker"]
    assert config.id_columns == ["expanded_sample_id"]
    assert config.coding_blind_cols == ["sample_id", "speaker"]
    assert config.analysis_blind_cols == ["sample_id", "speaker"]
    assert config.codebook_filename == "custom_codebook.xlsx"
    assert config.powers_coding_filename == "custom_powers.xlsx"
    assert config.powers_reliability_filename == "custom_powers_rel.xlsx"
    assert config.spacy_model_name == "en_core_web_trf"
    assert config.dct_coding_filename == "site_turns.xlsx"
    assert config.dct_coding_reliability == "site_turns_reliability.xlsx"
    assert config.config_source["kind"] == "split_dir"
    assert config.config_source["defaults_applied"] is True


def test_config_manager_applies_overrides_and_records_diff(tmp_path):
    config_dir = _make_config_dir(
        tmp_path,
        project={"input_dir": "input", "output_dir": "output"},
        advanced={"powers_coding_filename": "powers_coding.xlsx"},
    )

    config = ConfigManager(
        config_dir,
        config_overrides={
            "project.input_dir": "input/siteA",
            "advanced.powers_coding_filename": "siteA_powers.xlsx",
        },
    )

    assert config.input_dir == "input/siteA"
    assert config.powers_coding_filename == "siteA_powers.xlsx"
    assert config.override_diff["project.input_dir"] == {
        "source": "cli",
        "old": "input",
        "new": "input/siteA",
    }
    assert config.override_diff["advanced.powers_coding_filename"] == {
        "source": "cli",
        "old": "powers_coding.xlsx",
        "new": "siteA_powers.xlsx",
    }


def test_config_manager_loads_nested_yaml_file(tmp_path):
    config_path = tmp_path / "effective_config.yaml"
    write_yaml(
        config_path,
        {
            "project": {
                "input_dir": "nested/input",
                "output_dir": "nested/output",
                "metadata_fields": {"site": ["AC", "BU"]},
            },
            "advanced": {
                "sample_id_column": "sample",
                "powers_coding_filename": "nested_powers.xlsx",
            },
        },
    )

    config = ConfigManager(config_path)

    assert config.input_dir == "nested/input"
    assert config.output_dir == "nested/output"
    assert config.metadata_fields_config == {"tiers": {"site": ["AC", "BU"]}}
    assert config.sample_id_column == "sample"
    assert config.powers_coding_filename == "nested_powers.xlsx"
    assert config.config_source["kind"] == "nested_file"
    assert config.config_source["path"] == str(config_path)


def test_config_manager_fills_missing_nested_sections_from_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    write_yaml(config_path, {"project": {"input_dir": "only/project"}})

    config = ConfigManager(config_path)

    assert config.input_dir == "only/project"
    assert config.powers_coding_filename == "powers_coding.xlsx"
    assert config.config_source["missing_sections"] == ["advanced"]


def test_config_manager_loads_directory_config_yaml(tmp_path):
    config_dir = tmp_path / "config"
    write_yaml(
        config_dir / "config.yaml",
        {
            "project": {"input_dir": "joined/input"},
            "advanced": {"auto_blind": True},
        },
    )

    config = ConfigManager(config_dir)

    assert config.input_dir == "joined/input"
    assert config.auto_blind is True
    assert config.config_source["kind"] == "nested_file"
    assert config.config_source["path"] == str(config_dir / "config.yaml")


def test_config_manager_rejects_ambiguous_config_directory(tmp_path):
    config_dir = tmp_path / "config"
    write_yaml(config_dir / "project.yaml", {"input_dir": "split/input"})
    write_yaml(config_dir / "config.yaml", {"project": {"input_dir": "nested/input"}})

    with pytest.raises(ValueError, match="Ambiguous config directory"):
        ConfigManager(config_dir)


def test_config_manager_uses_packaged_defaults_when_config_omitted():
    config = ConfigManager(None)

    assert config.input_dir == "diaad_data/input"
    assert config.output_dir == "diaad_data/output"
    assert config.config_source["kind"] == "packaged_default"
    assert config.config_source["path"] is None
    assert config.config_source["missing_sections"] == ["project", "advanced"]


def test_config_manager_rejects_explicit_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="Config source not found"):
        ConfigManager(tmp_path / "missing_config")


def test_packaged_default_config_parses_cleanly():
    config = ConfigManager(ConfigManager.default_config_path())

    assert config.to_dict()["project"]["input_dir"] == "diaad_data/input"
    assert config.to_dict()["advanced"]["sample_id_column"] == "sample_id"
    assert config.to_dict()["advanced"]["spacy_model_name"] == "en_core_web_sm"
    assert config.to_dict()["advanced"]["dct_coding_filename"] == "conversation_turns.xlsx"
    assert config.override_diff == {}


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
    advanced = AdvancedConfig(auto_blind=True, blind_columns=["sample_id"])

    assert advanced.should_blind("coding") is True
    assert advanced.should_blind("analysis") is True
    assert advanced.blinded_suffix == "_blinded"

    with pytest.raises(ValueError, match="Unknown blinding mode"):
        advanced.get_blind_cols("bad")


def test_advanced_config_defaults_do_not_auto_blind():
    advanced = AdvancedConfig()

    assert advanced.sample_id_column == "sample_id"
    assert advanced.utterance_id_column == "utterance_id"
    assert advanced.dct_coding_filename == "conversation_turns.xlsx"
    assert advanced.dct_coding_reliability == "conversation_turns_reliability.xlsx"
    assert advanced.blind_columns == ["sample_id"]
    assert advanced.id_columns == ["sample_id", "utterance_id"]
    assert advanced.should_blind("coding") is False
    assert advanced.should_blind("analysis") is False


def test_advanced_config_accepts_blind_columns():
    advanced = AdvancedConfig(blind_columns=["sample_id", "speaker"])

    assert advanced.blind_columns == ["sample_id", "speaker"]
    assert advanced.analysis_blind_cols == ["sample_id", "speaker"]


def test_advanced_config_normalizes_id_columns():
    advanced = AdvancedConfig(id_columns=[" sample ", "utterance", "sample"])

    assert advanced.id_columns == ["sample", "utterance"]


def test_advanced_config_rejects_empty_id_columns():
    with pytest.raises(ValueError, match="id_columns"):
        AdvancedConfig(id_columns=["sample_id", " "])
