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
            "num_bins": 3.0,
            "num_coders": "2",
            "metadata_fields": {"group": r"group\d+"},
        },
        advanced={
            "coding_blind_cols": ["sample_id", "speaker"],
            "analysis_blind_cols": ["sample_id"],
            "id_cols": ["sample_id", "utterance_id"],
        },
    )

    config = ConfigManager(config_dir)

    assert config.input_dir == "inputs"
    assert config.output_dir == "outputs"
    assert config.random_seed == 7
    assert config.reliability_fraction == 0.5
    assert config.shuffle_samples is False
    assert config.exclude_participants == ["INV"]
    assert config.num_bins == 3
    assert config.num_coders == 2
    assert config.metadata_fields_config == {"tiers": {"group": r"group\d+"}}
    assert config.coding_blind_cols == ["sample_id", "speaker"]


def test_config_manager_rejects_invalid_reliability_fraction(tmp_path):
    config_dir = _make_config_dir(tmp_path, project={"reliability_fraction": 2})

    with pytest.raises(ValueError, match="reliability_fraction"):
        ConfigManager(config_dir)


def test_advanced_config_blinding_helpers():
    advanced = AdvancedConfig(coding_blind_cols=["sample_id"], analysis_blind_cols=[])

    assert advanced.should_blind("coding") is True
    assert advanced.should_blind("analysis") is True
    assert advanced.blinded_suffix == "_blinded"

    with pytest.raises(ValueError, match="Unknown blinding mode"):
        advanced.get_blind_cols("bad")
