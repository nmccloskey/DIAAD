from __future__ import annotations

import json

import pytest

from diaad.coding.target_vocab import resources
from ...helpers import sample_target_vocab_resource


def test_validate_resource_accepts_valid_shape():
    resources.validate_resource(sample_target_vocab_resource())


def test_validate_resource_rejects_duplicate_base_forms():
    resource = sample_target_vocab_resource()
    resource["base_forms"] = ["cat", "cat"]

    with pytest.raises(ValueError, match="duplicate base_forms"):
        resources.validate_resource(resource)


def test_load_resources_from_path_adds_runtime_fields(tmp_path):
    path = tmp_path / "resource.json"
    path.write_text(json.dumps(sample_target_vocab_resource("StoryX")), encoding="utf-8")

    loaded = resources.load_resources_from_path(path)

    assert "StoryX" in loaded
    assert loaded["StoryX"]["_base_form_set"] == {"cat", "dog"}
    assert loaded["StoryX"]["_reverse_variant_lookup"]["dogs"] == "dog"
