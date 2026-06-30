from __future__ import annotations

from pathlib import Path

from diaad.transcripts.transcription_reliability_evaluation import (
    _build_file_index,
    _match_reliability_pairs,
    _metadata_field_names,
)


class DefaultFileNameField:
    name = "file_name"
    kind = "default"

    def match_path_parts(self, parts, source=None):
        return Path(parts[-1]).stem


def test_default_file_name_metadata_matches_reliability_tagged_transcripts(tmp_path):
    original = tmp_path / "sample.cha"
    reliability = tmp_path / "sample_reliability.cha"
    original.write_text("@Begin\n@End\n", encoding="utf-8")
    reliability.write_text("@Begin\n@End\n", encoding="utf-8")

    metadata_fields = {"file_name": DefaultFileNameField()}
    org_index = _build_file_index(
        [original],
        metadata_fields,
        label="original",
        input_dir=tmp_path,
    )
    rel_index = _build_file_index(
        [reliability],
        metadata_fields,
        label="reliability",
        input_dir=tmp_path,
        reliability_tag="_reliability",
        strip_reliability_tag=True,
    )

    assert org_index == {("sample",): original}
    assert rel_index == {("sample",): reliability}
    assert _match_reliability_pairs(org_index, rel_index) == [
        (("sample",), original, reliability)
    ]


def test_empty_metadata_fields_still_emit_file_name_column():
    assert _metadata_field_names({}) == ["file_name"]