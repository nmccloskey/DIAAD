from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from diaad.transcripts import transcript_tables


class FakeMetadataField:
    def __init__(self, prefix):
        self.prefix = prefix

    def match_path_parts(self, parts, source=None):
        return f"{self.prefix}:{parts[0]}"


def test_zero_pad_and_sample_id_map():
    assert transcript_tables.zero_pad(7, 3) == 3
    assert transcript_tables.zero_pad(1000, 3) == 4

    file_to_sample, shuffled = transcript_tables._build_sample_id_map(
        ["a.cha", "b.cha"],
        shuffle=False,
        rng=None,
        sample_pad=3,
    )

    assert file_to_sample == {"a.cha": "S001", "b.cha": "S002"}
    assert shuffled == {}


def test_tabularize_transcripts_writes_excel_and_extracts_joined(tmp_path):
    chats = {
        "group1/sample1.cha": SimpleNamespace(
            utterances=lambda: [
                SimpleNamespace(participant="PAR", tiers={"PAR": "hello there", "%com": "note"}),
                SimpleNamespace(participant="INV", tiers={"INV": "second"}),
            ]
        )
    }
    metadata_fields = {"group": FakeMetadataField("g")}

    written = transcript_tables.tabularize_transcripts(
        metadata_fields=metadata_fields,
        chats=chats,
        output_dir=tmp_path,
    )

    assert len(written) == 1

    joined = transcript_tables.extract_transcript_data(written[0], kind="joined")
    samples = transcript_tables.extract_transcript_data(written[0], kind="sample")

    assert list(samples["group"]) == ["g:group1"]
    assert list(joined["sample_id"]) == ["S001", "S001"]
    assert list(joined["utterance"]) == ["hello there", "second"]


def test_extract_transcript_data_rejects_bad_kind(tmp_path):
    path = tmp_path / "empty.xlsx"
    pd.DataFrame({"sample_id": ["S1"]}).to_excel(path, index=False)

    with pytest.raises(ValueError, match="Invalid kind"):
        transcript_tables.extract_transcript_data(path, kind="bad")
