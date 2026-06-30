from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from diaad.transcripts import transcript_tables
from psair.metadata.metadata_fields import MetadataManager


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
    mismatches = pd.read_excel(
        written[0],
        sheet_name=transcript_tables.METADATA_MISMATCH_SHEET,
    )

    assert list(samples["group"]) == ["g:group1"]
    assert list(samples["metadata_mismatch"]) == [0]
    assert mismatches.empty
    assert list(joined["sample_id"]) == ["S001", "S001"]
    assert list(joined["utterance"]) == ["hello there", "second"]


def test_tabularize_transcripts_marks_and_logs_metadata_mismatches(
    tmp_path,
    caplog,
):
    chats = {
        "RU35_PreTx_BrokenWindow.cha": SimpleNamespace(
            utterances=lambda: [
                SimpleNamespace(participant="PAR", tiers={"PAR": "broken window"}),
            ]
        )
    }
    metadata_fields = MetadataManager(
        {"tiers": {"site": ["AC", "BU", "TU"], "timepoint": ["PreTx"]}}
    ).metadata_fields

    with caplog.at_level(logging.WARNING, logger="RunLogger"):
        written = transcript_tables.tabularize_transcripts(
            metadata_fields=metadata_fields,
            chats=chats,
            output_dir=tmp_path,
        )

    samples = transcript_tables.extract_transcript_data(written[0], kind="sample")
    mismatches = pd.read_excel(
        written[0],
        sheet_name=transcript_tables.METADATA_MISMATCH_SHEET,
    )

    assert pd.isna(samples.loc[0, "site"])
    assert samples.loc[0, "timepoint"] == "PreTx"
    assert samples.loc[0, "metadata_mismatch"] == 1
    assert list(mismatches["metadata_field"]) == ["site"]
    assert list(mismatches["source_path"]) == ["RU35_PreTx_BrokenWindow.cha"]
    assert list(mismatches["reason"]) == ["no_match"]
    assert pd.isna(mismatches.loc[0, "written_value"])
    assert "Metadata mismatch while tabularizing transcript" in caplog.text
    assert "RU35_PreTx_BrokenWindow.cha" in caplog.text
    assert "site" in caplog.text


def test_tabularize_transcripts_accepts_custom_identifier_fields(tmp_path):
    chats = {
        "group1/sample1.cha": SimpleNamespace(
            utterances=lambda: [
                SimpleNamespace(participant="PAR", tiers={"PAR": "hello there"}),
            ]
        )
    }

    written = transcript_tables.tabularize_transcripts(
        metadata_fields={},
        chats=chats,
        output_dir=tmp_path,
        sample_id_field="expanded_sample_id",
        utterance_id_field="expanded_utterance_id",
    )

    samples = transcript_tables.extract_transcript_data(written[0], kind="sample")
    utterances = transcript_tables.extract_transcript_data(written[0], kind="utterance")
    joined = transcript_tables.extract_transcript_data(
        written[0],
        kind="joined",
        sample_id_field="expanded_sample_id",
    )

    assert list(samples.columns[:2]) == ["expanded_sample_id", "file"]
    assert list(utterances.columns[:2]) == ["expanded_sample_id", "expanded_utterance_id"]
    assert list(joined["expanded_sample_id"]) == ["S001"]


def test_tabularize_transcripts_accepts_custom_transcript_table_filename(tmp_path):
    chats = {
        "group1/sample1.cha": SimpleNamespace(
            utterances=lambda: [
                SimpleNamespace(participant="PAR", tiers={"PAR": "custom file"}),
            ]
        )
    }

    written = transcript_tables.tabularize_transcripts(
        metadata_fields={},
        chats=chats,
        output_dir=tmp_path,
        transcript_table_filename="site_transcript_tables.xlsx",
    )

    expected = tmp_path / "transcript_tables" / "site_transcript_tables.xlsx"
    assert written == [str(expected)]
    assert expected.exists()
    assert not (tmp_path / "transcript_tables" / "transcript_tables.xlsx").exists()


def test_extract_transcript_data_rejects_bad_kind(tmp_path):
    path = tmp_path / "empty.xlsx"
    pd.DataFrame({"sample_id": ["S1"]}).to_excel(path, index=False)

    with pytest.raises(ValueError, match="Invalid kind"):
        transcript_tables.extract_transcript_data(path, kind="bad")


def test_tabularize_transcripts_extracts_exact_sandwich_narrative(tmp_path):
    chats = {
        "P1_BrokenWindow_post.cha": SimpleNamespace(
            utterances=lambda: [SimpleNamespace(participant="PAR", tiers={"PAR": "window"})]
        ),
        "P1_Sandwich_post.cha": SimpleNamespace(
            utterances=lambda: [SimpleNamespace(participant="PAR", tiers={"PAR": "sandwich"})]
        ),
        "P1_sandwich_post.cha": SimpleNamespace(
            utterances=lambda: [SimpleNamespace(participant="PAR", tiers={"PAR": "lowercase"})]
        ),
    }
    metadata_fields = MetadataManager(
        {"tiers": {"narrative": ["Sandwich", "BrokenWindow"]}}
    ).metadata_fields

    written = transcript_tables.tabularize_transcripts(
        metadata_fields=metadata_fields,
        chats=chats,
        output_dir=tmp_path,
    )

    samples = transcript_tables.extract_transcript_data(written[0], kind="sample")
    by_file = samples.set_index("file")

    assert by_file.loc["P1_BrokenWindow_post", "narrative"] == "BrokenWindow"
    assert by_file.loc["P1_Sandwich_post", "narrative"] == "Sandwich"
    assert pd.isna(by_file.loc["P1_sandwich_post", "narrative"])
