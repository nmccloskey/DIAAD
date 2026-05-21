from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from psair.metadata.discovery import MultipleFilesFoundError

from diaad.transcripts.detabularization import detabularize_transcripts


def _write_table(path: Path, samples: pd.DataFrame, utterances: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        samples.to_excel(writer, sheet_name="samples", index=False)
        utterances.to_excel(writer, sheet_name="utterances", index=False)


def test_detabularize_transcripts_writes_chat_files_and_derived_table(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    table_path = input_dir / "transcript_tables" / "transcript_tables.xlsx"

    samples = pd.DataFrame(
        [
            {
                "sample_id": "AC070",
                "input_order": 1,
                "shuffled_order": pd.NA,
                "study_id": "AC25",
                "test": "Pre",
                "narrative": "BrokenWindow",
            }
        ]
    )
    utterances = pd.DataFrame(
        [
            {
                "sample_id": "AC070",
                "utterance_id": "U0001",
                "position": 1,
                "position_sub": 0,
                "speaker": "PAR0",
                "utterance": "there was a window",
                "comment": "participant points",
            },
            {
                "sample_id": "AC070",
                "utterance_id": "U0002",
                "position": 2,
                "position_sub": 0,
                "speaker": "INV",
                "utterance": "what happened next?",
                "comment": pd.NA,
            },
        ]
    )
    _write_table(table_path, samples, utterances)

    written = detabularize_transcripts(input_dir=input_dir, output_dir=output_dir)

    chat_path = output_dir / "chat_files" / "AC25_Pre_BrokenWindow.cha"
    assert written == [str(chat_path)]
    assert chat_path.read_text(encoding="utf-8") == (
        "@Begin\n"
        "@Languages:\teng\n"
        "@Participants:\tPAR0 Participant, INV Investigator\n"
        "@ID:\teng|corpus_name|PAR0|||||Participant|||\n"
        "@ID:\teng|corpus_name|INV|||||Investigator|||\n"
        "*PAR0:\tthere was a window\n"
        "%com:\tparticipant points\n"
        "*INV:\twhat happened next?\n"
        "@End\n"
    )

    derived = pd.read_excel(output_dir / "transcript_tables" / "transcript_tables.xlsx", sheet_name="samples")
    assert list(derived["derived_file"]) == ["AC25_Pre_BrokenWindow.cha"]


def test_detabularize_transcripts_uses_template_header_and_indexes_duplicate_names(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    table_path = input_dir / "transcript_tables" / "transcript_tables.xlsx"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "custom_template_header.cha").write_text(
        "@Begin\n@Languages:\teng\n@Participants:\tPAR0 Participant\n@End\n",
        encoding="utf-8",
    )

    samples = pd.DataFrame(
        [
            {"sample_id": "S1", "input_order": 1, "study_id": "AC25", "test": "Pre"},
            {"sample_id": "S2", "input_order": 2, "study_id": "AC25", "test": "Pre"},
        ]
    )
    utterances = pd.DataFrame(
        [
            {"sample_id": "S1", "position": 1, "speaker": "PAR0", "utterance": "first", "comment": ""},
            {"sample_id": "S2", "position": 1, "speaker": "PAR0", "utterance": "second", "comment": ""},
        ]
    )
    _write_table(table_path, samples, utterances)

    detabularize_transcripts(input_dir=input_dir, output_dir=output_dir)

    assert (output_dir / "chat_files" / "AC25_Pre_1.cha").exists()
    assert (output_dir / "chat_files" / "AC25_Pre_2.cha").exists()
    assert "@End\n" == (output_dir / "chat_files" / "AC25_Pre_1.cha").read_text(encoding="utf-8")[-5:]

    derived = pd.read_excel(output_dir / "transcript_tables" / "transcript_tables.xlsx", sheet_name="samples")
    assert list(derived["derived_file"]) == ["AC25_Pre_1.cha", "AC25_Pre_2.cha"]


def test_detabularize_transcripts_matches_integer_and_float_sample_ids(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    table_path = input_dir / "transcript_tables" / "transcript_tables.xlsx"

    samples = pd.DataFrame(
        [{"sample_id": 1, "study_id": "TU", "test": "Pre", "narrative": "Story"}]
    )
    utterances = pd.DataFrame(
        [{"sample_id": 1.0, "speaker": "PAR0", "utterance": "matched", "comment": ""}]
    )
    _write_table(table_path, samples, utterances)

    detabularize_transcripts(input_dir=input_dir, output_dir=output_dir)

    chat_text = (output_dir / "chat_files" / "TU_Pre_Story.cha").read_text(encoding="utf-8")
    assert "*PAR0:\tmatched\n" in chat_text


def test_detabularize_transcripts_accepts_custom_sample_id_field(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    table_path = input_dir / "transcript_tables" / "transcript_tables.xlsx"

    samples = pd.DataFrame(
        [{"expanded_sample_id": "A-1", "study_id": "TU", "test": "Post"}]
    )
    utterances = pd.DataFrame(
        [
            {
                "expanded_sample_id": "A-1",
                "expanded_utterance_id": "A-1-U1",
                "speaker": "PAR0",
                "utterance": "custom id matched",
                "comment": "",
            }
        ]
    )
    _write_table(table_path, samples, utterances)

    written = detabularize_transcripts(
        input_dir=input_dir,
        output_dir=output_dir,
        sample_id_field="expanded_sample_id",
    )

    assert written == [str(output_dir / "chat_files" / "TU_Post.cha")]
    chat_text = (output_dir / "chat_files" / "TU_Post.cha").read_text(encoding="utf-8")
    assert "*PAR0:\tcustom id matched\n" in chat_text

    derived = pd.read_excel(
        output_dir / "transcript_tables" / "transcript_tables.xlsx",
        sheet_name="samples",
    )
    assert "expanded_sample_id" in derived.columns
    assert list(derived["derived_file"]) == ["TU_Post.cha"]


def test_detabularize_transcripts_requires_exact_transcript_table_filename(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    table_path = input_dir / "transcript_tables" / "study_transcript_tables.xlsx"

    samples = pd.DataFrame([{"sample_id": "S1"}])
    utterances = pd.DataFrame(
        [{"sample_id": "S1", "speaker": "PAR0", "utterance": "not discovered"}]
    )
    _write_table(table_path, samples, utterances)

    with pytest.raises(FileNotFoundError):
        detabularize_transcripts(input_dir=input_dir, output_dir=output_dir)


def test_detabularize_transcripts_errors_on_multiple_exact_tables(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    samples = pd.DataFrame([{"sample_id": "S1"}])
    utterances = pd.DataFrame(
        [{"sample_id": "S1", "speaker": "PAR0", "utterance": "ambiguous"}]
    )

    _write_table(input_dir / "site_a" / "transcript_tables.xlsx", samples, utterances)
    _write_table(output_dir / "site_b" / "transcript_tables.xlsx", samples, utterances)

    with pytest.raises(MultipleFilesFoundError):
        detabularize_transcripts(input_dir=input_dir, output_dir=output_dir)
