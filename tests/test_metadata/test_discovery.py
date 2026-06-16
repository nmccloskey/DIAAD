from __future__ import annotations

from pathlib import Path

import pytest

import diaad.metadata.discovery as discovery
from diaad.metadata.discovery import (
    MultipleFilesFoundError,
    find_files_by_extension,
    find_one_file_by_extension,
    find_one_matching_file,
    find_transcript_table,
)


def test_find_one_matching_file_uses_exact_filename(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    expected = root / "word_counting.xlsx"
    expected.touch()
    (root / "word_counting_by_sample.xlsx").touch()

    found = find_one_matching_file(
        directories=root,
        filename="word_counting.xlsx",
        label="word-count coding file",
    )

    assert found == expected


def test_find_one_matching_file_searches_recursively(tmp_path):
    root = tmp_path / "input"
    expected = root / "nested" / "tables" / "transcript_tables.xlsx"
    expected.parent.mkdir(parents=True)
    expected.touch()

    found = find_one_matching_file(
        directories=root,
        filename="transcript_tables.xlsx",
        label="transcript table file",
    )

    assert found == expected


def test_find_one_matching_file_raises_when_missing(tmp_path):
    root = tmp_path / "input"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="missing.xlsx"):
        find_one_matching_file(
            directories=root,
            filename="missing.xlsx",
            label="configured input file",
        )


def test_find_one_matching_file_raises_with_actionable_multiple_match_error(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    first = input_dir / "cu" / "cu_coding.xlsx"
    second = output_dir / "cu" / "cu_coding.xlsx"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()

    with pytest.raises(MultipleFilesFoundError) as exc_info:
        find_one_matching_file(
            directories=[input_dir, output_dir],
            filename="cu_coding.xlsx",
            label="CU coding file",
        )

    message = str(exc_info.value)
    assert "cu_coding.xlsx" in message
    assert "Searched directories" in message
    assert "Matched paths" in message
    assert "input" in message
    assert "output" in message
    assert "Please remove duplicates, rename files" in message


def test_find_one_file_by_extension_searches_recursively_and_skips_excel_temp_files(tmp_path):
    root = tmp_path / "input"
    expected = root / "nested" / "source.xlsx"
    temp_file = root / "~$source.xlsx"
    expected.parent.mkdir(parents=True)
    expected.touch()
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file.touch()

    found = find_one_file_by_extension(
        directories=root,
        search_ext=".xlsx",
        label="sample subset input workbook",
    )

    assert found == expected


def test_find_files_by_extension_recurses_keeps_duplicate_names_and_skips_temp_files(tmp_path):
    root = tmp_path / "input"
    first = root / "site_a" / "coding.xlsx"
    second = root / "site_b" / "coding.xlsx"
    temp_file = root / "site_c" / "~$coding.xlsx"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    temp_file.parent.mkdir(parents=True)
    first.touch()
    second.touch()
    temp_file.touch()

    found = find_files_by_extension(directories=root, search_ext="xlsx")

    assert found == [first, second]


def test_find_files_by_extension_delegates_to_psair_backend(monkeypatch, tmp_path):
    calls = []
    expected = tmp_path / "input" / "source.xlsx"

    def fake_find_matching_files(**kwargs):
        calls.append(kwargs)
        return [expected]

    monkeypatch.setattr(discovery, "psair_find_matching_files", fake_find_matching_files)

    found = discovery.find_files_by_extension(
        directories=tmp_path / "input",
        search_ext="xlsx",
    )

    assert found == [expected]
    assert calls == [
        {
            "directories": [Path(tmp_path / "input")],
            "search_base": "?",
            "search_ext": ".xlsx",
            "match_mode": "contains",
            "deduplicate": False,
            "ignore_excel_temp_files": True,
        }
    ]


def test_find_transcript_table_returns_none_when_optional_and_missing(tmp_path):
    root = tmp_path / "input"
    root.mkdir()

    found = find_transcript_table(directories=root, required=False)

    assert found is None


def test_find_transcript_table_raises_on_multiple_exact_matches(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    first = input_dir / "transcript_tables.xlsx"
    second = output_dir / "transcript_tables.xlsx"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.touch()
    second.touch()

    with pytest.raises(MultipleFilesFoundError):
        find_transcript_table(directories=[input_dir, output_dir])
