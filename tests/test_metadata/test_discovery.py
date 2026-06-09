from __future__ import annotations

import pytest

from diaad.metadata.discovery import (
    MultipleFilesFoundError,
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
