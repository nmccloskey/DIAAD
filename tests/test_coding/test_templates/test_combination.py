from __future__ import annotations

import pandas as pd
import pytest

from diaad.coding.templates.combination import (
    COMBINED_TEMPLATE_FILENAME,
    make_combined_template_file,
)
from diaad.coding.templates.utils import TEMPLATE_SUBDIR


def _write_workbook(path, sheets):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def _read_sheet(path, sheet_name):
    return pd.read_excel(path, sheet_name=sheet_name)


def test_make_combined_template_file_recurses_and_preserves_relative_sources(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = input_dir / "output"

    _write_workbook(
        input_dir / "site_a" / "coding.xlsx",
        {
            "samples": pd.DataFrame({"sample_id": ["A1", "A2"], "code": [1, 2]}),
            "utterances": pd.DataFrame({"utterance_id": [], "value": []}),
        },
    )
    _write_workbook(
        input_dir / "site_b" / "coding.xlsx",
        {
            "utterances": pd.DataFrame({"value": ["x"], "utterance_id": ["B1_u1"]}),
            "samples": pd.DataFrame({"code": [3], "sample_id": ["B1"]}),
        },
    )
    _write_workbook(
        output_dir / "old_output.xlsx",
        {"wrong": pd.DataFrame({"wrong": [1]})},
    )
    (input_dir / "~$ignored.xlsx").touch()

    outpath = make_combined_template_file(
        input_dir=input_dir,
        output_dir=output_dir,
    )

    assert outpath == output_dir / TEMPLATE_SUBDIR / COMBINED_TEMPLATE_FILENAME

    samples = _read_sheet(outpath, "samples")
    utterances = _read_sheet(outpath, "utterances")
    metadata = _read_sheet(outpath, "metadata")

    assert samples.to_dict("records") == [
        {
            "combined_id": 1,
            "source_file": "site_a/coding.xlsx",
            "sample_id": "A1",
            "code": 1,
        },
        {
            "combined_id": 2,
            "source_file": "site_a/coding.xlsx",
            "sample_id": "A2",
            "code": 2,
        },
        {
            "combined_id": 3,
            "source_file": "site_b/coding.xlsx",
            "sample_id": "B1",
            "code": 3,
        },
    ]
    assert utterances.to_dict("records") == [
        {
            "combined_id": 1,
            "source_file": "site_b/coding.xlsx",
            "utterance_id": "B1_u1",
            "value": "x",
        },
    ]
    assert metadata.to_dict("records") == [
        {
            "source_file": "site_a/coding.xlsx",
            "order": 1,
            "sheet": "samples",
            "num_rows": 2,
        },
        {
            "source_file": "site_a/coding.xlsx",
            "order": 1,
            "sheet": "utterances",
            "num_rows": 0,
        },
        {
            "source_file": "site_b/coding.xlsx",
            "order": 2,
            "sheet": "samples",
            "num_rows": 1,
        },
        {
            "source_file": "site_b/coding.xlsx",
            "order": 2,
            "sheet": "utterances",
            "num_rows": 1,
        },
    ]


def test_make_combined_template_file_errors_on_sheet_mismatch(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_workbook(input_dir / "a.xlsx", {"samples": pd.DataFrame({"id": [1]})})
    _write_workbook(input_dir / "b.xlsx", {"other": pd.DataFrame({"id": [2]})})

    with pytest.raises(ValueError, match="sheet names"):
        make_combined_template_file(input_dir=input_dir, output_dir=output_dir)


def test_make_combined_template_file_errors_on_column_mismatch(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_workbook(input_dir / "a.xlsx", {"samples": pd.DataFrame({"id": [1]})})
    _write_workbook(input_dir / "b.xlsx", {"samples": pd.DataFrame({"other": [2]})})

    with pytest.raises(ValueError, match="columns"):
        make_combined_template_file(input_dir=input_dir, output_dir=output_dir)


def test_make_combined_template_file_rejects_reserved_output_columns(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_workbook(
        input_dir / "a.xlsx",
        {"samples": pd.DataFrame({"source_file": ["already-there"]})},
    )

    with pytest.raises(ValueError, match="reserved output column"):
        make_combined_template_file(input_dir=input_dir, output_dir=output_dir)


def test_make_combined_template_file_rejects_input_metadata_sheet(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_workbook(
        input_dir / "a.xlsx",
        {"metadata": pd.DataFrame({"source_file": ["a.xlsx"]})},
    )

    with pytest.raises(ValueError, match="reserved"):
        make_combined_template_file(input_dir=input_dir, output_dir=output_dir)
