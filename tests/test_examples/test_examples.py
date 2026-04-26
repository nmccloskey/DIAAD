from __future__ import annotations

import pandas as pd

from diaad.examples import get_example_io_docs_path, iter_example_io_markdown_files
from diaad.examples.generate import generate_example_files
from diaad.examples.render_docs import render_example_docs


def test_generate_synthetic_project(tmp_path):
    project_dir = generate_example_files(tmp_path / "synthetic_project")

    assert (project_dir / "config" / "project.yaml").exists()
    assert (project_dir / "config" / "advanced.yaml").exists()
    assert not (project_dir / "config" / "advanced_project.yaml").exists()
    assert len(list((project_dir / "input" / "chat").glob("*.cha"))) == 3
    assert len(list((project_dir / "input" / "chat" / "reliability").glob("*.cha"))) == 2
    assert (
        project_dir
        / "input"
        / "transcription_reliability_selection"
        / "transcription_reliability_samples.xlsx"
    ).exists()

    workbook = (
        project_dir
        / "expected_outputs"
        / "transcripts_tabularize"
        / "transcript_table.xlsx"
    )
    assert workbook.exists()

    with pd.ExcelFile(workbook, engine="openpyxl") as xls:
        assert {"samples", "utterances"} <= set(xls.sheet_names)

    assert (
        project_dir
        / "expected_outputs"
        / "transcripts_select"
        / "transcription_reliability_samples.xlsx"
    ).exists()
    assert (
        project_dir
        / "expected_outputs"
        / "transcripts_evaluate"
        / "transcription_reliability_evaluation.xlsx"
    ).exists()
    assert (
        project_dir
        / "expected_outputs"
        / "transcripts_reselect"
        / "reselected_transcription_reliability"
        / "reselected_transcription_reliability_samples.xlsx"
    ).exists()


def test_render_example_docs():
    paths = render_example_docs()

    assert any(path.name == "01_overview.md" for path in paths)
    assert any(path.name == "tabularize.md" for path in paths)
    assert any(path.name == "select.md" for path in paths)
    assert any(path.name == "evaluate.md" for path in paths)
    assert any(path.name == "reselect.md" for path in paths)
    assert (get_example_io_docs_path() / "transcripts" / "tabularize.md").exists()
    assert any(path.name == "tabularize.md" for path in iter_example_io_markdown_files())
