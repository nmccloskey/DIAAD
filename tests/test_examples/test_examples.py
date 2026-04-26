from __future__ import annotations

import pandas as pd

from diaad.examples import get_example_io_docs_path, iter_example_io_markdown_files
from diaad.examples.generate import generate_example_files
from diaad.examples.render_docs import render_example_docs


def test_generate_synthetic_project(tmp_path):
    project_dir = generate_example_files(tmp_path / "synthetic_project")

    assert (project_dir / "config" / "project.yaml").exists()
    assert len(list((project_dir / "input" / "chat").glob("*.cha"))) == 3

    workbook = (
        project_dir
        / "expected_outputs"
        / "transcripts_tabularize"
        / "transcript_table.xlsx"
    )
    assert workbook.exists()

    with pd.ExcelFile(workbook, engine="openpyxl") as xls:
        assert {"samples", "utterances"} <= set(xls.sheet_names)


def test_render_example_docs():
    paths = render_example_docs()

    assert any(path.name == "01_overview.md" for path in paths)
    assert any(path.name == "tabularize.md" for path in paths)
    assert (get_example_io_docs_path() / "transcripts" / "tabularize.md").exists()
    assert any(path.name == "tabularize.md" for path in iter_example_io_markdown_files())
