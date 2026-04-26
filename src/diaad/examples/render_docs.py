from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from diaad.examples.generate import EXPECTED_WORKBOOK, _scratch_dir, generate_example_files


DOC_PACKAGE = "diaad.examples"
DOC_ROOT = ("assets", "rendered_docs", "example_io")
SPEC_ROOT = ("assets", "spec")


def _asset_path(*parts: str):
    path = resources.files(DOC_PACKAGE)
    for part in parts:
        path = path.joinpath(part)
    return path


def _read_yaml_asset(*parts: str) -> dict[str, Any]:
    with _asset_path(*parts).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML asset {'/'.join(parts)} must contain a mapping.")
    return data


def _write_doc(*parts: str, text: str) -> Path:
    path = Path(_asset_path(*DOC_ROOT, *parts))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _fenced(text: str, language: str = "") -> str:
    return f"```{language}\n{text.rstrip()}\n```"


def _preview_yaml(data: dict[str, Any], keys: list[str]) -> str:
    subset = {key: data[key] for key in keys if key in data}
    return yaml.safe_dump(subset, sort_keys=False, allow_unicode=False).rstrip()


def _markdown_table(df: pd.DataFrame, *, max_rows: int = 8) -> str:
    preview = df.head(max_rows).fillna("")
    headers = [str(col) for col in preview.columns]
    rows = [[str(value) for value in row] for row in preview.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _tree(command: str = "all") -> str:
    if command == "tabularize":
        expected = """      transcripts_tabularize/
        transcript_table.xlsx"""
    elif command == "select":
        expected = """      transcripts_select/
        P1_picnic_pre_reliability.cha
        P2_picnic_pre_reliability.cha
        transcription_reliability_samples.xlsx"""
    elif command == "evaluate":
        expected = """      transcripts_evaluate/
        transcription_reliability_evaluation.xlsx
        transcription_reliability_report.txt
        global_alignments/"""
    elif command == "reselect":
        expected = """      transcripts_reselect/
        reselected_transcription_reliability/
          reselected_transcription_reliability_samples.xlsx"""
    else:
        expected = """      transcripts_tabularize/
        transcript_table.xlsx
      transcripts_select/
        transcription_reliability_samples.xlsx
      transcripts_evaluate/
        transcription_reliability_evaluation.xlsx
        transcription_reliability_report.txt
      transcripts_reselect/
        reselected_transcription_reliability/
          reselected_transcription_reliability_samples.xlsx"""

    return """example_files/
  synthetic_project/
    README.md
    config/
      project.yaml
      advanced.yaml
    input/
      chat/
        P1_picnic_pre.cha
        P2_picnic_pre.cha
        P1_picnic_post.cha
        reliability/
          P1_picnic_pre.cha
          P2_picnic_pre.cha
      transcription_reliability_selection/
        transcription_reliability_samples.xlsx
    expected_outputs/
""" + expected


def _overview_doc() -> str:
    return """# DIAAD Example I/O Manual

The example I/O manual shows small, runnable DIAAD workflows alongside their inputs and outputs.

Runnable files are generated locally under `example_files/`, so the repository and installed package do not carry generated workbooks or CHAT copies. The manual-style markdown is packaged with DIAAD under `diaad.examples.assets.rendered_docs.example_io` for use by the webapp, manual renderer, or other documentation tools.

Synthetic data are defined in packaged YAML specs. Some markdown pages are authored directly, and others include tables, directory trees, and snippets rendered from those specs or from generated example files.

All example data are synthetic. They are not human-subjects data, participant records, clinical documentation, or de-identified real transcripts.
"""


def _read_specs() -> dict[str, dict[str, Any]]:
    return {
        "project_config": _read_yaml_asset(*SPEC_ROOT, "configs", "project.yaml"),
        "advanced_config": _read_yaml_asset(*SPEC_ROOT, "configs", "advanced.yaml"),
        "chat_files": _read_yaml_asset(*SPEC_ROOT, "transcripts", "chat_files.yaml"),
        "reliability_chat_files": _read_yaml_asset(
            *SPEC_ROOT,
            "transcripts",
            "reliability_chat_files.yaml",
        ),
    }


def _workbook_sheet_tables(path: Path, sheet_names: list[str]) -> str:
    sections = []
    for sheet_name in sheet_names:
        df = pd.read_excel(path, sheet_name=sheet_name)
        sections.append(f"### Sheet: {sheet_name}\n\n{_markdown_table(df)}")
    return "\n\n".join(sections)


def _tabularize_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    chat = specs["chat_files"]["chat_files"][0]
    workbook = project_dir / "expected_outputs" / "transcripts_tabularize" / EXPECTED_WORKBOOK

    project_snippet = _preview_yaml(
        specs["project_config"],
        ["input_dir", "output_dir", "random_seed", "shuffle_samples", "metadata_fields"],
    )
    advanced_snippet = _preview_yaml(
        specs["advanced_config"],
        ["reliability_tag", "reliability_dirname", "metadata_source", "id_cols"],
    )
    chat_excerpt = "\n".join(chat["content"].splitlines()[:12])

    return f"""# Transcript Tabularization Example

This example demonstrates how `diaad transcripts tabularize` converts tiny synthetic CHAT files into sample- and utterance-level workbook sheets.

## Command

{_fenced("diaad transcripts tabularize --config config", "bash")}

## Project Files

{_fenced(_tree("tabularize"))}

## Basic Config

{_fenced(project_snippet, "yaml")}

## Advanced Config

{_fenced(advanced_snippet, "yaml")}

## Input Snippet

`input/chat/{chat["filename"]}`

{_fenced(chat_excerpt, "text")}

## Output Preview

`expected_outputs/transcripts_tabularize/transcript_table.xlsx`

{_workbook_sheet_tables(workbook, ["samples", "utterances"])}

## Notes

These files are fully synthetic and regenerated from packaged YAML specs. The markdown preview shows only selected rows and snippets; the generated workbook contains the complete example output.
"""


def _select_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    workbook = (
        project_dir
        / "expected_outputs"
        / "transcripts_select"
        / "transcription_reliability_samples.xlsx"
    )
    project_snippet = _preview_yaml(
        specs["project_config"],
        ["input_dir", "output_dir", "random_seed", "reliability_fraction", "metadata_fields"],
    )
    chat_excerpt = "\n".join(specs["chat_files"]["chat_files"][1]["content"].splitlines()[:11])

    return f"""# Transcription Reliability Selection Example

This example demonstrates how `diaad transcripts select` selects synthetic CHAT files for secondary transcription and writes blank reliability templates.

## Command

{_fenced("diaad transcripts select --config config", "bash")}

## Project Files

{_fenced(_tree("select"))}

## Basic Config

{_fenced(project_snippet, "yaml")}

## Input Snippet

The command uses the synthetic CHAT files in `input/chat/`.

{_fenced(chat_excerpt, "text")}

## Output Preview

`expected_outputs/transcripts_select/transcription_reliability_samples.xlsx`

{_workbook_sheet_tables(workbook, ["reliability_selection", "all_transcripts"])}

## Notes

The blank reliability `.cha` files contain CHAT headers only. They are generated artifacts for transcription workflow setup, not completed reliability transcripts.
"""


def _evaluate_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    workbook = (
        project_dir
        / "expected_outputs"
        / "transcripts_evaluate"
        / "transcription_reliability_evaluation.xlsx"
    )
    report = (
        project_dir
        / "expected_outputs"
        / "transcripts_evaluate"
        / "transcription_reliability_report.txt"
    )
    reliability_chat = specs["reliability_chat_files"]["reliability_chat_files"][0]
    chat_excerpt = "\n".join(reliability_chat["content"].splitlines()[:12])
    report_excerpt = "\n".join(report.read_text(encoding="utf-8").splitlines()[:10])
    report_excerpt = report_excerpt.replace("â€¢", "-")
    report_excerpt = report_excerpt.replace("•", "-")

    return f"""# Transcription Reliability Evaluation Example

This example demonstrates how `diaad transcripts evaluate` compares original CHAT files with synthetic reliability transcriptions.

## Command

{_fenced("diaad transcripts evaluate --config config", "bash")}

## Project Files

{_fenced(_tree("evaluate"))}

## Advanced Config

{_fenced(_preview_yaml(specs["advanced_config"], ["reliability_tag", "reliability_dirname"]), "yaml")}

## Input Snippet

`input/chat/reliability/{reliability_chat["filename"]}`

{_fenced(chat_excerpt, "text")}

## Output Preview

`expected_outputs/transcripts_evaluate/transcription_reliability_evaluation.xlsx`

{_markdown_table(pd.read_excel(workbook))}

`expected_outputs/transcripts_evaluate/transcription_reliability_report.txt`

{_fenced(report_excerpt, "text")}

## Notes

Reliability transcripts are believable synthetic variants of the original examples. DIAAD matches originals and reliability files by configured metadata fields.
"""


def _reselect_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    workbook = (
        project_dir
        / "expected_outputs"
        / "transcripts_reselect"
        / "reselected_transcription_reliability"
        / "reselected_transcription_reliability_samples.xlsx"
    )

    return f"""# Transcription Reliability Reselection Example

This example demonstrates how `diaad transcripts reselect` chooses replacement reliability samples after an earlier selection has already been used.

## Command

{_fenced("diaad transcripts reselect --config config", "bash")}

## Project Files

{_fenced(_tree("reselect"))}

## Basic Config

{_fenced(_preview_yaml(specs["project_config"], ["input_dir", "output_dir", "reliability_fraction"]), "yaml")}

## Input Snippet

The reselection command reads the prior selection workbook:

`input/transcription_reliability_selection/transcription_reliability_samples.xlsx`

## Output Preview

`expected_outputs/transcripts_reselect/reselected_transcription_reliability/reselected_transcription_reliability_samples.xlsx`

{_workbook_sheet_tables(workbook, ["reselected_reliability"])}

## Notes

The synthetic project has three samples. Because two are already selected in the first reliability pass, only one unused candidate remains for reselection.
"""


def render_example_docs() -> list[Path]:
    """Create or update packaged example I/O markdown assets."""
    specs = _read_specs()
    with _scratch_dir(Path.cwd()) as tmpdir:
        project_dir = generate_example_files(tmpdir / "synthetic_project", force=True)
        tabularize_doc = _tabularize_doc(project_dir, specs)
        select_doc = _select_doc(project_dir, specs)
        evaluate_doc = _evaluate_doc(project_dir, specs)
        reselect_doc = _reselect_doc(project_dir, specs)

    return [
        _write_doc("01_overview.md", text=_overview_doc()),
        _write_doc("transcripts", "tabularize.md", text=tabularize_doc),
        _write_doc("transcripts", "select.md", text=select_doc),
        _write_doc("transcripts", "evaluate.md", text=evaluate_doc),
        _write_doc("transcripts", "reselect.md", text=reselect_doc),
    ]
