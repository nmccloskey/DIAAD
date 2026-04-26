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


def _tree() -> str:
    return """example_files/
  synthetic_project/
    README.md
    config/
      project.yaml
      advanced_project.yaml
      advanced.yaml
    input/
      chat/
        P1_picnic_pre.cha
        P2_picnic_pre.cha
        P1_picnic_post.cha
    expected_outputs/
      transcripts_tabularize/
        transcript_table.xlsx"""


def _overview_doc() -> str:
    return """# DIAAD Example I/O Manual

The example I/O manual shows small, runnable DIAAD workflows alongside their inputs and outputs.

Runnable files are generated locally under `example_files/`, so the repository and installed package do not carry generated workbooks or CHAT copies. The manual-style markdown is packaged with DIAAD under `diaad.examples.assets.rendered_docs.example_io` for use by the webapp, manual renderer, or other documentation tools.

Synthetic data are defined in packaged YAML specs. Some markdown pages are authored directly, and others include tables, directory trees, and snippets rendered from those specs or from generated example files.

All example data are synthetic. They are not human-subjects data, participant records, clinical documentation, or de-identified real transcripts.
"""


def _tabularize_doc() -> str:
    specs = {
        "project_config": _read_yaml_asset(*SPEC_ROOT, "configs", "project.yaml"),
        "advanced_config": _read_yaml_asset(*SPEC_ROOT, "configs", "advanced.yaml"),
        "chat_files": _read_yaml_asset(*SPEC_ROOT, "transcripts", "chat_files.yaml"),
    }
    chat = specs["chat_files"]["chat_files"][0]

    with _scratch_dir(Path.cwd()) as tmpdir:
        project_dir = generate_example_files(tmpdir / "synthetic_project", force=True)
        workbook = (
            project_dir
            / "expected_outputs"
            / "transcripts_tabularize"
            / EXPECTED_WORKBOOK
        )
        samples = pd.read_excel(workbook, sheet_name="samples")
        utterances = pd.read_excel(workbook, sheet_name="utterances")

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

{_fenced(_tree())}

## Basic Config

{_fenced(project_snippet, "yaml")}

## Advanced Config

{_fenced(advanced_snippet, "yaml")}

## Input Snippet

`input/chat/{chat["filename"]}`

{_fenced(chat_excerpt, "text")}

## Output Preview

`expected_outputs/transcripts_tabularize/transcript_table.xlsx`

### Sheet: samples

{_markdown_table(samples)}

### Sheet: utterances

{_markdown_table(utterances)}

## Notes

These files are fully synthetic and regenerated from packaged YAML specs. The markdown preview shows only selected rows and snippets; the generated workbook contains the complete example output.
"""


def render_example_docs() -> list[Path]:
    """Create or update packaged example I/O markdown assets."""
    return [
        _write_doc("01_overview.md", text=_overview_doc()),
        _write_doc("transcripts", "tabularize.md", text=_tabularize_doc()),
    ]
