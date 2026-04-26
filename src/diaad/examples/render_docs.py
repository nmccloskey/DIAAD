from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from diaad.examples.generate import (
    CU_OUTPUT_DIRS,
    CUS_MODULE_DIR,
    EXPECTED_WORKBOOK,
    TEMPLATE_OUTPUT_DIRS,
    TEMPLATES_MODULE_DIR,
    TRANSCRIPTS_MODULE_DIR,
    _scratch_dir,
    generate_example_files,
)


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


def _project_config_snippet(
    specs: dict[str, dict[str, Any]],
    keys: list[str],
) -> str:
    data = specs["project_config"].copy()
    if "input_dir" in keys:
        data["input_dir"] = "diaad_data/input"
    if "output_dir" in keys:
        data["output_dir"] = "diaad_data/output"
    return _preview_yaml(data, keys)


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


RUN_DIR = "diaad_YYMMDD_HHMM"


def _logs_tree(indent: str = "        ") -> str:
    return f"""{indent}logs/
{indent}  diaad_YYMMDD_HHMM.log
{indent}  diaad_YYMMDD_HHMM_metadata.json"""


def _output_tree(contents: str) -> str:
    return f"""    output/
      {RUN_DIR}/
{contents.rstrip()}
{_logs_tree()}"""


def _project_tree(command: str = "all") -> str:
    if command == "tabularize":
        input_files = """      chat/
        P1_picnic_pre.cha
        P2_picnic_pre.cha
        P1_picnic_post.cha"""
        outputs = """        transcript_tables/
          transcript_tables.xlsx"""
    elif command == "select":
        input_files = """      chat/
        P1_picnic_pre.cha
        P2_picnic_pre.cha
        P1_picnic_post.cha"""
        outputs = """        transcription_reliability_selection/
          P1_picnic_pre_reliability.cha
          P2_picnic_pre_reliability.cha
          transcription_reliability_samples.xlsx"""
    elif command == "evaluate":
        input_files = """      chat/
        P1_picnic_pre.cha
        P2_picnic_pre.cha
        P1_picnic_post.cha
        reliability/
          P1_picnic_pre.cha
          P2_picnic_pre.cha"""
        outputs = """        transcription_reliability_evaluation/
          transcription_reliability_evaluation.xlsx
          transcription_reliability_report.txt
          global_alignments/"""
    elif command == "reselect":
        input_files = """      transcription_reliability_selection/
        transcription_reliability_samples.xlsx"""
        outputs = """        reselected_transcription_reliability/
          reselected_transcription_reliability_samples.xlsx"""
    elif command in {"templates_utterances", "templates_samples", "templates_times"}:
        input_files = """      transcript_tables/
        transcript_tables.xlsx"""
        if command == "templates_utterances":
            outputs = """        coding_templates/
          utterance_coding_template.xlsx
          utterance_reliability_template.xlsx
          utterance_template_codebook.xlsx"""
        elif command == "templates_samples":
            outputs = """        coding_templates/
          sample_coding_template.xlsx
          sample_reliability_template.xlsx
          sample_template_codebook.xlsx"""
        else:
            outputs = """        coding_templates/
          speaking_times.xlsx"""
    elif command == "cus_files":
        input_files = """      transcript_tables/
        transcript_tables.xlsx"""
        outputs = """        cu_coding/
          cu_coding.xlsx
          cu_reliability_coding.xlsx
          cu_blind_codebook.xlsx"""
    elif command == "cus_evaluate":
        input_files = """      cu_coding/
        cu_coding.xlsx
        cu_reliability_coding.xlsx"""
        outputs = """        cu_reliability/
          cu_reliability_coding_by_utterance.xlsx
          cu_reliability_coding_by_sample.xlsx
          cu_reliability_coding_report.txt"""
    elif command == "cus_reselect":
        input_files = """      cu_coding/
        cu_coding.xlsx
        cu_reliability_coding.xlsx"""
        outputs = """        reselected_cu_coding_reliability/
          reselected_cu_reliability_coding.xlsx"""
    elif command == "cus_analyze":
        input_files = """      cu_coding/
        cu_coding.xlsx
        cu_blind_codebook.xlsx"""
        outputs = """        cu_coding_analysis/
          cu_coding_by_utterance.xlsx
          cu_coding_by_sample_long.xlsx
          cu_coding_by_sample.xlsx"""
    elif command == "cus_rates":
        input_files = """      cu_coding_analysis/
        cu_coding_by_sample_long.xlsx
      speaking_times/
        speaking_times.xlsx"""
        outputs = """        cu_coding_analysis/
          cu_coding_rates.xlsx"""
    else:
        input_files = """      chat/
        P1_picnic_pre.cha
        P2_picnic_pre.cha
        P1_picnic_post.cha"""
        outputs = """        transcript_tables/
          transcript_tables.xlsx"""

    return f"""your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
{input_files}
{_output_tree(outputs)}"""


def _example_files_tree() -> str:
    return (
        """example_files/
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
      cu_coding/
        cu_coding.xlsx
        cu_reliability_coding.xlsx
        cu_blind_codebook.xlsx
      cu_coding_analysis/
        cu_coding_by_sample_long.xlsx
      speaking_times/
        speaking_times.xlsx
    expected_outputs/
      transcripts_module/
        transcripts_tabularize/
          transcript_table.xlsx
        transcripts_select/
          transcription_reliability_samples.xlsx
        transcripts_evaluate/
          transcription_reliability_evaluation.xlsx
          transcription_reliability_report.txt
        transcripts_reselect/
          reselected_transcription_reliability/
            reselected_transcription_reliability_samples.xlsx
      templates_module/
        templates_utterances/
          utterance_coding_template.xlsx
          utterance_reliability_template.xlsx
          utterance_template_codebook.xlsx
        templates_samples/
          sample_coding_template.xlsx
          sample_reliability_template.xlsx
          sample_template_codebook.xlsx
        templates_times/
          speaking_times.xlsx"""
        + """
      cus_module/
        cus_files/
          cu_coding.xlsx
          cu_reliability_coding.xlsx
          cu_blind_codebook.xlsx
        cus_evaluate/
          cu_reliability_coding_by_utterance.xlsx
          cu_reliability_coding_by_sample.xlsx
          cu_reliability_coding_report.txt
        cus_reselect/
          reselected_cu_reliability_coding.xlsx
        cus_analyze/
          cu_coding_by_utterance.xlsx
          cu_coding_by_sample_long.xlsx
          cu_coding_by_sample.xlsx
        cus_rates/
          cu_coding_rates.xlsx"""
    )


def _overview_doc() -> str:
    return f"""# DIAAD Example I/O Manual

The example I/O manual shows small, runnable DIAAD workflows alongside their inputs and outputs.

Runnable files are generated locally under `example_files/`, so the repository and installed package do not carry generated workbooks or CHAT copies. The manual-style markdown is packaged with DIAAD under `diaad.examples.assets.rendered_docs.example_io` for use by the webapp, manual renderer, or other documentation tools.

Command pages show minimal user project structures: the config files a user needs, the required inputs for that one command, and the output files created in a timestamped DIAAD run directory. The local generated example project is a fuller teaching fixture because it contains inputs and expected outputs for several commands at once.

Generated example projects may include a `README.md` for navigation, but a README is not required in a user's DIAAD project.

Synthetic data are defined in packaged YAML specs. Some markdown pages are authored directly, and others include tables, directory trees, and snippets rendered from those specs or from generated example files.

All example data are synthetic. They are not human-subjects data, participant records, clinical documentation, or de-identified real transcripts.

## Generated Example Files

{_fenced(_example_files_tree())}
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


def _all_workbook_sheet_tables(path: Path) -> str:
    with pd.ExcelFile(path, engine="openpyxl") as xls:
        return _workbook_sheet_tables(path, xls.sheet_names)


def _tabularize_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    chat = specs["chat_files"]["chat_files"][0]
    workbook = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / "transcripts_tabularize"
        / EXPECTED_WORKBOOK
    )

    project_snippet = _project_config_snippet(
        specs,
        ["input_dir", "output_dir", "random_seed", "shuffle_samples", "metadata_fields"],
    )
    chat_excerpt = "\n".join(chat["content"].splitlines()[:12])

    return f"""# Transcript Tabularization Example

This example demonstrates how `diaad transcripts tabularize` converts tiny synthetic CHAT files into sample- and utterance-level workbook sheets.

## Command

{_fenced("diaad transcripts tabularize --config config", "bash")}

## Project Files

{_fenced(_project_tree("tabularize"))}

## Basic Config

{_fenced(project_snippet, "yaml")}

## Input Snippet

`diaad_data/input/chat/{chat["filename"]}`

{_fenced(chat_excerpt, "text")}

## Output Preview

`expected_outputs/transcripts_module/transcripts_tabularize/transcript_table.xlsx`

{_workbook_sheet_tables(workbook, ["samples", "utterances"])}

## Notes

These files are fully synthetic and regenerated from packaged YAML specs. The markdown preview shows only selected rows and snippets; the generated workbook contains the complete example output.
"""


def _select_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    workbook = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / "transcripts_select"
        / "transcription_reliability_samples.xlsx"
    )
    project_snippet = _project_config_snippet(
        specs,
        ["input_dir", "output_dir", "random_seed", "reliability_fraction", "metadata_fields"],
    )
    chat_excerpt = "\n".join(specs["chat_files"]["chat_files"][1]["content"].splitlines()[:11])

    return f"""# Transcription Reliability Selection Example

This example demonstrates how `diaad transcripts select` selects synthetic CHAT files for secondary transcription and writes blank reliability templates.

## Command

{_fenced("diaad transcripts select --config config", "bash")}

## Project Files

{_fenced(_project_tree("select"))}

## Basic Config

{_fenced(project_snippet, "yaml")}

## Input Snippet

The command uses the synthetic CHAT files in `diaad_data/input/chat/`.

{_fenced(chat_excerpt, "text")}

## Output Preview

`expected_outputs/transcripts_module/transcripts_select/transcription_reliability_samples.xlsx`

{_workbook_sheet_tables(workbook, ["reliability_selection", "all_transcripts"])}

## Notes

The blank reliability `.cha` files contain CHAT headers only. They are generated artifacts for transcription workflow setup, not completed reliability transcripts.
"""


def _evaluate_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    workbook = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / "transcripts_evaluate"
        / "transcription_reliability_evaluation.xlsx"
    )
    report = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
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

{_fenced(_project_tree("evaluate"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir", "metadata_fields"]), "yaml")}

## Advanced Config

{_fenced(_preview_yaml(specs["advanced_config"], ["reliability_tag", "reliability_dirname"]), "yaml")}

## Input Snippet

`diaad_data/input/chat/reliability/{reliability_chat["filename"]}`

{_fenced(chat_excerpt, "text")}

## Output Preview

`expected_outputs/transcripts_module/transcripts_evaluate/transcription_reliability_evaluation.xlsx`

{_markdown_table(pd.read_excel(workbook))}

`expected_outputs/transcripts_module/transcripts_evaluate/transcription_reliability_report.txt`

{_fenced(report_excerpt, "text")}

## Notes

Reliability transcripts are believable synthetic variants of the original examples. DIAAD matches originals and reliability files by configured metadata fields.
"""


def _reselect_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    workbook = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / "transcripts_reselect"
        / "reselected_transcription_reliability"
        / "reselected_transcription_reliability_samples.xlsx"
    )

    return f"""# Transcription Reliability Reselection Example

This example demonstrates how `diaad transcripts reselect` chooses replacement reliability samples after an earlier selection has already been used.

## Command

{_fenced("diaad transcripts reselect --config config", "bash")}

## Project Files

{_fenced(_project_tree("reselect"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir", "reliability_fraction"]), "yaml")}

## Input Snippet

The reselection command reads the prior selection workbook:

`diaad_data/input/transcription_reliability_selection/transcription_reliability_samples.xlsx`

## Output Preview

`expected_outputs/transcripts_module/transcripts_reselect/reselected_transcription_reliability/reselected_transcription_reliability_samples.xlsx`

{_workbook_sheet_tables(workbook, ["reselected_reliability"])}

## Notes

The synthetic project has three samples. Because two are already selected in the first reliability pass, only one unused candidate remains for reselection.
"""


def _template_config_snippet(specs: dict[str, dict[str, Any]]) -> str:
    return _project_config_snippet(
        specs,
        [
            "input_dir",
            "output_dir",
            "reliability_fraction",
            "num_bins",
            "num_coders",
            "stimulus_field",
        ],
    )


def _template_advanced_snippet(specs: dict[str, dict[str, Any]]) -> str:
    return _preview_yaml(
        specs["advanced_config"],
        ["metadata_source", "coding_blind_cols", "id_cols"],
    )


def _utterance_templates_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    template_dir = (
        project_dir
        / "expected_outputs"
        / TEMPLATES_MODULE_DIR
        / TEMPLATE_OUTPUT_DIRS["utterances"]
    )
    primary = template_dir / "utterance_coding_template.xlsx"
    reliability = template_dir / "utterance_reliability_template.xlsx"
    codebook = template_dir / "utterance_template_codebook.xlsx"

    return f"""# Utterance Template Example

This example demonstrates how `diaad templates utterances` creates blank utterance-level coding workbooks from transcript tables.

## Command

{_fenced("diaad templates utterances --config config", "bash")}

## Project Files

{_fenced(_project_tree("templates_utterances"))}

## Basic Config

{_fenced(_template_config_snippet(specs), "yaml")}

## Advanced Config

{_fenced(_template_advanced_snippet(specs), "yaml")}

## Input Snippet

The command uses `diaad_data/input/transcript_tables/transcript_tables.xlsx`. The preview below is from the generated utterance coding template.

## Output Preview

`expected_outputs/templates_module/templates_utterances/utterance_coding_template.xlsx`

{_all_workbook_sheet_tables(primary)}

`expected_outputs/templates_module/templates_utterances/utterance_reliability_template.xlsx`

{_all_workbook_sheet_tables(reliability)}

`expected_outputs/templates_module/templates_utterances/utterance_template_codebook.xlsx`

{_all_workbook_sheet_tables(codebook)}

## Notes

The primary and reliability workbooks are synthetic blank coding materials. The codebook maps blinded sample identifiers back to internal sample IDs.
"""


def _sample_templates_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    template_dir = (
        project_dir
        / "expected_outputs"
        / TEMPLATES_MODULE_DIR
        / TEMPLATE_OUTPUT_DIRS["samples"]
    )
    primary = template_dir / "sample_coding_template.xlsx"
    reliability = template_dir / "sample_reliability_template.xlsx"
    codebook = template_dir / "sample_template_codebook.xlsx"

    return f"""# Sample Template Example

This example demonstrates how `diaad templates samples` creates blank sample-level coding workbooks with bins, coder assignment, and reliability rows.

## Command

{_fenced("diaad templates samples --config config", "bash")}

## Project Files

{_fenced(_project_tree("templates_samples"))}

## Basic Config

{_fenced(_template_config_snippet(specs), "yaml")}

## Advanced Config

{_fenced(_template_advanced_snippet(specs), "yaml")}

## Output Preview

`expected_outputs/templates_module/templates_samples/sample_coding_template.xlsx`

{_all_workbook_sheet_tables(primary)}

`expected_outputs/templates_module/templates_samples/sample_reliability_template.xlsx`

{_all_workbook_sheet_tables(reliability)}

`expected_outputs/templates_module/templates_samples/sample_template_codebook.xlsx`

{_all_workbook_sheet_tables(codebook)}

## Notes

The example uses two bins and two coders so the assignment and reliability-subset behavior is visible in a tiny workbook.
"""


def _time_templates_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    workbook = (
        project_dir
        / "expected_outputs"
        / TEMPLATES_MODULE_DIR
        / TEMPLATE_OUTPUT_DIRS["times"]
        / "speaking_times.xlsx"
    )

    return f"""# Speaking-Time Template Example

This example demonstrates how `diaad templates times` creates a blank sample-level speaking-time workbook.

## Command

{_fenced("diaad templates times --config config", "bash")}

## Project Files

{_fenced(_project_tree("templates_times"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Output Preview

`expected_outputs/templates_module/templates_times/speaking_times.xlsx`

{_all_workbook_sheet_tables(workbook)}

## Notes

The `speaking_time` column is intentionally blank. It is a template for project-specific duration values used later by rate calculations.
"""


def _cu_config_snippet(specs: dict[str, dict[str, Any]]) -> str:
    return _project_config_snippet(
        specs,
        [
            "input_dir",
            "output_dir",
            "reliability_fraction",
            "num_coders",
            "stimulus_field",
            "exclude_participants",
        ],
    )


def _cu_advanced_snippet(specs: dict[str, dict[str, Any]]) -> str:
    return _preview_yaml(
        specs["advanced_config"],
        ["cu_paradigms", "metadata_source", "coding_blind_cols", "id_cols"],
    )


def _cu_files_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / CUS_MODULE_DIR / CU_OUTPUT_DIRS["files"]

    return f"""# CU Coding File Example

This example demonstrates how `diaad cus files` creates complete-utterance coding and reliability workbooks from transcript tables.

## Command

{_fenced("diaad cus files --config config", "bash")}

## Project Files

{_fenced(_project_tree("cus_files"))}

## Basic Config

{_fenced(_cu_config_snippet(specs), "yaml")}

## Advanced Config

{_fenced(_cu_advanced_snippet(specs), "yaml")}

## Input Snippet

The command uses `diaad_data/input/transcript_tables/transcript_tables.xlsx`.

## Output Preview

`expected_outputs/cus_module/cus_files/cu_coding.xlsx`

{_markdown_table(pd.read_excel(output_dir / "cu_coding.xlsx"))}

`expected_outputs/cus_module/cus_files/cu_reliability_coding.xlsx`

{_markdown_table(pd.read_excel(output_dir / "cu_reliability_coding.xlsx"))}

`expected_outputs/cus_module/cus_files/cu_blind_codebook.xlsx`

{_markdown_table(pd.read_excel(output_dir / "cu_blind_codebook.xlsx"))}

## Notes

The generated local example fills the blank CU fields with synthetic coding values so downstream CU examples can be demonstrated. Real `cus files` output starts as coding material for human review.
"""


def _cu_evaluate_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    del specs
    output_dir = project_dir / "expected_outputs" / CUS_MODULE_DIR / CU_OUTPUT_DIRS["evaluate"]
    report = output_dir / "cu_reliability_coding_report.txt"
    report_excerpt = "\n".join(report.read_text(encoding="utf-8").splitlines()[:10])

    return f"""# CU Reliability Evaluation Example

This example demonstrates how `diaad cus evaluate` compares primary CU coding with a synthetic reliability workbook.

## Command

{_fenced("diaad cus evaluate --config config", "bash")}

## Project Files

{_fenced(_project_tree("cus_evaluate"))}

## Basic Config

{_fenced("input_dir: diaad_data/input\noutput_dir: diaad_data/output", "yaml")}

## Input Snippet

The command reads `diaad_data/input/cu_coding/cu_coding.xlsx` and `diaad_data/input/cu_coding/cu_reliability_coding.xlsx`.

## Output Preview

`expected_outputs/cus_module/cus_evaluate/cu_reliability_coding_by_utterance.xlsx`

{_markdown_table(pd.read_excel(output_dir / "cu_reliability_coding_by_utterance.xlsx"))}

`expected_outputs/cus_module/cus_evaluate/cu_reliability_coding_by_sample.xlsx`

{_markdown_table(pd.read_excel(output_dir / "cu_reliability_coding_by_sample.xlsx"))}

`expected_outputs/cus_module/cus_evaluate/cu_reliability_coding_report.txt`

{_fenced(report_excerpt, "text")}

## Notes

The reliability coding values are synthetic and intentionally small. They are meant to show file shape and summary fields, not benchmark agreement.
"""


def _cu_reselect_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / CUS_MODULE_DIR / CU_OUTPUT_DIRS["reselect"]

    return f"""# CU Reliability Reselection Example

This example demonstrates how `diaad cus reselect` selects replacement CU reliability rows after an earlier reliability workbook has already been used.

## Command

{_fenced("diaad cus reselect --config config", "bash")}

## Project Files

{_fenced(_project_tree("cus_reselect"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir", "reliability_fraction", "metadata_fields"]), "yaml")}

## Input Snippet

The command reads prior CU coding and reliability workbooks from `diaad_data/input/cu_coding/`.

## Output Preview

`expected_outputs/cus_module/cus_reselect/reselected_cu_reliability_coding.xlsx`

{_markdown_table(pd.read_excel(output_dir / "reselected_cu_reliability_coding.xlsx"))}

## Notes

The synthetic example has only three samples, so the reselected workbook is intentionally tiny.
"""


def _cu_analyze_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / CUS_MODULE_DIR / CU_OUTPUT_DIRS["analyze"]

    return f"""# CU Coding Analysis Example

This example demonstrates how `diaad cus analyze` summarizes filled complete-utterance coding by utterance and by sample.

## Command

{_fenced("diaad cus analyze --config config", "bash")}

## Project Files

{_fenced(_project_tree("cus_analyze"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Advanced Config

{_fenced(_preview_yaml(specs["advanced_config"], ["metadata_source", "coding_blind_cols", "id_cols"]), "yaml")}

## Input Snippet

The command reads `diaad_data/input/cu_coding/cu_coding.xlsx`. The blind codebook is included so analysis outputs can recover sample identifiers.

## Output Preview

`expected_outputs/cus_module/cus_analyze/cu_coding_by_utterance.xlsx`

{_markdown_table(pd.read_excel(output_dir / "cu_coding_by_utterance.xlsx"))}

`expected_outputs/cus_module/cus_analyze/cu_coding_by_sample_long.xlsx`

{_markdown_table(pd.read_excel(output_dir / "cu_coding_by_sample_long.xlsx"))}

`expected_outputs/cus_module/cus_analyze/cu_coding_by_sample.xlsx`

{_markdown_table(pd.read_excel(output_dir / "cu_coding_by_sample.xlsx"))}

## Notes

The preview uses synthetic filled coding values generated from the packaged example specs.
"""


def _cu_rates_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / CUS_MODULE_DIR / CU_OUTPUT_DIRS["rates"]

    return f"""# CU Rate Calculation Example

This example demonstrates how `diaad cus rates` combines CU sample summaries with speaking times to calculate rates per minute.

## Command

{_fenced("diaad cus rates --config config", "bash")}

## Project Files

{_fenced(_project_tree("cus_rates"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Advanced Config

{_fenced(_preview_yaml(specs["advanced_config"], ["cu_samples_file", "speaking_time_file", "speaking_time_field"]), "yaml")}

## Input Snippet

The command reads `diaad_data/input/cu_coding_analysis/cu_coding_by_sample_long.xlsx` and `diaad_data/input/speaking_times/speaking_times.xlsx`.

## Output Preview

`expected_outputs/cus_module/cus_rates/cu_coding_rates.xlsx`

{_markdown_table(pd.read_excel(output_dir / "cu_coding_rates.xlsx"))}

## Notes

Speaking times are synthetic seconds added to the generated speaking-time template for this example.
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
        utterance_templates_doc = _utterance_templates_doc(project_dir, specs)
        sample_templates_doc = _sample_templates_doc(project_dir, specs)
        time_templates_doc = _time_templates_doc(project_dir, specs)
        cu_files_doc = _cu_files_doc(project_dir, specs)
        cu_evaluate_doc = _cu_evaluate_doc(project_dir, specs)
        cu_reselect_doc = _cu_reselect_doc(project_dir, specs)
        cu_analyze_doc = _cu_analyze_doc(project_dir, specs)
        cu_rates_doc = _cu_rates_doc(project_dir, specs)

    return [
        _write_doc("01_overview.md", text=_overview_doc()),
        _write_doc("transcripts", "tabularize.md", text=tabularize_doc),
        _write_doc("transcripts", "select.md", text=select_doc),
        _write_doc("transcripts", "evaluate.md", text=evaluate_doc),
        _write_doc("transcripts", "reselect.md", text=reselect_doc),
        _write_doc("templates", "utterances.md", text=utterance_templates_doc),
        _write_doc("templates", "samples.md", text=sample_templates_doc),
        _write_doc("templates", "times.md", text=time_templates_doc),
        _write_doc("cus", "files.md", text=cu_files_doc),
        _write_doc("cus", "evaluate.md", text=cu_evaluate_doc),
        _write_doc("cus", "reselect.md", text=cu_reselect_doc),
        _write_doc("cus", "analyze.md", text=cu_analyze_doc),
        _write_doc("cus", "rates.md", text=cu_rates_doc),
    ]
