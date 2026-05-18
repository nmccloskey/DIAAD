from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from diaad.examples.generate import (
    BLINDING_MODULE_DIR,
    BLINDING_OUTPUT_DIRS,
    CU_OUTPUT_DIRS,
    CUS_MODULE_DIR,
    EXPECTED_WORKBOOK,
    POWERS_MODULE_DIR,
    POWERS_OUTPUT_DIRS,
    TEMPLATE_OUTPUT_DIRS,
    TEMPLATES_MODULE_DIR,
    TRANSCRIPTS_MODULE_DIR,
    TURNS_MODULE_DIR,
    TURNS_OUTPUT_DIRS,
    VOCAB_MODULE_DIR,
    VOCAB_OUTPUT_DIRS,
    WORD_OUTPUT_DIRS,
    WORDS_MODULE_DIR,
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


def _preview_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2)


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
    contents = contents.rstrip()
    if contents:
        return f"""    output/
      {RUN_DIR}/
{contents}
{_logs_tree()}"""
    return f"""    output/
      {RUN_DIR}/
{_logs_tree()}"""


def _project_tree(command: str = "all") -> str:
    if command == "tabularize":
        input_files = """      chat/
        P1_picnic_pre.cha
        P2_picnic_pre.cha
        P1_picnic_post.cha"""
        outputs = """        transcript_tables/
          transcript_tables.xlsx"""
    elif command == "blinding_encode":
        input_files = """      powers_coding/
        powers_coding.xlsx"""
        outputs = """        blinding/
          powers_coding_blinded.xlsx
          powers_coding_blinding_diagnostics.xlsx
          blind_codebook.xlsx"""
    elif command == "blinding_decode":
        input_files = """      cu_coding/
        cu_coding.xlsx
        cu_blind_codebook.xlsx"""
        outputs = """        blinding/
          cu_coding_decoded.xlsx"""
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
    elif command == "words_files":
        input_files = """      transcript_tables/
        transcript_tables.xlsx"""
        outputs = """        word_counts/
          word_counting.xlsx
          word_count_reliability.xlsx
          word_count_blind_codebook.xlsx"""
    elif command == "words_evaluate":
        input_files = """      word_counts/
        word_counting.xlsx
        word_count_reliability.xlsx"""
        outputs = """        word_count_reliability/
          word_count_reliability_results.xlsx
          word_count_reliability_report.txt"""
    elif command == "words_reselect":
        input_files = """      word_counts/
        word_counting.xlsx
        word_count_reliability.xlsx"""
        outputs = """        reselected_word_count_reliability/
          reselected_word_count_reliability.xlsx"""
    elif command == "words_analyze":
        input_files = """      word_counts/
        word_counting.xlsx
        word_count_blind_codebook.xlsx"""
        outputs = """        word_count_analysis/
          word_counting_by_utterance.xlsx
          word_counting_by_sample.xlsx"""
    elif command == "words_rates":
        input_files = """      word_count_analysis/
        word_counting_by_sample.xlsx
      speaking_times/
        speaking_times.xlsx"""
        outputs = """        word_count_analysis/
          word_counting_rates.xlsx"""
    elif command == "powers_files":
        input_files = """      transcript_tables/
        transcript_tables.xlsx"""
        outputs = """        powers_coding/
          powers_coding.xlsx
          powers_reliability_coding.xlsx"""
    elif command == "powers_evaluate":
        input_files = """      powers_coding/
        powers_coding.xlsx
        powers_reliability_coding.xlsx"""
        outputs = """        powers_reliability/
          powers_reliability_results.xlsx
          powers_reliability_report.txt"""
    elif command == "powers_reselect":
        input_files = """      powers_coding/
        powers_coding.xlsx
        powers_reliability_coding.xlsx"""
        outputs = """        reselected_powers_reliability/
          reselected_powers_reliability_coding.xlsx"""
    elif command == "powers_analyze":
        input_files = """      powers_coding/
        powers_coding.xlsx"""
        outputs = """        powers_coding_analysis/
          powers_analysis.xlsx"""
    elif command == "powers_rates":
        input_files = """      powers_coding_analysis/
        powers_analysis.xlsx
      speaking_times/
        speaking_times.xlsx"""
        outputs = """        powers_coding_analysis/
          powers_coding_rates.xlsx"""
    elif command == "vocab_file":
        input_files = """      target_vocab/
        resources/"""
        outputs = """        target_vocab/
          target_vocabulary_resource_template.json"""
    elif command == "vocab_check":
        input_files = """      target_vocab/
        resources/
          picnic_target_vocab.json"""
        outputs = """        target_vocab/
          target_vocab_resource_check.txt"""
    elif command == "vocab_analyze":
        input_files = """      target_vocab/
        unblind_utterance_data.xlsx
        resources/
          picnic_target_vocab.json"""
        outputs = """        target_vocab/
          target_vocab_data_YYMMDD_HHMM.xlsx"""
    elif command == "vocab_rates":
        input_files = """      target_vocab_analysis/
        target_vocab_data_YYMMDD_HHMM.xlsx"""
        outputs = """        target_vocab/
          target_vocab_rates.xlsx"""
    elif command == "turns_files":
        input_files = """      transcript_tables/
        transcript_tables.xlsx"""
        outputs = """        coding_templates/
          conversation_turns_template.xlsx
          conversation_turns_reliability_template.xlsx
          conversation_turns_template_codebook.xlsx"""
    elif command == "turns_evaluate":
        input_files = """      conversation_turns/
        conversation_turns_template.xlsx
        conversation_turns_reliability_template.xlsx"""
        outputs = """        turns_reliability/
          conversation_turns_reliability_results.xlsx
          conversation_turns_reliability_report.txt
          global_alignments/"""
    elif command == "turns_reselect":
        input_files = """      conversation_turns/
        conversation_turns_template.xlsx
        conversation_turns_reliability_template.xlsx"""
        outputs = """        reselected_turns_reliability/
          reselected_conversation_turns_reliability_template.xlsx"""
    elif command == "turns_analyze":
        input_files = """      conversation_turns/
        conversation_turns_template.xlsx"""
        outputs = """        conversation_turns_template_analysis.xlsx"""
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
      word_counts/
        word_counting.xlsx
        word_count_reliability.xlsx
        word_count_blind_codebook.xlsx
      word_count_analysis/
        word_counting_by_sample.xlsx
      powers_coding/
        powers_coding.xlsx
        powers_reliability_coding.xlsx
      powers_coding_analysis/
        powers_analysis.xlsx
      target_vocab/
        resources/
          picnic_target_vocab.json
        unblind_utterance_data.xlsx
      target_vocab_analysis/
        target_vocab_data_260101_0000.xlsx
      conversation_turns/
        conversation_turns_template.xlsx
        conversation_turns_reliability_template.xlsx
      speaking_times/
        speaking_times.xlsx
    expected_outputs/
      blinding_module/
        blinding_encode/
          powers_coding_blinded.xlsx
          powers_coding_blinding_diagnostics.xlsx
          blind_codebook.xlsx
        blinding_decode/
          cu_coding_decoded.xlsx
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
        + """
      words_module/
        words_files/
          word_counting.xlsx
          word_count_reliability.xlsx
          word_count_blind_codebook.xlsx
        words_evaluate/
          word_count_reliability_results.xlsx
          word_count_reliability_report.txt
        words_reselect/
          reselected_word_count_reliability.xlsx
        words_analyze/
          word_counting_by_utterance.xlsx
          word_counting_by_sample.xlsx
        words_rates/
          word_counting_rates.xlsx"""
        + """
      powers_module/
        powers_files/
          powers_coding.xlsx
          powers_reliability_coding.xlsx
        powers_evaluate/
          powers_reliability_results.xlsx
          powers_reliability_report.txt
        powers_reselect/
          reselected_powers_reliability_coding.xlsx
        powers_analyze/
          powers_analysis.xlsx
        powers_rates/
          powers_coding_rates.xlsx"""
        + """
      vocab_module/
        vocab_file/
          target_vocabulary_resource_template.json
        vocab_check/
          target_vocab_resource_check.txt
        vocab_analyze/
          target_vocab_data_260101_0000.xlsx
        vocab_rates/
          target_vocab_rates.xlsx"""
        + """
      turns_module/
        turns_files/
          conversation_turns_template.xlsx
          conversation_turns_reliability_template.xlsx
          conversation_turns_template_codebook.xlsx
        turns_evaluate/
          conversation_turns_reliability_results.xlsx
          conversation_turns_reliability_report.txt
        turns_reselect/
          reselected_conversation_turns_reliability_template.xlsx
        turns_analyze/
          conversation_turns_template_analysis.xlsx"""
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
        "vocab_resource": _read_yaml_asset(*SPEC_ROOT, "vocab", "picnic_resource.yaml"),
        "turns_sessions": _read_yaml_asset(*SPEC_ROOT, "turns", "sessions.yaml"),
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


def _blinding_advanced_snippet(specs: dict[str, dict[str, Any]]) -> str:
    data = specs["advanced_config"].copy()
    data["auto_blind"] = True
    data["blind_cols"] = ["sample_id"]
    return _preview_yaml(data, ["auto_blind", "blind_cols"])


def _blinding_encode_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = (
        project_dir
        / "expected_outputs"
        / BLINDING_MODULE_DIR
        / BLINDING_OUTPUT_DIRS["encode"]
    )
    input_path = project_dir / "input" / "powers_coding" / "powers_coding.xlsx"

    return f"""# Blinding Encode Example

This example demonstrates how `diaad blinding encode` blinds `sample_id` in a standalone workbook and writes a reusable codebook.

## Command

{_fenced("diaad blinding encode --config config", "bash")}

## Project Files

{_fenced(_project_tree("blinding_encode"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir", "random_seed"]), "yaml")}

## Advanced Config

{_fenced(_blinding_advanced_snippet(specs), "yaml")}

## Input Snippet

`diaad_data/input/powers_coding/powers_coding.xlsx`

{_markdown_table(pd.read_excel(input_path))}

## Output Preview

`expected_outputs/blinding_module/blinding_encode/powers_coding_blinded.xlsx`

{_markdown_table(pd.read_excel(output_dir / "powers_coding_blinded.xlsx"))}

`expected_outputs/blinding_module/blinding_encode/blind_codebook.xlsx`

{_markdown_table(pd.read_excel(output_dir / "blind_codebook.xlsx"))}

`expected_outputs/blinding_module/blinding_encode/powers_coding_blinding_diagnostics.xlsx`

{_markdown_table(pd.read_excel(output_dir / "powers_coding_blinding_diagnostics.xlsx"))}

## Notes

The input is the synthetic POWERS coding workbook from the generated example project. The command discovers the first non-codebook `.xlsx` in the input folder and generates a new blind codebook because no codebook is supplied.
"""


def _blinding_decode_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = (
        project_dir
        / "expected_outputs"
        / BLINDING_MODULE_DIR
        / BLINDING_OUTPUT_DIRS["decode"]
    )
    input_path = project_dir / "input" / "cu_coding" / "cu_coding.xlsx"
    codebook_path = project_dir / "input" / "cu_coding" / "cu_blind_codebook.xlsx"

    return f"""# Blinding Decode Example

This example demonstrates how `diaad blinding decode` restores blinded identifiers in a standalone workbook using a blind codebook.

## Command

{_fenced("diaad blinding decode --config config", "bash")}

## Project Files

{_fenced(_project_tree("blinding_decode"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Input Snippet

`diaad_data/input/cu_coding/cu_coding.xlsx`

{_markdown_table(pd.read_excel(input_path))}

`diaad_data/input/cu_coding/cu_blind_codebook.xlsx`

{_markdown_table(pd.read_excel(codebook_path))}

## Output Preview

`expected_outputs/blinding_module/blinding_decode/cu_coding_decoded.xlsx`

{_markdown_table(pd.read_excel(output_dir / "cu_coding_decoded.xlsx"))}

## Notes

The input is the synthetic CU coding workbook and codebook created by the CU examples. The decode command discovers the codebook, restores `sample_id`, and removes the suffixed blinded identifier column.
"""


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
        ["auto_blind", "blind_cols", "metadata_source", "codebook_filename"],
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
        ["cu_paradigms", "auto_blind", "blind_cols", "metadata_source", "codebook_filename"],
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

{_fenced(_preview_yaml(specs["advanced_config"], ["auto_blind", "blind_cols", "metadata_source", "codebook_filename"]), "yaml")}

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


def _word_config_snippet(specs: dict[str, dict[str, Any]]) -> str:
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


def _word_advanced_snippet(specs: dict[str, dict[str, Any]], keys: list[str] | None = None) -> str:
    return _preview_yaml(
        specs["advanced_config"],
        keys
        or [
            "word_count_file",
            "word_count_field",
            "metadata_source",
            "auto_blind",
            "blind_cols",
        ],
    )


def _word_files_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / WORDS_MODULE_DIR / WORD_OUTPUT_DIRS["files"]

    return f"""# Word Count File Example

This example demonstrates how `diaad words files` creates word-count coding and reliability workbooks from transcript tables.

## Command

{_fenced("diaad words files --config config", "bash")}

## Project Files

{_fenced(_project_tree("words_files"))}

## Basic Config

{_fenced(_word_config_snippet(specs), "yaml")}

## Advanced Config

{_fenced(_word_advanced_snippet(specs), "yaml")}

## Input Snippet

The command uses `diaad_data/input/transcript_tables/transcript_tables.xlsx`.

## Output Preview

`expected_outputs/words_module/words_files/word_counting.xlsx`

{_markdown_table(pd.read_excel(output_dir / "word_counting.xlsx"))}

`expected_outputs/words_module/words_files/word_count_reliability.xlsx`

{_markdown_table(pd.read_excel(output_dir / "word_count_reliability.xlsx"))}

`expected_outputs/words_module/words_files/word_count_blind_codebook.xlsx`

{_markdown_table(pd.read_excel(output_dir / "word_count_blind_codebook.xlsx"))}

## Notes

The generated local example fills synthetic word counts into the blank coding workbooks so downstream word-count examples can be demonstrated. Real `words files` output starts as coding material for human review.
"""


def _word_evaluate_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / WORDS_MODULE_DIR / WORD_OUTPUT_DIRS["evaluate"]
    report = output_dir / "word_count_reliability_report.txt"
    report_excerpt = "\n".join(report.read_text(encoding="utf-8").splitlines()[:10])

    return f"""# Word Count Reliability Evaluation Example

This example demonstrates how `diaad words evaluate` compares primary word counts with a synthetic reliability workbook.

## Command

{_fenced("diaad words evaluate --config config", "bash")}

## Project Files

{_fenced(_project_tree("words_evaluate"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Advanced Config

{_fenced(_word_advanced_snippet(specs, ["word_count_file", "word_count_field"]), "yaml")}

## Input Snippet

The command reads `diaad_data/input/word_counts/word_counting.xlsx` and `diaad_data/input/word_counts/word_count_reliability.xlsx`.

## Output Preview

`expected_outputs/words_module/words_evaluate/word_count_reliability_results.xlsx`

{_markdown_table(pd.read_excel(output_dir / "word_count_reliability_results.xlsx"))}

`expected_outputs/words_module/words_evaluate/word_count_reliability_report.txt`

{_fenced(report_excerpt, "text")}

## Notes

Reliability word counts are synthetic, with small deterministic differences from the primary counts.
"""


def _word_reselect_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / WORDS_MODULE_DIR / WORD_OUTPUT_DIRS["reselect"]

    return f"""# Word Count Reliability Reselection Example

This example demonstrates how `diaad words reselect` selects replacement word-count reliability rows after an earlier reliability workbook has already been used.

## Command

{_fenced("diaad words reselect --config config", "bash")}

## Project Files

{_fenced(_project_tree("words_reselect"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir", "reliability_fraction", "metadata_fields"]), "yaml")}

## Input Snippet

The command reads prior word-count coding and reliability workbooks from `diaad_data/input/word_counts/`.

## Output Preview

`expected_outputs/words_module/words_reselect/reselected_word_count_reliability.xlsx`

{_markdown_table(pd.read_excel(output_dir / "reselected_word_count_reliability.xlsx"))}

## Notes

The synthetic example has only three samples, so the reselected workbook is intentionally tiny.
"""


def _word_analyze_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / WORDS_MODULE_DIR / WORD_OUTPUT_DIRS["analyze"]

    return f"""# Word Count Analysis Example

This example demonstrates how `diaad words analyze` summarizes filled word-count coding by utterance and by sample.

## Command

{_fenced("diaad words analyze --config config", "bash")}

## Project Files

{_fenced(_project_tree("words_analyze"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Advanced Config

{_fenced(_word_advanced_snippet(specs, ["word_count_file", "word_count_field", "auto_blind", "blind_cols", "metadata_source", "codebook_filename"]), "yaml")}

## Input Snippet

The command reads `diaad_data/input/word_counts/word_counting.xlsx`. The blind codebook is included so analysis outputs can recover sample identifiers.

## Output Preview

`expected_outputs/words_module/words_analyze/word_counting_by_utterance.xlsx`

{_markdown_table(pd.read_excel(output_dir / "word_counting_by_utterance.xlsx"))}

`expected_outputs/words_module/words_analyze/word_counting_by_sample.xlsx`

{_markdown_table(pd.read_excel(output_dir / "word_counting_by_sample.xlsx"))}

## Notes

The preview uses synthetic filled word counts generated from the packaged example specs.
"""


def _word_rates_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / WORDS_MODULE_DIR / WORD_OUTPUT_DIRS["rates"]

    return f"""# Word Count Rate Calculation Example

This example demonstrates how `diaad words rates` combines word-count sample summaries with speaking times to calculate rates per minute.

## Command

{_fenced("diaad words rates --config config", "bash")}

## Project Files

{_fenced(_project_tree("words_rates"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Advanced Config

{_fenced(_word_advanced_snippet(specs, ["wc_samples_file", "speaking_time_file", "speaking_time_field"]), "yaml")}

## Input Snippet

The command reads `diaad_data/input/word_count_analysis/word_counting_by_sample.xlsx` and `diaad_data/input/speaking_times/speaking_times.xlsx`.

## Output Preview

`expected_outputs/words_module/words_rates/word_counting_rates.xlsx`

{_markdown_table(pd.read_excel(output_dir / "word_counting_rates.xlsx"))}

## Notes

Speaking times are synthetic seconds added to the generated speaking-time template for this example.
"""


def _powers_config_snippet(specs: dict[str, dict[str, Any]]) -> str:
    return _project_config_snippet(
        specs,
        [
            "input_dir",
            "output_dir",
            "reliability_fraction",
            "num_coders",
            "stimulus_field",
            "exclude_participants",
            "automate_powers",
        ],
    )


def _powers_files_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / POWERS_MODULE_DIR / POWERS_OUTPUT_DIRS["files"]

    return f"""# POWERS Coding File Example

This example demonstrates how `diaad powers files` creates POWERS coding and reliability workbooks from transcript tables.

## Command

{_fenced("diaad powers files --config config", "bash")}

## Project Files

{_fenced(_project_tree("powers_files"))}

## Basic Config

{_fenced(_powers_config_snippet(specs), "yaml")}

## Input Snippet

The command uses `diaad_data/input/transcript_tables/transcript_tables.xlsx`.

## Output Preview

`expected_outputs/powers_module/powers_files/powers_coding.xlsx`

{_all_workbook_sheet_tables(output_dir / "powers_coding.xlsx")}

`expected_outputs/powers_module/powers_files/powers_reliability_coding.xlsx`

{_markdown_table(pd.read_excel(output_dir / "powers_reliability_coding.xlsx"))}

## Notes

The generated local example fills synthetic POWERS values into the blank coding workbooks so downstream POWERS examples can be demonstrated. Real `powers files` output starts as coding material for human review. Automation is disabled in the synthetic config to keep the example deterministic and dependency-light.
"""


def _powers_evaluate_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / POWERS_MODULE_DIR / POWERS_OUTPUT_DIRS["evaluate"]
    report = output_dir / "powers_reliability_report.txt"
    report_excerpt = "\n".join(report.read_text(encoding="utf-8").splitlines()[:12])

    return f"""# POWERS Reliability Evaluation Example

This example demonstrates how `diaad powers evaluate` compares primary POWERS coding with a synthetic reliability workbook.

## Command

{_fenced("diaad powers evaluate --config config", "bash")}

## Project Files

{_fenced(_project_tree("powers_evaluate"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Input Snippet

The command reads `diaad_data/input/powers_coding/powers_coding.xlsx` and `diaad_data/input/powers_coding/powers_reliability_coding.xlsx`.

## Output Preview

`expected_outputs/powers_module/powers_evaluate/powers_reliability_results.xlsx`

{_all_workbook_sheet_tables(output_dir / "powers_reliability_results.xlsx")}

`expected_outputs/powers_module/powers_evaluate/powers_reliability_report.txt`

{_fenced(report_excerpt, "text")}

## Notes

Reliability POWERS values are synthetic, with small deterministic differences from the primary coding.
"""


def _powers_reselect_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / POWERS_MODULE_DIR / POWERS_OUTPUT_DIRS["reselect"]

    return f"""# POWERS Reliability Reselection Example

This example demonstrates how `diaad powers reselect` selects replacement POWERS reliability rows after an earlier reliability workbook has already been used.

## Command

{_fenced("diaad powers reselect --config config", "bash")}

## Project Files

{_fenced(_project_tree("powers_reselect"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir", "reliability_fraction", "metadata_fields", "automate_powers"]), "yaml")}

## Input Snippet

The command reads prior POWERS coding and reliability workbooks from `diaad_data/input/powers_coding/`.

## Output Preview

`expected_outputs/powers_module/powers_reselect/reselected_powers_reliability_coding.xlsx`

{_markdown_table(pd.read_excel(output_dir / "reselected_powers_reliability_coding.xlsx"))}

## Notes

The synthetic example has only three samples, so the reselected workbook is intentionally tiny.
"""


def _powers_analyze_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / POWERS_MODULE_DIR / POWERS_OUTPUT_DIRS["analyze"]

    return f"""# POWERS Coding Analysis Example

This example demonstrates how `diaad powers analyze` summarizes filled POWERS coding by utterance, turn, speaker, and dialog.

## Command

{_fenced("diaad powers analyze --config config", "bash")}

## Project Files

{_fenced(_project_tree("powers_analyze"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Input Snippet

The command reads `diaad_data/input/powers_coding/powers_coding.xlsx`.

## Output Preview

`expected_outputs/powers_module/powers_analyze/powers_analysis.xlsx`

{_all_workbook_sheet_tables(output_dir / "powers_analysis.xlsx")}

## Notes

The preview uses synthetic filled POWERS values generated from the packaged example specs.
"""


def _powers_rates_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / POWERS_MODULE_DIR / POWERS_OUTPUT_DIRS["rates"]

    return f"""# POWERS Rate Calculation Example

This example demonstrates how `diaad powers rates` combines POWERS dialog summaries with speaking times to calculate rates per minute.

## Command

{_fenced("diaad powers rates --config config", "bash")}

## Project Files

{_fenced(_project_tree("powers_rates"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Advanced Config

{_fenced(_preview_yaml(specs["advanced_config"], ["speaking_time_file", "speaking_time_field"]), "yaml")}

## Input Snippet

The command reads `diaad_data/input/powers_coding_analysis/powers_analysis.xlsx` and `diaad_data/input/speaking_times/speaking_times.xlsx`.

## Output Preview

`expected_outputs/powers_module/powers_rates/powers_coding_rates.xlsx`

{_markdown_table(pd.read_excel(output_dir / "powers_coding_rates.xlsx"))}

## Notes

Speaking times are synthetic seconds added to the generated speaking-time template for this example.
"""


def _vocab_resource_snippet(specs: dict[str, dict[str, Any]]) -> str:
    resource = specs["vocab_resource"]
    subset = {
        "id": resource.get("id"),
        "display_name": resource.get("display_name"),
        "base_forms": resource.get("base_forms", [])[:8],
        "variant_map": {
            key: value
            for key, value in list(resource.get("variant_map", {}).items())[:3]
        },
    }
    return _preview_json(subset)


def _vocab_note() -> str:
    return (
        "DIAAD includes five built-in narrative resources: `BrokenWindow`, "
        "`CatRescue`, `Cinderella`, `RefusedUmbrella`, and `Sandwich`. Those "
        "built-ins require no user JSON. This synthetic picnic example uses a "
        "small custom JSON resource so the vocabulary targets match the synthetic "
        "transcripts."
    )


def _vocab_advanced_snippet() -> str:
    return yaml.safe_dump(
        {
            "target_vocabulary_resource_path": (
                "diaad_data/input/target_vocab/resources/picnic_target_vocab.json"
            )
        },
        sort_keys=False,
        allow_unicode=False,
    ).rstrip()


def _vocab_file_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / VOCAB_MODULE_DIR / VOCAB_OUTPUT_DIRS["file"]
    template_text = (output_dir / "target_vocabulary_resource_template.json").read_text(
        encoding="utf-8"
    )

    return f"""# Target Vocabulary Resource Template Example

This example demonstrates how `diaad vocab file` creates a blank JSON template for a custom target-vocabulary resource.

## Command

{_fenced("diaad vocab file --config config", "bash")}

## Project Files

{_fenced(_project_tree("vocab_file"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Output Preview

`expected_outputs/vocab_module/vocab_file/target_vocabulary_resource_template.json`

{_fenced(template_text[:1200], "json")}

## Notes

{_vocab_note()} Use `diaad vocab file` when a project needs to start authoring a custom resource.
"""


def _vocab_check_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / VOCAB_MODULE_DIR / VOCAB_OUTPUT_DIRS["check"]
    report = (output_dir / "target_vocab_resource_check.txt").read_text(encoding="utf-8")

    return f"""# Target Vocabulary Resource Check Example

This example demonstrates how `diaad vocab check` validates the active built-in and custom target-vocabulary resources.

## Command

{_fenced("diaad vocab check --config config", "bash")}

## Project Files

{_fenced(_project_tree("vocab_check"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Advanced Config

{_fenced(_vocab_advanced_snippet(), "yaml")}

## Input Snippet

`diaad_data/input/target_vocab/resources/picnic_target_vocab.json`

{_fenced(_vocab_resource_snippet(specs), "json")}

## Output Preview

`expected_outputs/vocab_module/vocab_check/target_vocab_resource_check.txt`

{_fenced(report, "text")}

## Notes

{_vocab_note()} The command reports validation details through the DIAAD run log; the generated text file shown here is a compact documentation preview of the same resource set.
"""


def _vocab_analyze_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / VOCAB_MODULE_DIR / VOCAB_OUTPUT_DIRS["analyze"]
    workbook = output_dir / "target_vocab_data_260101_0000.xlsx"
    input_path = project_dir / "input" / "target_vocab" / "unblind_utterance_data.xlsx"

    return f"""# Target Vocabulary Analysis Example

This example demonstrates how `diaad vocab analyze` calculates target-vocabulary coverage for synthetic picnic samples.

## Command

{_fenced("diaad vocab analyze --config config", "bash")}

## Project Files

{_fenced(_project_tree("vocab_analyze"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir", "metadata_fields", "stimulus_field", "exclude_participants"]), "yaml")}

## Advanced Config

{_fenced(_vocab_advanced_snippet(), "yaml")}

## Input Snippet

`diaad_data/input/target_vocab/resources/picnic_target_vocab.json`

{_fenced(_vocab_resource_snippet(specs), "json")}

`diaad_data/input/target_vocab/unblind_utterance_data.xlsx`

{_markdown_table(pd.read_excel(input_path))}

## Output Preview

`expected_outputs/vocab_module/vocab_analyze/target_vocab_data_260101_0000.xlsx`

{_workbook_sheet_tables(workbook, ["summary", "details"])}

## Notes

{_vocab_note()} The custom picnic resource intentionally has no norm tables, so percentile columns are blank while coverage counts and rates are still calculated.
"""


def _vocab_rates_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / VOCAB_MODULE_DIR / VOCAB_OUTPUT_DIRS["rates"]

    return f"""# Target Vocabulary Rate Calculation Example

This example demonstrates how `diaad vocab rates` converts target-vocabulary analysis counts into per-minute rates.

## Command

{_fenced("diaad vocab rates --config config", "bash")}

## Project Files

{_fenced(_project_tree("vocab_rates"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Input Snippet

The command reads `diaad_data/input/target_vocab_analysis/target_vocab_data_YYMMDD_HHMM.xlsx`.

## Output Preview

`expected_outputs/vocab_module/vocab_rates/target_vocab_rates.xlsx`

{_markdown_table(pd.read_excel(output_dir / "target_vocab_rates.xlsx"))}

## Notes

{_vocab_note()} Rates are based on the `speaking_time` values stored in the target-vocabulary analysis summary sheet.
"""


def _turns_rows_table(specs: dict[str, dict[str, Any]], key: str) -> str:
    return _markdown_table(pd.DataFrame(specs["turns_sessions"].get(key, [])))


def _turns_files_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / TURNS_MODULE_DIR / TURNS_OUTPUT_DIRS["files"]

    return f"""# Conversation Turns File Example

This example demonstrates how `diaad turns files` creates blank digital conversation-turn coding and reliability workbooks.

## Command

{_fenced("diaad turns files --config config", "bash")}

## Project Files

{_fenced(_project_tree("turns_files"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir", "reliability_fraction", "num_bins", "num_coders", "metadata_fields"]), "yaml")}

## Advanced Config

{_fenced(_preview_yaml(specs["advanced_config"], ["auto_blind", "blind_cols", "metadata_source", "codebook_filename"]), "yaml")}

## Input Snippet

The command uses `diaad_data/input/transcript_tables/transcript_tables.xlsx` to create one row per sample and bin.

## Output Preview

`expected_outputs/turns_module/turns_files/conversation_turns_template.xlsx`

{_markdown_table(pd.read_excel(output_dir / "conversation_turns_template.xlsx"))}

`expected_outputs/turns_module/turns_files/conversation_turns_reliability_template.xlsx`

{_markdown_table(pd.read_excel(output_dir / "conversation_turns_reliability_template.xlsx"))}

## Notes

The generated local example fills separate synthetic turn strings into conversation-turn workbooks so downstream examples can be demonstrated. Digits identify speakers and dot markers are preserved for analysis.
"""


def _turns_evaluate_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / TURNS_MODULE_DIR / TURNS_OUTPUT_DIRS["evaluate"]
    report = output_dir / "conversation_turns_reliability_report.txt"
    report_excerpt = "\n".join(report.read_text(encoding="utf-8").splitlines()[:14])

    return f"""# Conversation Turns Reliability Evaluation Example

This example demonstrates how `diaad turns evaluate` compares primary and reliability-coded digital conversation turns.

## Command

{_fenced("diaad turns evaluate --config config", "bash")}

## Project Files

{_fenced(_project_tree("turns_evaluate"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir", "metadata_fields"]), "yaml")}

## Input Snippet

`diaad_data/input/conversation_turns/conversation_turns_template.xlsx`

{_turns_rows_table(specs, "primary_rows")}

`diaad_data/input/conversation_turns/conversation_turns_reliability_template.xlsx`

{_turns_rows_table(specs, "reliability_rows")}

## Output Preview

`expected_outputs/turns_module/turns_evaluate/conversation_turns_reliability_results.xlsx`

{_workbook_sheet_tables(output_dir / "conversation_turns_reliability_results.xlsx", ["counts", "sequences", "samples"])}

`expected_outputs/turns_module/turns_evaluate/conversation_turns_reliability_report.txt`

{_fenced(report_excerpt, "text")}

## Notes

The synthetic turn strings use four speakers (`0`, `1`, `2`, and `3`) and include both mark1 (`.`) and mark2 (`..`) examples.
"""


def _turns_reselect_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / TURNS_MODULE_DIR / TURNS_OUTPUT_DIRS["reselect"]
    workbook = next(output_dir.glob("*.xlsx"))

    return f"""# Conversation Turns Reliability Reselection Example

This example demonstrates how `diaad turns reselect` selects replacement samples for digital conversation-turn reliability coding.

## Command

{_fenced("diaad turns reselect --config config", "bash")}

## Project Files

{_fenced(_project_tree("turns_reselect"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir", "reliability_fraction", "random_seed", "metadata_fields"]), "yaml")}

## Input Snippet

The primary turns workbook has two synthetic sample IDs. The prior reliability workbook contains only `S001`, so reselection can choose a replacement sample.

## Output Preview

`expected_outputs/turns_module/turns_reselect/{workbook.name}`

{_markdown_table(pd.read_excel(workbook))}

## Notes

Reselected rows keep the session and bin structure while clearing the `turns` cells for fresh reliability coding.
"""


def _turns_analyze_doc(project_dir: Path, specs: dict[str, dict[str, Any]]) -> str:
    output_dir = project_dir / "expected_outputs" / TURNS_MODULE_DIR / TURNS_OUTPUT_DIRS["analyze"]
    workbook = output_dir / "conversation_turns_template_analysis.xlsx"

    return f"""# Conversation Turns Analysis Example

This example demonstrates how `diaad turns analyze` summarizes digital conversation-turn strings across speakers, bins, sessions, and groups.

## Command

{_fenced("diaad turns analyze --config config", "bash")}

## Project Files

{_fenced(_project_tree("turns_analyze"))}

## Basic Config

{_fenced(_project_config_snippet(specs, ["input_dir", "output_dir"]), "yaml")}

## Input Snippet

`diaad_data/input/conversation_turns/conversation_turns_template.xlsx`

{_turns_rows_table(specs, "primary_rows")}

## Output Preview

`expected_outputs/turns_module/turns_analyze/conversation_turns_template_analysis.xlsx`

{_all_workbook_sheet_tables(workbook)}

## Notes

The strings are deliberately tiny but include two sessions, two bins, four speakers, and both dot-marker forms.
"""


def render_example_docs() -> list[Path]:
    """Create or update packaged example I/O markdown assets."""
    specs = _read_specs()
    with _scratch_dir(Path.cwd()) as tmpdir:
        project_dir = generate_example_files(tmpdir / "synthetic_project", force=True)
        blinding_encode_doc = _blinding_encode_doc(project_dir, specs)
        blinding_decode_doc = _blinding_decode_doc(project_dir, specs)
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
        word_files_doc = _word_files_doc(project_dir, specs)
        word_evaluate_doc = _word_evaluate_doc(project_dir, specs)
        word_reselect_doc = _word_reselect_doc(project_dir, specs)
        word_analyze_doc = _word_analyze_doc(project_dir, specs)
        word_rates_doc = _word_rates_doc(project_dir, specs)
        powers_files_doc = _powers_files_doc(project_dir, specs)
        powers_evaluate_doc = _powers_evaluate_doc(project_dir, specs)
        powers_reselect_doc = _powers_reselect_doc(project_dir, specs)
        powers_analyze_doc = _powers_analyze_doc(project_dir, specs)
        powers_rates_doc = _powers_rates_doc(project_dir, specs)
        vocab_file_doc = _vocab_file_doc(project_dir, specs)
        vocab_check_doc = _vocab_check_doc(project_dir, specs)
        vocab_analyze_doc = _vocab_analyze_doc(project_dir, specs)
        vocab_rates_doc = _vocab_rates_doc(project_dir, specs)
        turns_files_doc = _turns_files_doc(project_dir, specs)
        turns_evaluate_doc = _turns_evaluate_doc(project_dir, specs)
        turns_reselect_doc = _turns_reselect_doc(project_dir, specs)
        turns_analyze_doc = _turns_analyze_doc(project_dir, specs)

    return [
        _write_doc("01_overview.md", text=_overview_doc()),
        _write_doc("blinding", "encode.md", text=blinding_encode_doc),
        _write_doc("blinding", "decode.md", text=blinding_decode_doc),
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
        _write_doc("words", "files.md", text=word_files_doc),
        _write_doc("words", "evaluate.md", text=word_evaluate_doc),
        _write_doc("words", "reselect.md", text=word_reselect_doc),
        _write_doc("words", "analyze.md", text=word_analyze_doc),
        _write_doc("words", "rates.md", text=word_rates_doc),
        _write_doc("powers", "files.md", text=powers_files_doc),
        _write_doc("powers", "evaluate.md", text=powers_evaluate_doc),
        _write_doc("powers", "reselect.md", text=powers_reselect_doc),
        _write_doc("powers", "analyze.md", text=powers_analyze_doc),
        _write_doc("powers", "rates.md", text=powers_rates_doc),
        _write_doc("vocab", "file.md", text=vocab_file_doc),
        _write_doc("vocab", "check.md", text=vocab_check_doc),
        _write_doc("vocab", "analyze.md", text=vocab_analyze_doc),
        _write_doc("vocab", "rates.md", text=vocab_rates_doc),
        _write_doc("turns", "files.md", text=turns_files_doc),
        _write_doc("turns", "evaluate.md", text=turns_evaluate_doc),
        _write_doc("turns", "reselect.md", text=turns_reselect_doc),
        _write_doc("turns", "analyze.md", text=turns_analyze_doc),
    ]
