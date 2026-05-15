from __future__ import annotations

import re
import zipfile
from io import BytesIO
from typing import Any

import streamlit as st
import yaml


def _split_values(text: str) -> list[str]:
    """Split comma- or newline-delimited UI input into non-empty strings."""
    if not text:
        return []
    parts = re.split(r"[,\n]", text)
    return [p.strip() for p in parts if p.strip()]


def _config_zip(configs: dict[str, dict]) -> BytesIO:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in ("project", "advanced"):
            zf.writestr(
                f"{name}.yaml",
                yaml.safe_dump(configs[name], sort_keys=False, allow_unicode=True),
            )
    buffer.seek(0)
    return buffer


def _section_toggle(label: str, *, key: str, expanded: bool = False) -> bool:
    state_key = f"{key}_expanded"
    button_key = f"{key}_button"
    if state_key not in st.session_state:
        st.session_state[state_key] = expanded
        st.rerun()

    is_expanded = bool(st.session_state[state_key])
    caret = "v" if is_expanded else ">"
    if st.button(f"{caret} {label}", key=button_key):
        st.session_state[state_key] = not is_expanded
        is_expanded = bool(st.session_state[state_key])
        st.rerun()

    return is_expanded


def _text_list_input(
    label: str,
    *,
    value: list[str],
    key: str,
    help: str,
) -> list[str]:
    text = st.text_area(label, value=", ".join(value), key=key, help=help)
    return _split_values(text)


def _build_field(
    field_type: str,
    label: str,
    *,
    default: Any,
    key: str,
    help: str,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
    step: int | float | None = None,
) -> Any:
    if field_type == "bool":
        return st.checkbox(label, value=bool(default), key=key, help=help)
    if field_type == "int":
        return st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=int(default),
            step=int(step or 1),
            key=key,
            help=help,
        )
    if field_type == "float":
        return st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            value=float(default),
            step=float(step or 0.05),
            key=key,
            help=help,
        )
    if field_type == "list":
        return _text_list_input(label, value=list(default), key=key, help=help)
    return st.text_input(label, value=str(default), key=key, help=help)


def _render_fields(
    config_name: str,
    section_name: str,
    fields: list[dict[str, Any]],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    section_key = re.sub(r"\W+", "_", f"{config_name}_{section_name}").strip("_").lower()
    if not _section_toggle(section_name, key=f"show_{section_key}"):
        return {
            field["name"]: st.session_state.get(
                f"{config_name}_{field['name']}",
                field["default"],
            )
            for field in fields
        }

    for field in fields:
        values[field["name"]] = _build_field(
            field.get("type", "str"),
            field["label"],
            default=field["default"],
            key=f"{config_name}_{field['name']}",
            help=field["help"],
            min_value=field.get("min_value"),
            max_value=field.get("max_value"),
            step=field.get("step"),
        )

    return values


PROJECT_SECTIONS: list[tuple[str, list[dict[str, Any]]]] = [
    (
        "Sampling and reproducibility",
        [
            {
                "name": "random_seed",
                "label": "Random seed",
                "type": "int",
                "default": 99,
                "min_value": 0,
                "help": "Random seed used for reproducible sampling and blinding.",
            },
            {
                "name": "reliability_fraction",
                "label": "Reliability fraction",
                "type": "float",
                "default": 0.2,
                "min_value": 0.0,
                "max_value": 1.0,
                "step": 0.05,
                "help": "Fraction of samples to select for reliability workflows.",
            },
            {
                "name": "shuffle_samples",
                "label": "Shuffle samples",
                "type": "bool",
                "default": True,
                "help": "If true, shuffle sample order during selection / preparation steps.",
            },
        ],
    ),
    (
        "Transcript processing",
        [
            {
                "name": "strip_clan",
                "label": "Strip CLAN annotations",
                "type": "bool",
                "default": True,
                "help": "If true, remove CHAT markup that should not remain in plain-language text.",
            },
            {
                "name": "prefer_correction",
                "label": "Prefer correction over original",
                "type": "bool",
                "default": True,
                "help": "If true, prefer corrected forms when CHAT provides revisions.",
            },
            {
                "name": "lowercase",
                "label": "Lowercase transcript text",
                "type": "bool",
                "default": True,
                "help": "If true, lowercase transcript text during processing.",
            },
            {
                "name": "exclude_participants",
                "label": "Speakers to exclude",
                "type": "list",
                "default": ["INV"],
                "help": "Speakers to exclude from transcript-derived analyses. Enter comma- or newline-separated speaker codes.",
            },
        ],
    ),
    (
        "Coding workflow",
        [
            {
                "name": "num_bins",
                "label": "Number of sample bins",
                "type": "int",
                "default": 4,
                "min_value": 1,
                "help": "Number of bins for sample-level coding template generation.",
            },
            {
                "name": "num_coders",
                "label": "Number of coders",
                "type": "int",
                "default": 3,
                "min_value": 0,
                "help": "Number of coders to assign in coding template workflows.",
            },
            {
                "name": "stimulus_field",
                "label": "Stimulus field",
                "default": "narrative",
                "help": "Column containing the elicitation stimulus / narrative label.",
            },
            {
                "name": "automate_powers",
                "label": "Automate POWERS support",
                "type": "bool",
                "default": True,
                "help": "If true, generate automated POWERS support where available.",
            },
        ],
    ),
]


ADVANCED_SECTIONS: list[tuple[str, list[dict[str, Any]]]] = [
    (
        "Unique identifiers",
        [
            {
                "name": "sample_id_field",
                "label": "Sample ID column",
                "default": "sample_id",
                "help": "Column DIAAD should use as the sample-level identifier where supported.",
            },
            {
                "name": "utterance_id_field",
                "label": "Utterance ID column",
                "default": "utterance_id",
                "help": "Column DIAAD should use as the utterance-level identifier where supported.",
            },
        ],
    ),
    (
        "Reliability file conventions",
        [
            {
                "name": "reliability_tag",
                "label": "Reliability tag",
                "default": "_reliability",
                "help": "Tag used to identify transcription reliability files.",
            },
            {
                "name": "reliability_dirname",
                "label": "Reliability folder name",
                "default": "reliability",
                "help": "Name of the folder containing transcription reliability files.",
            },
        ],
    ),
    (
        "CU coding",
        [
            {
                "name": "cu_paradigms",
                "label": "CU paradigms",
                "type": "list",
                "default": ["SAE", "AAE"],
                "help": "Complete Utterance paradigms to include. Leave empty to use no paradigm-specific expansion.",
            },
            {
                "name": "cu_samples_file",
                "label": "CU sample-level coding file name",
                "default": "cu_coding_by_sample_long.xlsx",
                "help": "Expected CU sample-level coding file name.",
            },
            {
                "name": "cu_utts_file",
                "label": "CU utterance-level coding file name",
                "default": "cu_coding_by_utterance.xlsx",
                "help": "Expected CU utterance-level coding file name.",
            },
        ],
    ),
    (
        "Word counting",
        [
            {
                "name": "word_count_file",
                "label": "Word-count coding file name",
                "default": "word_counting.xlsx",
                "help": "Expected word-count coding file name.",
            },
            {
                "name": "word_count_field",
                "label": "Word-count column",
                "default": "word_count",
                "help": "Column containing utterance-level word counts.",
            },
            {
                "name": "wc_samples_file",
                "label": "Word-count sample summary file name",
                "default": "word_counting_by_sample.xlsx",
                "help": "Expected word-count sample-level summary file name.",
            },
        ],
    ),
    (
        "Rate analysis",
        [
            {
                "name": "speaking_time_file",
                "label": "Speaking-time file name",
                "default": "speaking_times.xlsx",
                "help": "File containing speaking-time values for rate calculations.",
            },
            {
                "name": "speaking_time_field",
                "label": "Speaking-time column",
                "default": "speaking_time",
                "help": "Column containing speaking-time values.",
            },
        ],
    ),
    (
        "Target vocabulary coverage",
        [
            {
                "name": "target_vocabulary_resource_path",
                "label": "Target vocabulary resource path",
                "default": "",
                "help": "Leave blank to use built-in resources. Set to a JSON file or directory to use custom resources.",
            },
        ],
    ),
    (
        "Project blinding",
        [
            {
                "name": "auto_blind",
                "label": "Automatically apply blinding",
                "type": "bool",
                "default": False,
                "help": "If true, DIAAD automatically blinds applicable coding files and analysis outputs.",
            },
            {
                "name": "blind_cols",
                "label": "Blind columns",
                "type": "list",
                "default": ["sample_id"],
                "help": "Columns to blind when blinding is requested. Missing columns are skipped with a warning.",
            },
        ],
    ),
    (
        "Optional metadata and codebook recovery",
        [
            {
                "name": "metadata_source",
                "label": "Metadata source",
                "default": "transcript_tables",
                "help": "Source from which metadata may be recovered for blinding / unblinding.",
            },
            {
                "name": "id_cols",
                "label": "ID columns",
                "type": "list",
                "default": ["sample_id", "utterance_id"],
                "help": "Join keys used when recovering metadata.",
            },
            {
                "name": "codebook_filename",
                "label": "Codebook filename",
                "default": "",
                "help": "Leave blank to search for *blind_codebook*.xlsx, or enter a specific codebook filename to require.",
            },
        ],
    ),
]


def _metadata_field_rows() -> list[dict[str, str]]:
    if "metadata_field_rows" not in st.session_state:
        st.session_state.metadata_field_rows = [
            {"label": "site", "values": "AC, BU, TU"},
            {"label": "test", "values": "Pre, Post, Maint"},
            {"label": "study_id", "values": r"(AC|BU|TU)\d+"},
            {
                "label": "narrative",
                "values": "CATGrandpa, BrokenWindow, RefusedUmbrella, CatRescue, BirthdayScene",
            },
        ]
    return st.session_state.metadata_field_rows


def _metadata_fields_from_rows(rows: list[dict[str, str]]) -> dict[str, str | list[str]]:
    metadata_fields: dict[str, str | list[str]] = {}
    for row in rows:
        name = (row.get("label") or "").strip()
        values = _split_values(row.get("values", ""))
        if not name:
            continue
        metadata_fields[name] = (
            values if len(values) > 1 else (values[0] if values else "")
        )
    return metadata_fields


def _build_metadata_fields_ui() -> tuple[dict[str, str | list[str]], list[str]]:
    rows = _metadata_field_rows()
    if not _section_toggle(
        "Optional metadata parsing from folder names and file names",
        key="show_project_metadata_fields",
    ):
        return _metadata_fields_from_rows(rows), []

    st.markdown(
        """
Metadata fields tell DIAAD how to recover labels from transcript paths. DIAAD
searches each file's path relative to the web input area, including any folders
and the file name. Use one row per field you want in downstream tables.

Enter multiple comma- or newline-separated values when the field has a fixed set
of choices, such as `AC, BU, TU`. Enter one regular expression when the field
should be captured by a pattern, such as `(AC|BU|TU)\\d+`. Leave this section
empty if you do not want DIAAD to extract metadata from paths.
        """
    )

    regex_errors: list[str] = []
    for i, row in enumerate(rows):
        cols = st.columns([2, 5])
        row["label"] = cols[0].text_input(
            f"Metadata field {i + 1} name",
            value=row["label"],
            key=f"metadata_field_label_{i}",
            help="Name for the metadata column DIAAD should create.",
        )
        field_name = row["label"].strip() or "metadata field"
        row["values"] = cols[1].text_area(
            f"Values or regex for {field_name}",
            value=row["values"],
            key=f"metadata_field_values_{i}",
            help=(
                "Use multiple values for a fixed choice list, or one regular "
                "expression for pattern-based matching."
            ),
        )

        values = _split_values(row["values"])
        if len(values) == 1 and values[0]:
            try:
                re.compile(values[0])
            except re.error as e:
                msg = f"Metadata field '{field_name}': invalid regex: {e}"
                regex_errors.append(msg)
                st.error(msg)

    col_add, col_remove = st.columns(2)
    if col_add.button("Add metadata field"):
        rows.append({"label": "", "values": ""})
    if col_remove.button("Remove last metadata field") and rows:
        rows.pop()

    return _metadata_fields_from_rows(rows), regex_errors


def _build_config_from_sections(
    config_name: str,
    sections: list[tuple[str, list[dict[str, Any]]]],
) -> tuple[dict[str, Any], list[str]]:
    config: dict[str, Any] = {}
    errors: list[str] = []

    for section_name, fields in sections:
        config.update(_render_fields(config_name, section_name, fields))

    return config, errors


def _build_project_ui() -> tuple[dict[str, Any], list[str]]:
    st.markdown("Most users only need to edit this file.")
    project, errors = _build_config_from_sections(
        "project",
        PROJECT_SECTIONS,
    )
    metadata_fields, metadata_errors = _build_metadata_fields_ui()
    project["metadata_fields"] = metadata_fields
    errors.extend(metadata_errors)
    return project, errors


def _build_advanced_ui() -> tuple[dict[str, Any], list[str]]:
    st.markdown("Most users can leave this file unchanged.")
    return _build_config_from_sections("advanced", ADVANCED_SECTIONS)


def build_config_ui() -> tuple[dict[str, dict], bool]:
    """Render the two-file configuration builder."""
    st.subheader("Create DIAAD Config")

    project: dict[str, Any] | None = None
    advanced: dict[str, Any] | None = None
    errors: list[str] = []

    with st.expander("project.yaml", expanded=False):
        project, project_errors = _build_project_ui()
        errors.extend(project_errors)

    with st.expander("advanced.yaml", expanded=False):
        advanced, advanced_errors = _build_advanced_ui()
        errors.extend(advanced_errors)

    configs = {
        "project": project or {},
        "advanced": advanced or {},
    }

    with st.expander("Config Preview", expanded=False):
        tabs = st.tabs(["project.yaml", "advanced.yaml"])
        for tab, name in zip(tabs, ("project", "advanced")):
            with tab:
                st.code(
                    yaml.safe_dump(configs[name], sort_keys=False, allow_unicode=True),
                    language="yaml",
                )

    valid = not errors
    if not valid:
        st.warning("Fix the errors above before using this config.")

    st.download_button(
        "Download config ZIP",
        data=_config_zip(configs),
        file_name="diaad_config.zip",
        mime="application/zip",
        disabled=not valid,
    )

    return configs, valid
