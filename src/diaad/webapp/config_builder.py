from __future__ import annotations

import re
import zipfile
from io import BytesIO

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
        for name in ("project", "tiers", "blinding"):
            zf.writestr(
                f"{name}.yaml",
                yaml.safe_dump(configs[name], sort_keys=False, allow_unicode=True),
            )
    buffer.seek(0)
    return buffer


def _build_project_ui() -> dict:
    st.subheader("Project")

    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        value=99,
        step=1,
        help="Used for deterministic selections and shuffling.",
    )
    reliability_fraction = st.number_input(
        "Reliability fraction",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )
    shuffle_samples = st.checkbox("Shuffle samples", value=True)

    st.markdown("**CHAT processing**")
    strip_clan = st.checkbox("Strip CLAN annotations", value=True)
    prefer_correction = st.checkbox("Prefer correction over original", value=True)
    lowercase = st.checkbox("Convert text to lowercase", value=True)

    st.markdown("**Coding defaults**")
    num_bins = st.number_input("Number of sample bins", min_value=1, value=4, step=1)
    num_coders = st.number_input("Number of coders", min_value=0, value=3, step=1)
    stimulus_field = st.text_input("Stimulus field", value="narrative")
    exclude_participants = _split_values(
        st.text_input("Exclude participants", value="INV")
    )

    cu_paradigms = _split_values(
        st.text_input("CU paradigms", value="SAE, AAE")
    )

    st.markdown("**Input filenames**")
    cu_samples_file = st.text_input(
        "CU sample summary file",
        value="cu_coding_by_sample_long.xlsx",
    )
    cu_utts_file = st.text_input(
        "CU utterance file",
        value="cu_coding_by_utterance.xlsx",
    )
    word_count_file = st.text_input("Word-count coding file", value="word_counting.xlsx")
    word_count_field = st.text_input("Word-count field", value="word_count")
    wc_samples_file = st.text_input(
        "Word-count sample summary file",
        value="word_counting_by_sample.xlsx",
    )
    speaking_time_file = st.text_input("Speaking-time file", value="speaking_times.xlsx")
    speaking_time_field = st.text_input("Speaking-time field", value="speaking_time")

    st.markdown("**Target vocabulary coverage**")
    target_vocabulary_resource_path = st.text_input(
        "Target vocabulary resource path",
        value="",
        help="Leave blank for bundled resources. Use a JSON file or directory path for custom resources.",
    )

    st.markdown("**POWERS**")
    automate_powers = st.checkbox("Automate POWERS features", value=True)
    just_c2_powers = st.checkbox("Analyze only C2 POWERS fields", value=False)

    return {
        # The web app writes uploaded files into these temp-root-relative paths.
        "input_dir": "input",
        "output_dir": "output",
        "random_seed": int(random_seed),
        "reliability_fraction": float(reliability_fraction),
        "shuffle_samples": bool(shuffle_samples),
        "strip_clan": bool(strip_clan),
        "prefer_correction": bool(prefer_correction),
        "lowercase": bool(lowercase),
        "reliability_tag": "_reliability",
        "reliability_dirname": "reliability",
        "exclude_participants": exclude_participants,
        "num_bins": int(num_bins),
        "num_coders": int(num_coders),
        "stimulus_field": stimulus_field,
        "target_vocabulary_resource_path": target_vocabulary_resource_path,
        "cu_paradigms": cu_paradigms,
        "cu_samples_file": cu_samples_file,
        "cu_utts_file": cu_utts_file,
        "word_count_file": word_count_file,
        "word_count_field": word_count_field,
        "wc_samples_file": wc_samples_file,
        "automate_powers": bool(automate_powers),
        "just_c2_powers": bool(just_c2_powers),
        "speaking_time_file": speaking_time_file,
        "speaking_time_field": speaking_time_field,
    }


def _build_tiers_ui() -> tuple[dict, list[str]]:
    st.subheader("Tiers")

    with st.expander("Tier entry instructions", expanded=False):
        st.markdown(
            """
Use one row per filename tier.

Multiple comma- or newline-separated values are treated as literal choices.
A single value is treated as a regular expression.
            """
        )

    if "tiers" not in st.session_state:
        st.session_state.tiers = [
            {"label": "site", "values": "AC, BU, TU"},
            {"label": "test", "values": "Pre, Post, Maint"},
            {"label": "study_id", "values": r"(AC|BU|TU)\d+"},
            {"label": "narrative", "values": "BrokenWindow, RefusedUmbrella, CatRescue"},
        ]

    regex_errors: list[str] = []
    for i, tier in enumerate(st.session_state.tiers):
        cols = st.columns([2, 5])
        tier["label"] = cols[0].text_input(
            f"Tier {i + 1} name",
            value=tier["label"],
            key=f"tier_label_{i}",
        )
        tier["values"] = cols[1].text_area(
            f"Values or regex for {tier['label'] or 'tier'}",
            value=tier["values"],
            key=f"tier_values_{i}",
        )

        values = _split_values(tier["values"])
        if len(values) == 1 and values[0]:
            try:
                re.compile(values[0])
            except re.error as e:
                msg = f"Tier '{tier['label']}': invalid regex: {e}"
                regex_errors.append(msg)
                st.error(msg)

    col_add, col_remove = st.columns(2)
    if col_add.button("Add tier"):
        st.session_state.tiers.append({"label": "", "values": ""})
    if col_remove.button("Remove last tier") and st.session_state.tiers:
        st.session_state.tiers.pop()

    tiers: dict[str, str | list[str]] = {}
    for tier in st.session_state.tiers:
        name = (tier.get("label") or "").strip()
        values = _split_values(tier.get("values", ""))
        if not name:
            continue
        tiers[name] = values if len(values) > 1 else (values[0] if values else "")

    return {"tiers": tiers}, regex_errors


def _build_blinding_ui() -> dict:
    st.subheader("Blinding")

    blind_files = st.checkbox("Blind manual coding files", value=True)
    blind_analysis = st.checkbox("Blind analysis outputs", value=True)
    metadata_source = st.text_input("Metadata source", value="transcript_tables")

    coding_blind_cols = _split_values(
        st.text_area("Coding blind columns", value="sample_id")
    )
    analysis_blind_cols = _split_values(
        st.text_area("Analysis blind columns", value="sample_id, site, test")
    )
    id_cols = _split_values(
        st.text_area("ID columns", value="sample_id, utterance_id")
    )

    return {
        "blind_files": bool(blind_files),
        "blind_analysis": bool(blind_analysis),
        "metadata_source": metadata_source,
        "coding_blind_cols": coding_blind_cols,
        "analysis_blind_cols": analysis_blind_cols,
        "id_cols": id_cols,
    }


def build_config_ui() -> tuple[dict[str, dict], bool]:
    """Render the three-file configuration builder."""
    st.subheader("Create DIAAD Config")

    project = _build_project_ui()
    tiers, regex_errors = _build_tiers_ui()
    blinding = _build_blinding_ui()

    configs = {
        "project": project,
        "tiers": tiers,
        "blinding": blinding,
    }

    st.subheader("Config Preview")
    tabs = st.tabs(["project.yaml", "tiers.yaml", "blinding.yaml"])
    for tab, name in zip(tabs, ("project", "tiers", "blinding")):
        with tab:
            st.code(
                yaml.safe_dump(configs[name], sort_keys=False, allow_unicode=True),
                language="yaml",
            )

    valid = not regex_errors
    if not valid:
        st.warning("Fix the regex errors above before using this config.")

    st.download_button(
        "Download config ZIP",
        data=_config_zip(configs),
        file_name="diaad_config.zip",
        mime="application/zip",
        disabled=not valid,
    )

    return configs, valid
