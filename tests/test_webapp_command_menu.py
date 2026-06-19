from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

import streamlit as st

from psair.examples import ManualSource, build_composed_manual

from diaad.cli.commands import MODULE_COMMANDS
from diaad.examples import generate as generate_module
from diaad.webapp import config_builder
from diaad.webapp import streamlit_app


def test_web_command_menu_represents_cli_registry():
    grouped = streamlit_app._command_options_by_module()
    flattened = [
        command
        for module_commands in grouped.values()
        for command in module_commands
    ]

    assert set(flattened) == {
        command
        for module_commands in MODULE_COMMANDS.values()
        for command in module_commands
    } - {"examples"}
    assert grouped["examples"] == []
    assert grouped["blinding"] == ["blinding encode", "blinding decode"]
    assert "transcripts chats" in grouped["transcripts"]
    assert "examples" not in flattened


def test_web_command_menu_uses_readable_module_labels():
    assert streamlit_app.MODULE_LABELS["cus"] == "Complete Utterances"
    assert streamlit_app.MODULE_LABELS["words"] == "Word Counting"
    assert streamlit_app.MODULE_LABELS["turns"] == "Digital Conversational Turns"


def test_web_command_helpers_cover_selectable_commands():
    selectable_commands = set(streamlit_app._command_options())

    assert set(streamlit_app.COMMAND_HELP) == selectable_commands

    for command, help_text in streamlit_app.COMMAND_HELP.items():
        assert "Select this canonical DIAAD command" not in help_text
        assert help_text.endswith(".")
        assert "For full details, see docs\\manual\\" in help_text

        section_path = help_text.removesuffix(".").split("For full details, see ")[1]
        section_dir = Path(section_path.replace("\\", "/"))
        assert (section_dir / "01_quickstart.md").exists(), command


def test_uploaded_relative_path_preserves_nested_directories():
    assert streamlit_app._safe_uploaded_relative_path(
        "site_a/session_1/sample.cha"
    ) == Path("site_a", "session_1", "sample.cha")


@pytest.mark.parametrize(
    "uploaded_name",
    [
        "../sample.cha",
        "site_a/../sample.cha",
        "/tmp/sample.cha",
        "C:/tmp/sample.cha",
    ],
)
def test_uploaded_relative_path_rejects_unsafe_paths(uploaded_name):
    with pytest.raises(ValueError):
        streamlit_app._safe_uploaded_relative_path(uploaded_name)


def test_examples_web_zip_contains_full_dataset_folder(monkeypatch):
    calls = []

    def fake_generate(destination, *, force=False, commands=None):
        destination = Path(destination)
        calls.append((destination, force, commands))
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "README.md").write_text("full", encoding="utf-8")
        return destination

    monkeypatch.setattr(generate_module, "generate_example_files", fake_generate)

    zip_buffer = streamlit_app._run_examples_web()

    with zipfile.ZipFile(zip_buffer) as zf:
        assert "example_files_full_dataset/README.md" in zf.namelist()

    assert calls == [
        (
            calls[0][0],
            True,
            None,
        )
    ]
    assert calls[0][0].name == "example_files_full_dataset"


def test_examples_web_zip_contains_command_package(monkeypatch):
    calls = []

    def fake_generate(destination, *, force=False, commands=None):
        destination = Path(destination)
        calls.append((destination, force, commands))
        package_dir = destination / "example_files_cus_analyze_cus_evaluate"
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "README.md").write_text("commands", encoding="utf-8")
        return package_dir

    monkeypatch.setattr(generate_module, "generate_example_files", fake_generate)

    zip_buffer = streamlit_app._run_examples_web(["cus analyze", "cus evaluate"])

    with zipfile.ZipFile(zip_buffer) as zf:
        assert "example_files_cus_analyze_cus_evaluate/README.md" in zf.namelist()

    assert calls == [
        (
            calls[0][0],
            True,
            ["cus analyze", "cus evaluate"],
        )
    ]
    assert calls[0][0].name == "examples"


@pytest.mark.parametrize(
    ("command", "expected_outputs"),
    [
        (
            "cus files",
            [
                "example_files_cus_files/example_output/cu_coding/cu_coding.xlsx",
                "example_files_cus_files/example_output/cu_coding/cu_reliability_coding.xlsx",
            ],
        ),
        (
            "words files",
            [
                "example_files_words_files/example_output/word_counts/word_counting.xlsx",
                "example_files_words_files/example_output/word_counts/word_count_reliability.xlsx",
            ],
        ),
    ],
)
def test_examples_web_files_zip_contains_outputs_under_strict_string_inference(
    command,
    expected_outputs,
):
    with pd.option_context("future.infer_string", True):
        zip_buffer = streamlit_app._run_examples_web([command])

    with zipfile.ZipFile(zip_buffer) as zf:
        names = set(zf.namelist())

    assert set(expected_outputs) <= names


def test_examples_zip_filename_uses_stable_slug():
    assert streamlit_app._examples_zip_filename(
        ["cus analyze", "cus evaluate"],
        timestamp=streamlit_app.datetime(2026, 1, 1, 0, 0),
    ) == "diaad_example_files_cus_analyze_cus_evaluate_260101_0000.zip"


def test_restore_default_config_ui_state_keeps_unrelated_state():
    st.session_state["project_random_seed"] = 123
    st.session_state["metadata_field_rows"] = [{"label": "custom", "values": "x"}]
    st.session_state["metadata_field_label_0"] = "custom"
    st.session_state["unrelated"] = "keep"

    config_builder.restore_default_config_ui_state()

    assert "project_random_seed" not in st.session_state
    assert "metadata_field_rows" not in st.session_state
    assert "metadata_field_label_0" not in st.session_state
    assert st.session_state["unrelated"] == "keep"

    del st.session_state["unrelated"]


def test_web_config_builder_defaults_come_from_canonical_config():
    defaults = config_builder.CANONICAL_CONFIG_DEFAULTS
    represented_fields = {
        "project": [
            field["name"]
            for _, fields in config_builder.PROJECT_SECTIONS
            for field in fields
        ],
        "advanced": [
            field["name"]
            for _, fields in config_builder.ADVANCED_SECTIONS
            for field in fields
        ],
    }

    assert set(represented_fields["project"]) == set(defaults["project"]) - {
        "input_dir",
        "output_dir",
        "metadata_fields",
    }
    assert set(represented_fields["advanced"]) == set(defaults["advanced"])

    for section_name, sections in (
        ("project", config_builder.PROJECT_SECTIONS),
        ("advanced", config_builder.ADVANCED_SECTIONS),
    ):
        for _, fields in sections:
            for field in fields:
                assert field["default"] == defaults[section_name][field["name"]]


def test_web_config_builder_metadata_defaults_to_canonical_empty_mapping():
    st.session_state.pop("metadata_field_rows", None)

    assert config_builder._metadata_field_rows() == []
    assert (
        config_builder._metadata_fields_from_rows(config_builder._metadata_field_rows())
        == config_builder.CANONICAL_CONFIG_DEFAULTS["project"]["metadata_fields"]
    )

    st.session_state.pop("metadata_field_rows", None)


def test_web_config_builder_metadata_row_controls_update_state_immediately():
    st.session_state.pop("metadata_field_rows", None)

    config_builder._add_metadata_field_row()
    assert config_builder._metadata_field_rows() == [{"label": "", "values": ""}]

    config_builder._add_metadata_field_row()
    assert config_builder._metadata_field_rows() == [
        {"label": "", "values": ""},
        {"label": "", "values": ""},
    ]

    st.session_state["metadata_field_label_1"] = "old"
    st.session_state["metadata_field_values_1"] = "AC, BU"
    config_builder._remove_metadata_field_row()

    assert config_builder._metadata_field_rows() == [{"label": "", "values": ""}]
    assert "metadata_field_label_1" not in st.session_state
    assert "metadata_field_values_1" not in st.session_state

    config_builder._remove_metadata_field_row()
    config_builder._remove_metadata_field_row()
    assert config_builder._metadata_field_rows() == []

    st.session_state.pop("metadata_field_rows", None)


def test_manual_sources_include_authored_and_generated_roots():
    repo_root = Path(streamlit_app.__file__).resolve().parents[3]
    package_root = Path(streamlit_app.__file__).resolve().parents[1]

    sources = streamlit_app._manual_sources(
        repo_root=repo_root,
        package_root=package_root,
    )

    assert sources == [
        {
            "root": repo_root / "docs/manual",
            "name": "diaad_manual",
            "source_manual": "authored",
            "role": "authored",
        },
        {
            "root": streamlit_app.get_example_io_docs_path(),
            "name": "diaad_example_io",
            "source_manual": "generated_example_io",
            "role": "generated",
        },
    ]


def test_render_manual_uses_one_composed_manual_viewer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        streamlit_app,
        "render_manual_ui",
        lambda **kwargs: calls.append(kwargs),
    )

    streamlit_app._render_manual()

    assert len(calls) == 1
    call = calls[0]
    assert call["manual_rel_dir"] == "docs/manual"
    assert call["expander_label"] == "Show / Hide DIAAD Manual Menu"
    assert call["ui_key"] == "diaad_manual"
    assert call["compose_infer_from_paths"] is True
    assert call["compose_path_module_aliases"] == streamlit_app._MANUAL_PATH_MODULE_ALIASES
    assert call["compose_unmatched_policy"] == "generated_root"
    assert [source["role"] for source in call["manual_sources"]] == [
        "authored",
        "generated",
    ]


def test_composed_manual_threads_tabularize_example_io():
    repo_root = Path(streamlit_app.__file__).resolve().parents[3]
    package_root = Path(streamlit_app.__file__).resolve().parents[1]
    source_dicts = streamlit_app._manual_sources(
        repo_root=repo_root,
        package_root=package_root,
    )
    sources = [
        ManualSource(
            root=source["root"],
            name=str(source["name"]),
            source_manual=str(source["source_manual"]),
            role=str(source["role"]),
        )
        for source in source_dicts
    ]

    composed = build_composed_manual(
        sources,
        infer_from_paths=True,
        path_module_aliases=streamlit_app._MANUAL_PATH_MODULE_ALIASES,
        unmatched_policy="generated_root",
    )

    rel = "04_modules/01_transcripts/05_commands/01_tabularize/05_example_io.md"
    assert rel in composed.flat
    assert composed.flat[rel].abs_path == (
        streamlit_app.get_example_io_docs_path() / "transcripts" / "tabularize.md"
    ).resolve()
    assert composed.flat[rel].title == "Example I/O"
    assert composed.flat[rel].text.startswith("# Transcript Tabularization Example")

def _build_diaad_composed_manual():
    repo_root = Path(streamlit_app.__file__).resolve().parents[3]
    package_root = Path(streamlit_app.__file__).resolve().parents[1]
    source_dicts = streamlit_app._manual_sources(
        repo_root=repo_root,
        package_root=package_root,
    )
    sources = [
        ManualSource(
            root=source["root"],
            name=str(source["name"]),
            source_manual=str(source["source_manual"]),
            role=str(source["role"]),
        )
        for source in source_dicts
    ]
    return build_composed_manual(
        sources,
        infer_from_paths=True,
        path_module_aliases=streamlit_app._MANUAL_PATH_MODULE_ALIASES,
        unmatched_policy="generated_root",
    )


def test_composed_manual_threads_aliased_module_example_io_without_generated_branch():
    composed = _build_diaad_composed_manual()

    rel = "04_modules/03_complete_utterances/05_commands/01_files/05_example_io.md"
    assert rel in composed.flat
    assert composed.flat[rel].abs_path == (
        streamlit_app.get_example_io_docs_path() / "cus" / "files.md"
    ).resolve()
    assert not any(path.startswith("generated/") for path in composed.flat)
    assert not any("No authored anchor" in item for item in composed.diagnostics)


def test_composed_manual_threads_full_example_dataset_overview():
    composed = _build_diaad_composed_manual()

    rel = "04_modules/09_examples/05_commands/01_examples/05_example_io.md"
    assert rel in composed.flat
    assert composed.flat[rel].abs_path == (
        streamlit_app.get_example_io_docs_path() / "01_overview.md"
    ).resolve()
    assert composed.flat[rel].title == "Example I/O"
    assert composed.flat[rel].text.startswith("# DIAAD Example I/O Manual")
