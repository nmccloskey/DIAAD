from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

import streamlit as st

from diaad.cli.commands import MODULE_COMMANDS
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
    }
    assert grouped["examples"] == ["examples"]
    assert grouped["blinding"] == ["blinding encode", "blinding decode"]
    assert "transcripts chats" in grouped["transcripts"]


def test_web_command_menu_uses_readable_module_labels():
    assert streamlit_app.MODULE_LABELS["cus"] == "Complete Utterances"
    assert streamlit_app.MODULE_LABELS["words"] == "Word Counting"
    assert streamlit_app.MODULE_LABELS["turns"] == "Digital Conversational Turns"


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
