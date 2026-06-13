from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from diaad.cli.commands import MODULE_COMMANDS
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
