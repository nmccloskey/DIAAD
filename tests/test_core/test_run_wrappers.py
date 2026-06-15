from __future__ import annotations

from types import SimpleNamespace

import pytest

import diaad.core.run_wrappers as run_wrappers
import diaad.coding.convo_turns.analysis as turns_analysis
from diaad.metadata.discovery import MultipleFilesFoundError


def test_turns_analyze_wrapper_falls_back_to_transcript_tables(monkeypatch):
    events = []
    kwargs = {
        "input_dir": "input",
        "output_dir": "output",
        "sample_id_field": "sample_id",
        "dct_coding_filename": "conversation_turns.xlsx",
        "transcript_table_filename": "transcript_tables.xlsx",
        "exclude_speakers": ["INV"],
    }

    def fake_analyze_digital_convo_turns(**call_kwargs):
        events.append(("analyze", call_kwargs))
        if not call_kwargs.get("use_transcript_tables"):
            raise FileNotFoundError("missing DCT coding workbook")
        return "transcript-result"

    monkeypatch.setattr(
        turns_analysis,
        "analyze_digital_convo_turns",
        fake_analyze_digital_convo_turns,
    )

    ctx = SimpleNamespace(
        kwargs_digital_convo_turns=lambda: dict(kwargs),
        ensure_transcript_tables=lambda: events.append(("ensure", {})),
    )

    result = run_wrappers.run_analyze_digital_convo_turns(ctx)

    assert result == "transcript-result"
    assert events == [
        ("analyze", kwargs),
        ("ensure", {}),
        ("analyze", {**kwargs, "use_transcript_tables": True}),
    ]


def test_turns_analyze_wrapper_does_not_fallback_on_duplicate_dct(monkeypatch):
    events = []

    def fake_analyze_digital_convo_turns(**call_kwargs):
        events.append(("analyze", call_kwargs))
        raise MultipleFilesFoundError("duplicate DCT coding workbooks")

    monkeypatch.setattr(
        turns_analysis,
        "analyze_digital_convo_turns",
        fake_analyze_digital_convo_turns,
    )

    ctx = SimpleNamespace(
        kwargs_digital_convo_turns=lambda: {"input_dir": "input", "output_dir": "output"},
        ensure_transcript_tables=lambda: events.append(("ensure", {})),
    )

    with pytest.raises(MultipleFilesFoundError):
        run_wrappers.run_analyze_digital_convo_turns(ctx)

    assert events == [
        ("analyze", {"input_dir": "input", "output_dir": "output"}),
    ]
