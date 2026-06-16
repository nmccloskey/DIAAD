from __future__ import annotations

from types import SimpleNamespace

import diaad.cli.dispatch as dispatch_module


def test_command_requirement_helpers():
    assert dispatch_module.commands_require_chats(["transcripts tabularize"])
    assert not dispatch_module.commands_require_chats(["cus analyze"])
    assert dispatch_module.commands_require_transcript_tables(["templates samples"])
    assert not dispatch_module.commands_require_transcript_tables(["templates subset"])
    assert not dispatch_module.commands_require_transcript_tables(["words evaluate"])


def test_prepare_dispatch_prerequisites_loads_needed_inputs():
    events = []
    ctx = SimpleNamespace(
        load_chats=lambda: events.append("load_chats"),
        ensure_transcript_tables=lambda: events.append("ensure_transcript_tables"),
    )

    dispatch_module.prepare_dispatch_prerequisites(
        ctx,
        ["transcripts select", "templates utterances"],
    )

    assert events == ["load_chats", "ensure_transcript_tables"]


def test_prepare_dispatch_prerequisites_skips_transcript_creation_when_tabularizing():
    events = []
    ctx = SimpleNamespace(
        load_chats=lambda: events.append("load_chats"),
        ensure_transcript_tables=lambda: events.append("ensure_transcript_tables"),
    )

    dispatch_module.prepare_dispatch_prerequisites(
        ctx,
        ["transcripts tabularize", "templates utterances"],
    )

    assert events == ["load_chats"]


def test_build_dispatch_invokes_wrapper_with_context(monkeypatch):
    ctx = object()
    seen = []
    monkeypatch.setattr(dispatch_module, "run_make_cu_coding_files", lambda value: seen.append(value))

    dispatch = dispatch_module.build_dispatch(ctx)
    dispatch["cus files"]()

    assert seen == [ctx]


def test_build_dispatch_includes_blinding_commands(monkeypatch):
    ctx = object()
    seen = []
    monkeypatch.setattr(dispatch_module, "run_encode_blinding", lambda value: seen.append(("encode", value)))
    monkeypatch.setattr(dispatch_module, "run_decode_blinding", lambda value: seen.append(("decode", value)))

    dispatch = dispatch_module.build_dispatch(ctx)
    dispatch["blinding encode"]()
    dispatch["blinding decode"]()

    assert seen == [("encode", ctx), ("decode", ctx)]


def test_build_dispatch_includes_template_subset(monkeypatch):
    ctx = object()
    seen = []
    monkeypatch.setattr(
        dispatch_module,
        "run_make_sample_subset",
        lambda value: seen.append(value),
    )

    dispatch = dispatch_module.build_dispatch(ctx)
    dispatch["templates subset"]()

    assert seen == [ctx]


def test_build_dispatch_includes_template_combine(monkeypatch):
    ctx = object()
    seen = []
    monkeypatch.setattr(
        dispatch_module,
        "run_make_combined_template",
        lambda value: seen.append(value),
    )

    dispatch = dispatch_module.build_dispatch(ctx)
    dispatch["templates combine"]()

    assert seen == [ctx]
