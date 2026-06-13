from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import diaad.main as main_module
from diaad.examples import generate as generate_module


class DummyLogger:
    def __init__(self) -> None:
        self.messages = []

    def info(self, *args, **kwargs) -> None:
        self.messages.append(("info", args, kwargs))

    def error(self, *args, **kwargs) -> None:
        self.messages.append(("error", args, kwargs))

    def warning(self, *args, **kwargs) -> None:
        self.messages.append(("warning", args, kwargs))


def test_main_executes_requested_commands(monkeypatch, tmp_path):
    events = []
    logger = DummyLogger()

    class FakeContext:
        def __init__(self, config_dir, project_root, start_time, **kwargs):
            self.config_dir = config_dir
            self.project_root = project_root
            self.start_time = start_time
            self.out_dir = tmp_path / "out"
            self.out_dir.mkdir()
            self.commands = []

        def set_commands(self, commands):
            self.commands = list(commands)
            events.append(("set_commands", list(commands)))

        def termination_kwargs(self):
            return {"output_dir": self.out_dir}

    def fake_dispatch(ctx):
        return {
            "cus files": lambda: events.append(("ran", ctx.commands)),
        }

    monkeypatch.setattr(main_module, "RunContext", FakeContext)
    monkeypatch.setattr(main_module, "parse_cli_commands", lambda arg, logger=None: ["cus files"])
    monkeypatch.setattr(main_module, "prepare_dispatch_prerequisites", lambda ctx, commands: events.append(("prep", list(commands))))
    monkeypatch.setattr(main_module, "build_dispatch", fake_dispatch)
    monkeypatch.setattr(main_module, "initialize_logger", lambda *a, **k: events.append(("init_logger",)))
    monkeypatch.setattr(main_module, "add_finalization_hook", lambda hook: events.append(("hook",)))
    monkeypatch.setattr(main_module, "write_start_artifacts", lambda ctx, args: events.append(("start_artifacts",)))
    monkeypatch.setattr(main_module, "terminate_logger", lambda **k: events.append(("terminate", k)))
    monkeypatch.setattr(main_module, "set_root", lambda path: events.append(("set_root", Path(path))))
    monkeypatch.setattr(main_module, "build_cli_config_overrides", lambda args: {})
    monkeypatch.setattr(main_module, "logger", logger)

    args = SimpleNamespace(command=["cus", "files"], config="config", dry_run_config=False)
    main_module.main(args)

    assert ("prep", ["cus files"]) in events
    assert ("set_commands", ["cus files"]) in events
    assert ("start_artifacts",) in events
    assert any(event[0] == "ran" for event in events)
    assert any(event[0] == "terminate" for event in events)


def test_main_returns_early_when_no_valid_commands(monkeypatch, tmp_path):
    events = []

    class FakeContext:
        def __init__(self, config_dir, project_root, start_time, **kwargs):
            self.out_dir = tmp_path / "out"
            self.out_dir.mkdir()

        def set_commands(self, commands):
            self.commands = list(commands)

        def termination_kwargs(self):
            return {"output_dir": self.out_dir}

    monkeypatch.setattr(main_module, "RunContext", FakeContext)
    monkeypatch.setattr(main_module, "parse_cli_commands", lambda arg, logger=None: [])
    monkeypatch.setattr(main_module, "initialize_logger", lambda *a, **k: None)
    monkeypatch.setattr(main_module, "add_finalization_hook", lambda hook: None)
    monkeypatch.setattr(main_module, "write_start_artifacts", lambda ctx, args: None)
    monkeypatch.setattr(main_module, "terminate_logger", lambda **k: events.append(("terminate", k)))
    monkeypatch.setattr(main_module, "set_root", lambda path: None)
    monkeypatch.setattr(main_module, "build_cli_config_overrides", lambda args: {})
    monkeypatch.setattr(main_module, "logger", DummyLogger())

    args = SimpleNamespace(command=["bad"], config="config", dry_run_config=False)
    main_module.main(args)

    assert events == [("terminate", {"output_dir": tmp_path / "out", "status": "skipped"})]


def test_main_dry_run_config_exits_before_logger_and_dispatch(monkeypatch, tmp_path):
    events = []

    class FakeContext:
        def __init__(self, config_dir, project_root, start_time, **kwargs):
            self.config_dir = config_dir
            self.project_root = project_root
            self.start_time = start_time
            self.out_dir = tmp_path / "out"
            self.commands = []
            events.append(("ctx_kwargs", kwargs))

        def set_commands(self, commands):
            self.commands = list(commands)

    monkeypatch.setattr(main_module, "RunContext", FakeContext)
    monkeypatch.setattr(main_module, "parse_cli_commands", lambda arg, logger=None: ["powers evaluate"])
    monkeypatch.setattr(main_module, "emit_dry_run_config", lambda ctx, args, commands: events.append(("dry_run", commands)))
    monkeypatch.setattr(main_module, "initialize_logger", lambda *a, **k: events.append(("init_logger",)))
    monkeypatch.setattr(main_module, "prepare_dispatch_prerequisites", lambda *a, **k: events.append(("prep",)))
    monkeypatch.setattr(main_module, "build_cli_config_overrides", lambda args: {"project.input_dir": "input/siteA"})
    monkeypatch.setattr(main_module, "set_root", lambda path: None)
    monkeypatch.setattr(main_module, "logger", DummyLogger())

    args = SimpleNamespace(command=["powers", "evaluate"], config="config", dry_run_config=True)
    main_module.main(args)

    assert ("dry_run", ["powers evaluate"]) in events
    assert ("init_logger",) not in events
    assert ("prep",) not in events
    assert events[0][1]["config_overrides"] == {"project.input_dir": "input/siteA"}
    assert events[0][1]["create_output_dir"] is False


def test_examples_command_defaults_files_to_configured_output_dir(monkeypatch, tmp_path):
    events = []

    class FakeContext:
        def __init__(self, config_dir, project_root, start_time, **kwargs):
            self.out_dir = tmp_path / "run-output"
            self.commands = []
            events.append(
                (
                    "ctx",
                    config_dir,
                    Path(project_root),
                    isinstance(start_time, datetime),
                    kwargs,
                )
            )

        def set_commands(self, commands):
            self.commands = list(commands)
            events.append(("set_commands", list(commands)))

        def termination_kwargs(self):
            return {"output_dir": self.out_dir}

    def fake_generate(destination, *, force=False):
        events.append(("generate", Path(destination), force))
        return Path(destination)

    monkeypatch.setattr(main_module, "RunContext", FakeContext)
    monkeypatch.setattr(
        main_module,
        "set_root",
        lambda path: events.append(("set_root", Path(path))),
    )
    monkeypatch.setattr(main_module, "build_cli_config_overrides", lambda args: {})
    monkeypatch.setattr(
        main_module,
        "initialize_logger",
        lambda *a, **k: events.append(("init_logger",)),
    )
    monkeypatch.setattr(main_module, "add_finalization_hook", lambda hook: events.append(("hook",)))
    monkeypatch.setattr(
        main_module,
        "write_start_artifacts",
        lambda ctx, args: events.append(("start_artifacts",)),
    )
    monkeypatch.setattr(main_module, "terminate_logger", lambda **k: events.append(("terminate", k)))
    monkeypatch.setattr(generate_module, "generate_example_files", fake_generate)

    args = SimpleNamespace(
        command=["examples"],
        config="config",
        input_dir=None,
        output_dir=None,
        set_values=None,
        render_docs=False,
        force=False,
    )

    main_module.main(args)

    assert (
        "generate",
        tmp_path / "run-output" / "example_files_full_dataset",
        False,
    ) in events
    assert events[1][0] == "ctx"
    assert events[1][4]["config_overrides"] == {}
    assert ("set_commands", ["examples"]) in events
    assert ("start_artifacts",) in events
    assert (
        "terminate",
        {"output_dir": tmp_path / "run-output", "status": "completed"},
    ) in events


def test_examples_command_honors_output_dir_override(monkeypatch, tmp_path):
    events = []
    overridden_output = tmp_path / "overridden-output"

    class FakeContext:
        def __init__(self, config_dir, project_root, start_time, **kwargs):
            assert kwargs["config_overrides"] == {
                "project.output_dir": str(overridden_output)
            }
            self.out_dir = overridden_output / "diaad_260101_0000"

        def set_commands(self, commands):
            self.commands = list(commands)

        def termination_kwargs(self):
            return {"output_dir": self.out_dir}

    def fake_generate(destination, *, force=False):
        events.append(("generate", Path(destination), force))
        return Path(destination)

    monkeypatch.setattr(main_module, "RunContext", FakeContext)
    monkeypatch.setattr(main_module, "set_root", lambda path: None)
    monkeypatch.setattr(main_module, "initialize_logger", lambda *a, **k: None)
    monkeypatch.setattr(main_module, "add_finalization_hook", lambda hook: None)
    monkeypatch.setattr(main_module, "write_start_artifacts", lambda ctx, args: None)
    monkeypatch.setattr(main_module, "terminate_logger", lambda **k: None)
    monkeypatch.setattr(generate_module, "generate_example_files", fake_generate)

    args = SimpleNamespace(
        command=["examples"],
        config=None,
        input_dir=None,
        output_dir=str(overridden_output),
        set_values=None,
        render_docs=False,
        force=True,
    )

    main_module.main(args)

    assert events == [
        (
            "generate",
            overridden_output / "diaad_260101_0000" / "example_files_full_dataset",
            True,
        )
    ]


def test_examples_command_passes_requested_example_commands(monkeypatch, tmp_path):
    events = []

    class FakeContext:
        def __init__(self, config_dir, project_root, start_time, **kwargs):
            self.out_dir = tmp_path / "run-output"

        def set_commands(self, commands):
            self.commands = list(commands)

        def termination_kwargs(self):
            return {"output_dir": self.out_dir}

    def fake_generate(destination, *, force=False, commands=None):
        events.append(("generate", Path(destination), force, commands))
        return Path(destination) / "example_files_cus_analyze_cus_evaluate"

    monkeypatch.setattr(main_module, "RunContext", FakeContext)
    monkeypatch.setattr(main_module, "set_root", lambda path: None)
    monkeypatch.setattr(main_module, "build_cli_config_overrides", lambda args: {})
    monkeypatch.setattr(main_module, "initialize_logger", lambda *a, **k: None)
    monkeypatch.setattr(main_module, "add_finalization_hook", lambda hook: None)
    monkeypatch.setattr(main_module, "write_start_artifacts", lambda ctx, args: None)
    monkeypatch.setattr(main_module, "terminate_logger", lambda **k: None)
    monkeypatch.setattr(generate_module, "generate_example_files", fake_generate)

    args = SimpleNamespace(
        command=["examples"],
        config=None,
        input_dir=None,
        output_dir=None,
        set_values=None,
        render_docs=False,
        example_commands=["cus analyze", "cus evaluate"],
        force=True,
    )

    main_module.main(args)

    assert events == [
        (
            "generate",
            tmp_path / "run-output",
            True,
            ["cus analyze", "cus evaluate"],
        )
    ]


def test_examples_options_are_rejected_for_non_examples_commands():
    args = SimpleNamespace(
        command=["cus", "analyze"],
        render_docs=False,
        example_commands=["cus analyze"],
        force=False,
    )

    with pytest.raises(ValueError, match="--for-command"):
        main_module.main(args)
