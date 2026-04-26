from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import diaad.main as main_module


class DummyLogger:
    def __init__(self) -> None:
        self.messages = []

    def info(self, *args, **kwargs) -> None:
        self.messages.append(("info", args, kwargs))

    def error(self, *args, **kwargs) -> None:
        self.messages.append(("error", args, kwargs))


def test_main_executes_requested_commands(monkeypatch, tmp_path):
    events = []
    logger = DummyLogger()

    class FakeContext:
        def __init__(self, config_dir, project_root, start_time):
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
    monkeypatch.setattr(main_module, "terminate_logger", lambda **k: events.append(("terminate", k)))
    monkeypatch.setattr(main_module, "set_root", lambda path: events.append(("set_root", Path(path))))
    monkeypatch.setattr(main_module, "logger", logger)

    args = SimpleNamespace(command=["cus", "files"], config="config")
    main_module.main(args)

    assert ("prep", ["cus files"]) in events
    assert ("set_commands", ["cus files"]) in events
    assert any(event[0] == "ran" for event in events)
    assert any(event[0] == "terminate" for event in events)


def test_main_returns_early_when_no_valid_commands(monkeypatch, tmp_path):
    events = []

    class FakeContext:
        def __init__(self, config_dir, project_root, start_time):
            self.out_dir = tmp_path / "out"
            self.out_dir.mkdir()

        def termination_kwargs(self):
            return {"output_dir": self.out_dir}

    monkeypatch.setattr(main_module, "RunContext", FakeContext)
    monkeypatch.setattr(main_module, "parse_cli_commands", lambda arg, logger=None: [])
    monkeypatch.setattr(main_module, "initialize_logger", lambda *a, **k: None)
    monkeypatch.setattr(main_module, "terminate_logger", lambda **k: events.append(("terminate", k)))
    monkeypatch.setattr(main_module, "set_root", lambda path: None)
    monkeypatch.setattr(main_module, "logger", DummyLogger())

    args = SimpleNamespace(command=["bad"], config="config")
    main_module.main(args)

    assert events == [("terminate", {"output_dir": tmp_path / "out"})]
