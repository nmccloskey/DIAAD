from __future__ import annotations

from diaad.cli.commands import MODULE_COMMANDS, VALID_COMMANDS, parse_cli_commands


class DummyLogger:
    def __init__(self) -> None:
        self.warnings = []

    def warning(self, message, *args) -> None:
        self.warnings.append(message % args)


def test_valid_commands_matches_module_registry():
    flattened = {cmd for commands in MODULE_COMMANDS.values() for cmd in commands}
    assert VALID_COMMANDS == flattened


def test_parse_cli_commands_normalizes_and_filters_unknowns():
    logger = DummyLogger()

    commands = parse_cli_commands(
        "CUS FILES, words analyze, not real, turns evaluate",
        logger=logger,
    )

    assert commands == ["cus files", "words analyze", "turns evaluate"]
    assert logger.warnings == ["Command 'not real' not recognized - skipping"]


def test_parse_cli_commands_accepts_argparse_style_list():
    commands = parse_cli_commands(["transcripts", "tabularize,", "cus", "files"])
    assert commands == ["transcripts tabularize", "cus files"]
