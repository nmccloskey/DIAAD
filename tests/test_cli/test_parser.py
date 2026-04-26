from __future__ import annotations

from diaad.cli.parser import build_arg_parser


def test_build_arg_parser_parses_command_and_config():
    parser = build_arg_parser()
    args = parser.parse_args(["transcripts", "tabularize", "--config", "example_config"])

    assert args.command == ["transcripts", "tabularize"]
    assert args.config == "example_config"


def test_build_arg_parser_includes_registry_help():
    parser = build_arg_parser()

    assert "Available Commands" in parser.epilog
    assert "transcripts tabularize" in parser.epilog
    assert "powers analyze" in parser.epilog
