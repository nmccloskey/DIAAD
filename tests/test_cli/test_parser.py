from __future__ import annotations

import pytest

from diaad import __version__
from diaad.cli.parser import build_arg_parser


def test_parser_leaves_config_unset_when_omitted() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(["powers", "evaluate"])

    assert args.config is None


def test_parser_accepts_batch_config_overrides() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "powers",
            "evaluate",
            "--config",
            "config",
            "--input-dir",
            "data/input/siteA",
            "--output-dir",
            "data/output/siteA",
            "--set",
            "powers_coding_filename=siteA_powers.xlsx",
            "--set",
            "powers_reliability_filename=siteA_powers_rel.xlsx",
            "--dry-run-config",
        ]
    )

    assert args.command == ["powers", "evaluate"]
    assert args.config == "config"
    assert args.input_dir == "data/input/siteA"
    assert args.output_dir == "data/output/siteA"
    assert args.set_values == [
        "powers_coding_filename=siteA_powers.xlsx",
        "powers_reliability_filename=siteA_powers_rel.xlsx",
    ]
    assert args.dry_run_config is True


def test_parser_rejects_removed_examples_files_option() -> None:
    parser = build_arg_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["examples", "--files", "diaad_data/output"])


def test_parser_accepts_repeatable_examples_for_command_option() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "examples",
            "--for-command",
            "cus analyze",
            "--for-command",
            "cus evaluate",
        ]
    )

    assert args.command == ["examples"]
    assert args.example_commands == ["cus analyze", "cus evaluate"]


def test_parser_version_option_exits_without_command(capsys) -> None:
    parser = build_arg_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"DIAAD {__version__}"


def test_parser_help_lists_streamlit_launcher() -> None:
    parser = build_arg_parser()

    help_text = parser.format_help()

    assert "--version" in help_text
    assert "diaad streamlit" in help_text
    assert "    - streamlit" in help_text
