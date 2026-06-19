import argparse

from diaad import __version__
from diaad.cli.commands import MODULE_COMMANDS


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the DIAAD CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            f"DIAAD version {__version__} command-line interface.\n\n"
            "Commands are organized by module and follow the pattern:\n"
            "  diaad <module> <action>\n\n"
            "Examples:\n"
            "  diaad transcripts reselect\n"
            "  diaad transcripts tabularize\n"
            "  diaad cus analyze\n"
            "  diaad transcripts tabularize, cus files, words files\n"
            "  diaad streamlit\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "command",
        nargs="+",
        help=(
            "Command to run. For multiple commands, separate them with commas.\n"
            'Examples: "transcripts tabularize" or '
            '"transcripts tabularize, cus files"'
        ),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"DIAAD {__version__}",
        help="Print the installed version of DIAAD.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a split config directory or nested YAML config file. "
            "If omitted, DIAAD uses ./config when present, otherwise packaged defaults."
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override project input_dir for this run.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override project output_dir for this run.",
    )

    parser.add_argument(
        "--set",
        dest="set_values",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override a DIAAD config value. May be repeated.",
    )

    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Print the resolved configuration and exit without running commands.",
    )

    parser.add_argument(
        "--dry-run-config-out",
        type=str,
        default=None,
        help="Optional path to save --dry-run-config output as JSON or YAML.",
    )

    parser.add_argument(
        "--dry-run-config-format",
        choices=["json", "yaml"],
        default="json",
        help="Format for --dry-run-config stdout output (default: json).",
    )

    parser.add_argument(
        "--render-docs",
        action="store_true",
        help="For 'diaad examples': regenerate packaged example I/O markdown.",
    )

    parser.add_argument(
        "--for-command",
        dest="example_commands",
        action="append",
        default=None,
        metavar="COMMAND",
        help=(
            "For 'diaad examples': generate example files for a canonical DIAAD "
            "command. May be repeated."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="For 'diaad examples': overwrite existing generated example files.",
    )

    help_lines = ["\nAvailable Commands:\n"]
    for module, commands in MODULE_COMMANDS.items():
        help_lines.append(f"  {module}:")
        for cmd in commands:
            help_lines.append(f"    - {cmd}")
    help_lines.extend(
        [
            "  web:",
            "    - streamlit",
        ]
    )

    parser.epilog = "\n".join(help_lines)
    return parser
