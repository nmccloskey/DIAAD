import argparse

from diaad.cli.commands import MODULE_COMMANDS


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the DIAAD CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "DIAAD command-line interface.\n\n"
            "Commands are organized by module and follow the pattern:\n"
            "  diaad <module> <action>\n\n"
            "Examples:\n"
            "  diaad transcripts reselect\n"
            "  diaad transcripts tabularize\n"
            "  diaad cus analyze\n"
            "  diaad transcripts tabularize, cus make, words make\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "command",
        nargs="+",
        help=(
            "Command to run. For multiple commands, separate them with commas.\n"
            'Examples: "transcripts tabularize" or '
            '"transcripts tabularize, cus make"'
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Path to the configuration directory (default: config)",
    )

    help_lines = ["\nAvailable Commands:\n"]
    for module, commands in MODULE_COMMANDS.items():
        help_lines.append(f"  {module}:")
        for cmd in commands:
            help_lines.append(f"    - {cmd}")

    parser.epilog = "\n".join(help_lines)
    return parser
