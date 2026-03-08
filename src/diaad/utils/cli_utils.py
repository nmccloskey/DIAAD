import argparse


# -------------------------------------------------------------
# Canonical CLI command registry
# -------------------------------------------------------------
MODULE_COMMANDS = {
    "transcripts": [
        "transcripts select",
        "transcripts evaluate",
        "transcripts reselect",
        "transcripts tabularize",
    ],
    "cus": [
        "cus make",
        "cus evaluate",
        "cus reselect",
        "cus analyze",
        "cus summarize",
    ],
    "words": [
        "words make",
        "words evaluate",
        "words reselect",
    ],
    "corelex": [
        "corelex analyze",
    ],
}

VALID_COMMANDS = {
    cmd
    for commands in MODULE_COMMANDS.values()
    for cmd in commands
}


# -------------------------------------------------------------
# CLI parser
# -------------------------------------------------------------
def build_arg_parser():
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
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)",
    )

    help_lines = ["\nAvailable Commands:\n"]
    for module, commands in MODULE_COMMANDS.items():
        help_lines.append(f"  {module}:")
        for cmd in commands:
            help_lines.append(f"    - {cmd}")

    parser.epilog = "\n".join(help_lines)
    return parser


# -------------------------------------------------------------
# CLI command parsing
# -------------------------------------------------------------
def parse_cli_commands(command_arg, logger=None):
    """
    Parse and validate canonical CLI commands.

    Parameters
    ----------
    command_arg : list[str] | str
        Raw argparse command value.
    logger : logging.Logger | None
        Optional logger for warnings.

    Returns
    -------
    list[str]
        Valid canonical commands in requested order.
    """
    if isinstance(command_arg, list):
        command_arg = " ".join(command_arg)

    raw_commands = [c.strip().lower() for c in command_arg.split(",") if c.strip()]
    valid = []

    for cmd in raw_commands:
        if cmd in VALID_COMMANDS:
            valid.append(cmd)
        else:
            if logger:
                logger.warning(f"Command {cmd!r} not recognized - skipping")

    return valid
