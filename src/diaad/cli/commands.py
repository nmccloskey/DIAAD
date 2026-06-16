from __future__ import annotations

# -------------------------------------------------------------
# Canonical CLI command registry
# -------------------------------------------------------------

MODULE_COMMANDS = {

    "examples": [
        "examples",
    ],

    "blinding": [
        "blinding encode",
        "blinding decode",
    ],

    "transcripts": [
        "transcripts select",
        "transcripts evaluate",
        "transcripts reselect",
        "transcripts tabularize",
        "transcripts chats",
    ],

    "templates": [
        "templates utterances",
        "templates samples",
        "templates times",
        "templates subset",
        "templates combine",
    ],

    "cus": [
        "cus files",
        "cus evaluate",
        "cus reselect",
        "cus analyze",
        "cus rates",
    ],

    "words": [
        "words files",
        "words evaluate",
        "words reselect",
        "words analyze",
        "words rates",
    ],

    "powers": [
        "powers files",
        "powers analyze",
        "powers rates",
        "powers evaluate",
        "powers reselect",
    ],

    "vocab": [
        "vocab file",
        "vocab check",
        "vocab analyze",
        "vocab rates",
    ],

    "turns": [
        "turns evaluate",
        "turns analyze",
    ],

}

VALID_COMMANDS = {
    cmd
    for commands in MODULE_COMMANDS.values()
    for cmd in commands
}

# -------------------------------------------------------------
# CLI command parsing
# -------------------------------------------------------------

def parse_cli_commands(command_arg: list[str] | str, logger=None) -> list[str]:
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
    valid: list[str] = []

    for cmd in raw_commands:
        if cmd in VALID_COMMANDS:
            valid.append(cmd)
        elif logger:
            logger.warning("Command %r not recognized - skipping", cmd)

    return valid
