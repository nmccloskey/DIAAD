#!/usr/bin/env python3
from diaad.main import main as main_core
from diaad.core.logger import logger
from diaad.cli.parser import build_arg_parser


def main() -> None:
    """Entry point for the DIAAD CLI."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        main_core(args)
    except Exception as e:
        logger.error("DIAAD execution failed: %s", e)
        raise


if __name__ == "__main__":
    main()
