#!/usr/bin/env python3
from .main import main as main_core
from src.diaad.core.logger import logger
from diaad.utils.cli_utils import build_arg_parser


def main():
    """Entry point for diaad CLI wrapper."""
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        main_core(args)
    except Exception as e:
        logger.error(f"DIAAD execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
