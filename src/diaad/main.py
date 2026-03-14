#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from diaad import __version__
from src.diaad.core.run_context import RunContext
from diaad.utils.dispatch import build_dispatch, prepare_dispatch_prerequisites
from diaad.utils.cli_utils import build_arg_parser, parse_cli_commands
from src.diaad.core.logger import (
    initialize_logger,
    logger,
    set_root,
    terminate_logger,
)


def main(args) -> None:
    """Parse CLI arguments and execute the requested DIAAD commands."""
    ctx = None

    try:
        start_time = datetime.now()
        set_root(Path.cwd())

        print("ARGS.CONFIG:", args.config)

        ctx = RunContext(
            config_dir=args.config or "config",
            start_time=start_time,
        )

        # -----------------------------------------------------------------
        # Initialize logger once output folder is ready
        # -----------------------------------------------------------------
        initialize_logger(
            start_time,
            ctx.out_dir,
            program_name="DIAAD",
            version=__version__,
        )
        logger.info("Logger initialized and early logs flushed.")

        # ---------------------------------------------------------
        # Parse commands
        # ---------------------------------------------------------
        commands = parse_cli_commands(args.command, logger=logger)

        if not commands:
            logger.error("No valid commands recognized - exiting.")
            return

        ctx.set_commands(commands)
        logger.info("Executing command(s): %s", ", ".join(commands))

        # ---------------------------------------------------------
        # Prepare prerequisites and dispatch
        # ---------------------------------------------------------
        prepare_dispatch_prerequisites(ctx, commands)
        dispatch = build_dispatch(ctx)

        # ---------------------------------------------------------
        # Execute all requested commands
        # ---------------------------------------------------------
        executed = []
        for cmd in commands:
            func = dispatch.get(cmd)
            if func is None:
                logger.error("Unknown command: %s", cmd)
                continue

            func()
            executed.append(cmd)

        if executed:
            logger.info("Completed: %s", ", ".join(executed))

    except Exception as e:
        logger.error("DIAAD execution failed: %s", e, exc_info=True)
        raise

    finally:
        if ctx is not None:
            terminate_logger(**ctx.termination_kwargs())


# -------------------------------------------------------------
# Direct execution
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
