#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from diaad import __version__
from diaad.core.run_context import RunContext
from diaad.cli.dispatch import build_dispatch, prepare_dispatch_prerequisites
from diaad.cli.parser import build_arg_parser
from diaad.cli.commands import parse_cli_commands
from psair.core.logger import (
    initialize_logger,
    logger,
    set_root,
    terminate_logger,
)


def _is_examples_command(args) -> bool:
    return " ".join(args.command).strip().lower() == "examples"


def _has_examples_options(args) -> bool:
    return bool(
        getattr(args, "example_files", None)
        or getattr(args, "render_docs", False)
        or getattr(args, "force", False)
    )


def _run_examples_command(args) -> None:
    if not getattr(args, "example_files", None) and not getattr(args, "render_docs", False):
        raise ValueError(
            "diaad examples requires --files <directory>, --render-docs, or both."
        )

    if getattr(args, "example_files", None):
        from diaad.examples.generate import generate_example_files

        project_dir = generate_example_files(
            args.example_files,
            force=getattr(args, "force", False),
        )
        print(f"Generated DIAAD example files: {project_dir}")

    if getattr(args, "render_docs", False):
        from diaad.examples.render_docs import render_example_docs

        paths = render_example_docs()
        for path in paths:
            print(f"Rendered DIAAD example doc: {path}")


def main(args) -> None:
    """Parse CLI arguments and execute the requested DIAAD commands."""
    ctx = None

    try:
        if _is_examples_command(args):
            _run_examples_command(args)
            return
        if _has_examples_options(args):
            raise ValueError("--files, --render-docs, and --force are only valid with 'diaad examples'.")

        start_time = datetime.now()
        project_root = Path.cwd().resolve()
        set_root(project_root)

        print("ARGS.CONFIG:", args.config)

        ctx = RunContext(
            config_dir=args.config or "config",
            project_root=project_root,
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
