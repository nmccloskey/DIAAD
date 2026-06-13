#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from diaad import __version__
from diaad.core.run_context import RunContext
from diaad.core.config_overrides import build_cli_config_overrides
from diaad.core.provenance import (
    emit_dry_run_config,
    finalize_run_artifacts,
    write_start_artifacts,
)
from diaad.cli.dispatch import build_dispatch, prepare_dispatch_prerequisites
from diaad.cli.parser import build_arg_parser
from diaad.cli.commands import parse_cli_commands
from psair.core.logger import (
    add_finalization_hook,
    initialize_logger,
    logger,
    set_root,
    terminate_logger,
)


def _is_examples_command(args) -> bool:
    return " ".join(args.command).strip().lower() == "examples"


def _has_examples_options(args) -> bool:
    return bool(
        getattr(args, "render_docs", False)
        or getattr(args, "example_commands", None)
        or getattr(args, "force", False)
    )


def _run_examples_command(args) -> None:
    render_docs = getattr(args, "render_docs", False)
    example_commands = getattr(args, "example_commands", None)
    if render_docs:
        from diaad.examples.render_docs import render_example_docs

        paths = render_example_docs()
        for path in paths:
            print(f"Rendered DIAAD example doc: {path}")
        return

    ctx = None
    status = "completed"
    logger_initialized = False

    try:
        start_time = datetime.now()
        project_root = Path.cwd().resolve()
        set_root(project_root)
        config_overrides = build_cli_config_overrides(args)
        ctx = RunContext(
            config_dir=args.config,
            project_root=project_root,
            start_time=start_time,
            config_overrides=config_overrides,
        )

        initialize_logger(
            start_time,
            ctx.out_dir,
            program_name="DIAAD",
            version=__version__,
        )
        logger_initialized = True
        logger.info("Logger initialized and early logs flushed.")
        add_finalization_hook(lambda context: finalize_run_artifacts(ctx, context))
        ctx.set_commands(["examples"])
        write_start_artifacts(ctx, args)

        logger.info("Executing command(s): examples")

        from diaad.examples.generate import generate_example_files

        if example_commands:
            destination = ctx.out_dir
            project_dir = generate_example_files(
                destination,
                force=getattr(args, "force", False),
                commands=example_commands,
            )
        else:
            destination = ctx.out_dir / "example_files_full_dataset"
            project_dir = generate_example_files(
                destination,
                force=getattr(args, "force", False),
            )
        logger.info("Generated DIAAD example files: %s", project_dir)
        print(f"Generated DIAAD example files: {project_dir}")

    except Exception as e:
        status = "failed"
        logger.error("DIAAD examples generation failed: %s", e, exc_info=True)
        raise
    finally:
        if ctx is not None and logger_initialized:
            terminate_logger(**ctx.termination_kwargs(), status=status)


def main(args) -> None:
    """Parse CLI arguments and execute the requested DIAAD commands."""
    ctx = None
    status = "completed"
    logger_initialized = False

    try:
        if _is_examples_command(args):
            _run_examples_command(args)
            return
        if _has_examples_options(args):
            raise ValueError(
                "--render-docs, --for-command, and --force are only valid with "
                "'diaad examples'."
            )

        start_time = datetime.now()
        project_root = Path.cwd().resolve()
        set_root(project_root)

        config_overrides = build_cli_config_overrides(args)

        ctx = RunContext(
            config_dir=args.config,
            project_root=project_root,
            start_time=start_time,
            config_overrides=config_overrides,
            create_output_dir=not getattr(args, "dry_run_config", False),
        )

        commands = parse_cli_commands(args.command, logger=logger)
        ctx.set_commands(commands)

        if getattr(args, "dry_run_config", False):
            emit_dry_run_config(ctx, args, commands)
            return

        # -----------------------------------------------------------------
        # Initialize logger once output folder is ready
        # -----------------------------------------------------------------
        initialize_logger(
            start_time,
            ctx.out_dir,
            program_name="DIAAD",
            version=__version__,
        )
        logger_initialized = True
        logger.info("Logger initialized and early logs flushed.")
        add_finalization_hook(lambda context: finalize_run_artifacts(ctx, context))
        write_start_artifacts(ctx, args)

        if not commands:
            status = "skipped"
            logger.error("No valid commands recognized - exiting.")
            return

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
        status = "failed"
        logger.error("DIAAD execution failed: %s", e, exc_info=True)
        raise

    finally:
        if ctx is not None and logger_initialized:
            terminate_logger(**ctx.termination_kwargs(), status=status)


# -------------------------------------------------------------
# Direct execution
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
