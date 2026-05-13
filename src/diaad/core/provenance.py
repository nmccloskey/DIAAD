from __future__ import annotations

from argparse import Namespace
from datetime import datetime
from typing import Any

from diaad import __version__
from psair.core.dry_run import (
    build_config_dry_run_payload,
    print_config_dry_run,
    write_config_dry_run,
)
from psair.core.provenance import (
    capture_directory_snapshot,
    capture_environment,
    serialize_cli_args,
    write_cli_args,
    write_effective_config,
    write_json,
    write_manifest,
)


ENVIRONMENT_PACKAGES = [
    "diaad",
    "psair",
    "pandas",
    "numpy",
    "openpyxl",
    "xlsxwriter",
    "scikit-learn",
    "scipy",
]

LOG_ARTIFACTS = {
    "log": "logs/run_log.log",
    "run_metadata": "logs/run_metadata.json",
    "effective_config": "logs/effective_config.yaml",
    "cli_args": "logs/cli_args.json",
    "config_overrides": "logs/config_overrides.json",
    "directory_snapshot_start": "logs/directory_snapshot_start.json",
    "directory_snapshot_end": "logs/directory_snapshot_end.json",
    "environment": "logs/environment.json",
    "manifest": "logs/manifest.json",
}


def build_dry_run_payload(ctx, args: Namespace, commands: list[str]) -> dict[str, Any]:
    """Build DIAAD's resolved-config dry-run payload."""
    return build_config_dry_run_payload(
        program={"name": "DIAAD", "version": __version__},
        commands=commands,
        paths=ctx.run_paths(),
        cli_args=serialize_cli_args(args),
        config_overrides=ctx.config.override_diff,
        effective_config=ctx.config.to_dict(),
        environment=capture_environment(ENVIRONMENT_PACKAGES),
    )


def emit_dry_run_config(ctx, args: Namespace, commands: list[str]) -> None:
    """Print and optionally save DIAAD dry-run config output."""
    payload = build_dry_run_payload(ctx, args, commands)
    out_path = getattr(args, "dry_run_config_out", None)
    if out_path:
        write_config_dry_run(out_path, payload)
    print_config_dry_run(
        payload,
        format=getattr(args, "dry_run_config_format", "json"),
    )


def write_start_artifacts(ctx, args: Namespace) -> None:
    """Write DIAAD provenance files known before command dispatch."""
    logs_dir = ctx.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    ctx.start_snapshot = _run_directory_snapshot(ctx)
    write_json(logs_dir / "directory_snapshot_start.json", ctx.start_snapshot)
    write_effective_config(logs_dir / "effective_config.yaml", ctx.config.to_dict())
    write_cli_args(logs_dir / "cli_args.json", args)
    write_json(logs_dir / "config_overrides.json", ctx.config.override_diff)
    write_json(
        logs_dir / "environment.json",
        capture_environment(ENVIRONMENT_PACKAGES),
    )


def finalize_run_artifacts(ctx, context: dict[str, Any]) -> None:
    """Write end-of-run snapshots, compact metadata, and manifest."""
    logs_dir = ctx.out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    end_snapshot = _run_directory_snapshot(ctx)
    write_json(logs_dir / "directory_snapshot_end.json", end_snapshot)

    status = str(context.get("status", "completed"))
    start_time = context["start_time"]
    end_time = context.get("end_time", datetime.now())
    results = _produced_result_files(ctx.start_snapshot, end_snapshot)
    command = _command_value(ctx.commands)

    manifest_artifacts = {
        **LOG_ARTIFACTS,
        "results": results,
    }
    write_manifest(
        logs_dir / "manifest.json",
        run_id=_run_id(ctx),
        command=command,
        status=status,
        artifacts=manifest_artifacts,
    )
    write_json(
        logs_dir / "run_metadata.json",
        _compact_run_metadata(
            ctx=ctx,
            status=status,
            start_time=start_time,
            end_time=end_time,
        ),
    )


def _run_directory_snapshot(ctx) -> dict[str, Any]:
    return {
        "input_contents": capture_directory_snapshot(
            ctx.input_dir,
            root=ctx.project_root,
        ),
        "output_contents": capture_directory_snapshot(
            ctx.out_dir,
            root=ctx.out_dir,
        ),
    }


def _compact_run_metadata(
    *,
    ctx,
    status: str,
    start_time: datetime,
    end_time: datetime,
) -> dict[str, Any]:
    runtime_seconds = round((end_time - start_time).total_seconds(), 2)
    return {
        "run_id": _run_id(ctx),
        "program": {"name": "DIAAD", "version": __version__},
        "status": status,
        "commands": ctx.commands,
        "started_at": start_time.isoformat(timespec="seconds"),
        "ended_at": end_time.isoformat(timespec="seconds"),
        "runtime_seconds": runtime_seconds,
        "paths": ctx.run_paths(),
        "logs": dict(LOG_ARTIFACTS),
    }


def _produced_result_files(
    start_snapshot: dict[str, Any] | None,
    end_snapshot: dict[str, Any],
) -> list[str]:
    start_files = _snapshot_file_paths(start_snapshot)
    end_files = _snapshot_file_paths(end_snapshot)
    produced = sorted(end_files - start_files)
    return [path for path in produced if not path.replace("\\", "/").startswith("logs/")]


def _snapshot_file_paths(snapshot: dict[str, Any] | None) -> set[str]:
    if not snapshot:
        return set()
    output_contents = snapshot.get("output_contents", {})
    paths = set()
    for item in output_contents.get("files", []):
        if isinstance(item, dict):
            path = item.get("path")
        else:
            path = item
        if path:
            paths.add(str(path))
    return paths


def _command_value(commands: list[str]) -> str | list[str]:
    if len(commands) == 1:
        return commands[0]
    return list(commands)


def _run_id(ctx) -> str:
    return f"diaad_{ctx.timestamp}"
