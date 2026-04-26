from __future__ import annotations

import tempfile
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
import yaml

from diaad import __version__
from diaad.cli.commands import MODULE_COMMANDS
from diaad.cli.dispatch import build_dispatch, prepare_dispatch_prerequisites
from psair.core.logger import initialize_logger, logger, set_root, terminate_logger
from diaad.core.run_context import RunContext
from diaad.webapp.config_builder import build_config_ui


try:
    from psair.webapp.manual_viewer import render_manual_ui
except Exception:
    render_manual_ui = None


CONFIG_FILENAMES = {
    "project.yaml": "project",
    "project.yml": "project",
    "advanced.yaml": "advanced",
    "advanced.yml": "advanced",
}

DISPATCHED_COMMANDS = {
    "transcripts tabularize",
    "transcripts select",
    "transcripts reselect",
    "transcripts evaluate",
    "cus files",
    "cus reselect",
    "cus evaluate",
    "cus analyze",
    "cus rates",
    "words files",
    "words reselect",
    "words evaluate",
    "words analyze",
    "words rates",
    "vocab file",
    "vocab analyze",
    "turns files",
    "turns evaluate",
    "turns reselect",
    "turns analyze",
    "templates utterances",
    "templates samples",
    "templates times",
    "powers files",
    "powers analyze",
    "powers rates",
    "powers evaluate",
    "powers reselect",
}


def zip_folder(folder_path: Path) -> BytesIO:
    """Compress a folder into an in-memory ZIP buffer."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in folder_path.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(folder_path)
                zf.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _read_uploaded_configs(files) -> tuple[dict[str, dict] | None, list[str]]:
    if not files:
        return None, []

    configs: dict[str, dict] = {}
    errors: list[str] = []

    for uploaded in files:
        filename = Path(uploaded.name).name.lower()
        part = CONFIG_FILENAMES.get(filename)
        if part is None:
            errors.append(
                f"Ignoring {uploaded.name}: expected project.yaml or advanced.yaml."
            )
            continue

        try:
            data = yaml.safe_load(uploaded.getvalue().decode("utf-8")) or {}
        except Exception as e:
            errors.append(f"Could not parse {uploaded.name}: {e}")
            continue

        if not isinstance(data, dict):
            errors.append(f"{uploaded.name} must contain a YAML mapping.")
            continue

        configs[part] = data

    missing = [name for name in ("project", "advanced") if name not in configs]
    if missing:
        errors.append(
            "Missing config file(s): "
            + ", ".join(f"{name}.yaml" for name in missing)
            + "."
        )
        return None, errors

    return configs, errors


def _web_project_config(project_config: dict) -> dict:
    """
    Keep web runs inside the temporary project root regardless of uploaded paths.
    """
    project = dict(project_config)
    project["input_dir"] = "input"
    project["output_dir"] = "output"
    return project


def _write_config_dir(config_dir: Path, configs: dict[str, dict]) -> None:
    _write_yaml(config_dir / "project.yaml", _web_project_config(configs["project"]))
    _write_yaml(config_dir / "advanced.yaml", configs["advanced"])


def _save_uploaded_inputs(input_dir: Path, uploaded_files) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files:
        file_path = input_dir / Path(uploaded.name).name
        with file_path.open("wb") as f:
            f.write(uploaded.getbuffer())


def _command_options() -> list[str]:
    return [
        command
        for module_commands in MODULE_COMMANDS.values()
        for command in module_commands
        if command in DISPATCHED_COMMANDS
    ]


def _render_manual() -> None:
    if render_manual_ui is None:
        return
    repo_root = Path(__file__).resolve().parents[3]
    render_manual_ui(
        repo_root=repo_root,
        manual_rel_dir="docs/manual",
        expander_label="Show / Hide DIAAD Manual Menu",
    )


def _run_diaad_web(configs: dict[str, dict], uploaded_inputs, commands: list[str]) -> BytesIO:
    start_time = datetime.now()

    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir).resolve()
        set_root(project_root)

        config_dir = project_root / "config"
        _write_config_dir(config_dir, configs)

        ctx = RunContext(
            config_dir=config_dir,
            project_root=project_root,
            start_time=start_time,
        )
        _save_uploaded_inputs(ctx.input_dir, uploaded_inputs)

        initialize_logger(
            start_time,
            ctx.out_dir,
            program_name="DIAAD",
            version=__version__,
        )
        logger.info("DIAAD web run initialized.")
        logger.info("Executing command(s): %s", ", ".join(commands))

        try:
            ctx.set_commands(commands)
            prepare_dispatch_prerequisites(ctx, commands)
            dispatch = build_dispatch(ctx)

            executed: list[str] = []
            for command in commands:
                func = dispatch.get(command)
                if func is None:
                    logger.error("Unknown command: %s", command)
                    continue
                func()
                executed.append(command)

            if executed:
                logger.info("Completed: %s", ", ".join(executed))
        finally:
            terminate_logger(**ctx.termination_kwargs())

        return zip_folder(ctx.out_dir)


def render_app() -> None:
    st.title("DIAAD Web App")
    st.subheader("Database-oriented, Integrative Architecture for Analyzing Discourse")
    _render_manual()

    if "confirmed_config" not in st.session_state:
        st.session_state.confirmed_config = False
    if "built_configs" not in st.session_state:
        st.session_state.built_configs = None

    st.header("Part 1: Configuration")
    uploaded_config_files = st.file_uploader(
        "Upload project.yaml and advanced.yaml",
        type=["yaml", "yml"],
        accept_multiple_files=True,
    )

    configs = None
    config_errors: list[str] = []
    if uploaded_config_files:
        st.session_state.confirmed_config = False
        st.session_state.built_configs = None
        configs, config_errors = _read_uploaded_configs(uploaded_config_files)
        for error in config_errors:
            st.warning(error)
        if configs:
            st.success("Config files uploaded.")
    else:
        st.caption("No config files uploaded? Build them here.")
        built_configs, valid = build_config_ui()
        if st.button("Use this built config", disabled=not valid):
            st.session_state.built_configs = built_configs
            st.session_state.confirmed_config = True
            st.success("Built config confirmed.")
        if st.session_state.confirmed_config:
            configs = st.session_state.built_configs

    st.header("Part 2: Input Files")
    uploaded_inputs = st.file_uploader(
        "Upload input files",
        type=["cha", "xlsx", "csv", "json"],
        accept_multiple_files=True,
    )

    st.header("Part 3: Commands")
    commands = st.multiselect(
        "Select commands",
        _command_options(),
        help="These are the same canonical commands used by the CLI.",
    )

    if st.button("Run selected commands"):
        if configs is None:
            st.warning("Please upload or build a complete two-file config first.")
            st.stop()
        if not uploaded_inputs:
            st.warning("Please upload at least one input file.")
            st.stop()
        if not commands:
            st.warning("Please select at least one command.")
            st.stop()

        try:
            zip_buffer = _run_diaad_web(configs, uploaded_inputs, commands)
            st.success("All selected commands completed successfully.")
            timestamp = datetime.now().strftime("%y%m%d_%H%M")
            st.download_button(
                label="Download Results ZIP",
                data=zip_buffer,
                file_name=f"diaad_web_output_{timestamp}.zip",
                mime="application/zip",
            )
        except Exception:
            logger.exception("Unhandled error during DIAAD web run.")
            st.error(
                "An unexpected error occurred while running DIAAD. "
                "Please check the logs in the downloaded ZIP if one was created."
            )


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def main() -> None:
    """Launch the Streamlit app, or render it when already inside Streamlit."""
    if _running_under_streamlit():
        render_app()
        return

    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=False)


if __name__ == "__main__":
    render_app()
