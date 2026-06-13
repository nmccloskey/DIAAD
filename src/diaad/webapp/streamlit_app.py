from __future__ import annotations

import tempfile
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath

import streamlit as st
import yaml

from diaad import __version__
from diaad.cli.commands import MODULE_COMMANDS
from diaad.cli.dispatch import build_dispatch, prepare_dispatch_prerequisites
from psair.core.logger import initialize_logger, logger, set_root, terminate_logger
from diaad.core.run_context import RunContext
from diaad.webapp.config_builder import (
    build_config_ui,
    restore_default_config_ui_state,
)


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

MODULE_LABELS = {
    "examples": "Examples / Example I/O",
    "blinding": "Blinding",
    "transcripts": "Transcripts",
    "templates": "Templates",
    "cus": "Complete Utterances",
    "words": "Word Counting",
    "powers": "POWERS",
    "vocab": "Target Vocabulary Coverage",
    "turns": "Digital Conversational Turns",
}

SPECIAL_WEB_COMMANDS = {"examples"}


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
        file_path = input_dir / _safe_uploaded_relative_path(uploaded.name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as f:
            f.write(uploaded.getbuffer())


def _safe_uploaded_relative_path(uploaded_name: str) -> Path:
    normalized_name = uploaded_name.replace("\\", "/")
    upload_path = PurePosixPath(normalized_name)
    parts = [part for part in upload_path.parts if part not in ("", ".")]

    if (
        upload_path.is_absolute()
        or not parts
        or any(part == ".." or ":" in part for part in parts)
    ):
        raise ValueError(f"Unsafe uploaded file path: {uploaded_name}")

    return Path(*parts)


def _dispatched_commands() -> set[str]:
    return set(build_dispatch(object()).keys())


def _web_supported_commands() -> set[str]:
    return _dispatched_commands() | SPECIAL_WEB_COMMANDS


def _command_options_by_module() -> dict[str, list[str]]:
    supported_commands = _web_supported_commands()
    return {
        module: [
            command
            for command in module_commands
            if command in supported_commands
        ]
        for module, module_commands in MODULE_COMMANDS.items()
    }


def _command_options() -> list[str]:
    return [
        command
        for module_commands in _command_options_by_module().values()
        for command in module_commands
    ]


def _command_checkbox_key(command: str) -> str:
    return "diaad_command_" + command.replace(" ", "_").replace("/", "_")


def _command_module_key(module: str) -> str:
    return f"diaad_command_module_{module}_expanded"


def _clear_command_selection() -> None:
    for command in _command_options():
        st.session_state[_command_checkbox_key(command)] = False


def _render_command_menu() -> list[str]:
    command_groups = _command_options_by_module()

    with st.expander("Show / Hide DIAAD Command Menu", expanded=False):
        for module, commands in command_groups.items():
            if not commands:
                continue

            label = MODULE_LABELS.get(module, module.title())
            state_key = _command_module_key(module)
            if state_key not in st.session_state:
                st.session_state[state_key] = False

            is_expanded = bool(st.session_state[state_key])
            caret = "v" if is_expanded else ">"
            if st.button(f"{caret} {label}", key=f"{state_key}_button"):
                st.session_state[state_key] = not is_expanded
                st.rerun()

            if st.session_state[state_key]:
                for command in commands:
                    st.checkbox(
                        command,
                        key=_command_checkbox_key(command),
                        help="Select this canonical DIAAD command for the next run.",
                    )

    selected = [
        command
        for command in _command_options()
        if st.session_state.get(_command_checkbox_key(command), False)
    ]

    if selected:
        st.markdown("Selected commands:")
        st.code(", ".join(selected), language="text")
    else:
        st.caption("No DIAAD commands selected.")

    return selected


def _request_restore_config_defaults() -> None:
    st.session_state.restore_config_defaults_requested = True


def _cancel_restore_config_defaults() -> None:
    st.session_state.restore_config_defaults_requested = False


def _restore_config_defaults() -> None:
    restore_default_config_ui_state()
    st.session_state.confirmed_config = False
    st.session_state.built_configs = None
    st.session_state.restore_config_defaults_requested = False


def _render_restore_defaults_controls() -> None:
    if "restore_config_defaults_requested" not in st.session_state:
        st.session_state.restore_config_defaults_requested = False

    st.button("Restore defaults", on_click=_request_restore_config_defaults)

    if not st.session_state.restore_config_defaults_requested:
        return

    st.warning(
        "Restore the config builder to DIAAD's default values? "
        "This will clear the current built config."
    )
    col_yes, col_no = st.columns(2)
    col_yes.button(
        "Yes, restore defaults",
        type="primary",
        on_click=_restore_config_defaults,
    )
    col_no.button("No, keep current", on_click=_cancel_restore_config_defaults)


def _render_overview() -> None:
    with st.expander("Overview", expanded=True):
        st.markdown(
            """
Welcome to the DIAAD web app.

Use this page to build or upload configuration files, upload your DIAAD input
files, choose one or more DIAAD commands, and download the resulting output ZIP.

The manual and example menus below provide command-specific guidance and example
input/output files. DIAAD runs in a temporary workspace for each web run, so
uploaded inputs are processed only within that run.

Source code: [nmccloskey/DIAAD](https://github.com/nmccloskey/DIAAD)
            """.strip()
        )


def _render_manual() -> None:
    if render_manual_ui is None:
        return
    
    repo_root = Path(__file__).resolve().parents[3]

    render_manual_ui(
        repo_root=repo_root,
        manual_rel_dir="docs/manual",
        expander_label="Show / Hide DIAAD Manual Menu",
        ui_key="diaad_manual",
    )

    # Convenient for development, but will be threaded into the user manual.
    render_manual_ui(
        repo_root=repo_root,
        manual_rel_dir="src/diaad/examples/assets/rendered_docs/example_io",
        expander_label="Show / Hide DIAAD Example Input/Output Menu",
        ui_key="diaad_examples",
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


def _run_examples_web() -> BytesIO:
    from diaad.examples.generate import generate_example_files

    with tempfile.TemporaryDirectory() as tmpdir:
        project_dir = generate_example_files(
            Path(tmpdir) / "synthetic_project",
            force=True,
        )
        return zip_folder(project_dir)


def render_app() -> None:
    st.title("DIAAD Web App")
    st.subheader("Database-oriented, Integrative Architecture for Analyzing Discourse")

    if "confirmed_config" not in st.session_state:
        st.session_state.confirmed_config = False
    if "built_configs" not in st.session_state:
        st.session_state.built_configs = None

    st.header("Part 0: Instructions")
    _render_overview()
    _render_manual()
    

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
        _render_restore_defaults_controls()
        if st.session_state.confirmed_config:
            configs = st.session_state.built_configs

    st.header("Part 2: Input Files")
    uploaded_inputs = st.file_uploader(
        "Upload input files or a folder",
        type=["cha", "xlsx", "csv", "json"],
        accept_multiple_files="directory",
        help=(
            "Upload individual files or choose a directory. Nested folders are "
            "preserved under DIAAD's web input directory."
        ),
    )

    st.header("Part 3: Commands")
    commands = _render_command_menu()

    if st.button("Run selected commands"):
        if not commands:
            st.warning("Please select at least one command.")
            st.stop()

        if commands == ["examples"]:
            try:
                zip_buffer = _run_examples_web()
                st.success("DIAAD example files generated successfully.")
                timestamp = datetime.now().strftime("%y%m%d_%H%M")
                st.download_button(
                    label="Download Example Files ZIP",
                    data=zip_buffer,
                    file_name=f"diaad_example_files_{timestamp}.zip",
                    mime="application/zip",
                )
            except Exception:
                logger.exception("Unhandled error while generating DIAAD examples.")
                st.error("An unexpected error occurred while generating DIAAD examples.")
            st.stop()

        if "examples" in commands:
            st.warning(
                "Please run the examples command by itself. It generates a "
                "standalone synthetic project ZIP rather than using uploaded inputs."
            )
            st.stop()
        if configs is None:
            st.warning("Please upload or build a complete two-file config first.")
            st.stop()
        if not uploaded_inputs:
            st.warning("Please upload at least one input file.")
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

    if commands:
        st.button("Clear selection", on_click=_clear_command_selection)


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
