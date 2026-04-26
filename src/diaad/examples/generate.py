from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import random
import yaml

from diaad.transcripts.cha_files import read_cha_files
from diaad.transcripts.transcript_tables import tabularize_transcripts
from diaad.transcripts.transcription_reliability_evaluation import (
    evaluate_transcription_reliability,
)
from diaad.transcripts.transcription_reliability_selection import (
    reselect_transcription_reliability_samples,
    select_transcription_reliability_samples,
)
from psair.metadata.metadata_fields import MetadataManager


SPEC_PACKAGE = "diaad.examples"
SPEC_ROOT = ("assets", "spec")
EXPECTED_WORKBOOK = "transcript_table.xlsx"
SELECT_OUTPUT_DIR = "transcripts_select"
EVALUATE_OUTPUT_DIR = "transcripts_evaluate"
RESELECT_OUTPUT_DIR = "transcripts_reselect"


@contextmanager
def _scratch_dir(parent: Path):
    path = parent / f"_dx_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _asset_path(*parts: str):
    path = resources.files(SPEC_PACKAGE)
    for part in parts:
        path = path.joinpath(part)
    return path


def _read_yaml_asset(*parts: str) -> dict[str, Any]:
    with _asset_path(*parts).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML asset {'/'.join(parts)} must contain a mapping.")
    return data


def _write_text(path: Path, text: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_yaml(path: Path, data: dict[str, Any], *, force: bool) -> None:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=False)
    _write_text(path, text, force=force)


def _validate_chat_spec(chat_spec: dict[str, Any]) -> None:
    chats = chat_spec.get("chat_files")
    if not isinstance(chats, list) or not chats:
        raise ValueError("chat_files.yaml must define a non-empty chat_files list.")

    filenames = set()
    for item in chats:
        filename = item.get("filename")
        content = item.get("content")
        if not filename or not str(filename).endswith(".cha"):
            raise ValueError("Each chat file needs a .cha filename.")
        if filename in filenames:
            raise ValueError(f"Duplicate synthetic CHAT filename: {filename}")
        if not isinstance(content, str) or "*PAR:" not in content:
            raise ValueError(f"Synthetic CHAT file {filename} is missing PAR utterances.")
        filenames.add(filename)


def _validate_reliability_spec(
    chat_spec: dict[str, Any],
    reliability_spec: dict[str, Any],
) -> None:
    original_names = {chat["filename"] for chat in chat_spec.get("chat_files", [])}
    reliability_files = reliability_spec.get("reliability_chat_files")

    if not isinstance(reliability_files, list) or not reliability_files:
        raise ValueError(
            "reliability_chat_files.yaml must define a non-empty reliability_chat_files list."
        )

    for item in reliability_files:
        filename = item.get("filename")
        original_filename = item.get("original_filename")
        content = item.get("content")
        if not filename or not str(filename).endswith(".cha"):
            raise ValueError("Each reliability CHAT file needs a .cha filename.")
        if original_filename not in original_names:
            raise ValueError(
                f"Reliability CHAT {filename} points to unknown original {original_filename}."
            )
        if not isinstance(content, str) or "*PAR:" not in content:
            raise ValueError(
                f"Synthetic reliability CHAT file {filename} is missing PAR utterances."
            )


def _count_utterance_lines(chat_spec: dict[str, Any]) -> int:
    return sum(
        1
        for chat in chat_spec.get("chat_files", [])
        for line in chat.get("content", "").splitlines()
        if line.startswith("*")
    )


def _validate_expected_tables(
    chat_spec: dict[str, Any],
    expected_spec: dict[str, Any],
) -> None:
    expected = expected_spec.get("transcripts_tabularize", {})
    expected_count = expected.get("expected_utterance_count")
    if expected_count is not None and expected_count != _count_utterance_lines(chat_spec):
        raise ValueError(
            "expected_tables.yaml expected_utterance_count does not match CHAT specs."
        )


def _read_specs() -> dict[str, dict[str, Any]]:
    specs = {
        "dataset": _read_yaml_asset(*SPEC_ROOT, "dataset.yaml"),
        "project_config": _read_yaml_asset(*SPEC_ROOT, "configs", "project.yaml"),
        "advanced_config": _read_yaml_asset(*SPEC_ROOT, "configs", "advanced.yaml"),
        "chat_files": _read_yaml_asset(*SPEC_ROOT, "transcripts", "chat_files.yaml"),
        "reliability_chat_files": _read_yaml_asset(
            *SPEC_ROOT,
            "transcripts",
            "reliability_chat_files.yaml",
        ),
        "expected_tables": _read_yaml_asset(*SPEC_ROOT, "transcripts", "expected_tables.yaml"),
    }
    _validate_chat_spec(specs["chat_files"])
    _validate_reliability_spec(specs["chat_files"], specs["reliability_chat_files"])
    _validate_expected_tables(specs["chat_files"], specs["expected_tables"])
    return specs


def _write_readme(project_dir: Path, dataset: dict[str, Any], *, force: bool) -> None:
    text = f"""# {dataset.get("name", "DIAAD Synthetic Example Project")}

This runnable project is generated from packaged DIAAD YAML specs.

The CHAT files are fully synthetic and intentionally tiny. They are meant to
demonstrate DIAAD transcript commands, not to represent human-subjects data or
a clinical record.

Key files:

- `config/project.yaml`: minimal DIAAD project settings for this example.
- `config/advanced.yaml`: advanced DIAAD settings for this example.
- `input/chat/*.cha`: synthetic CHAT inputs.
- `input/chat/reliability/*.cha`: synthetic reliability transcriptions.
- `expected_outputs/transcripts_tabularize/transcript_table.xlsx`: workbook
  generated from the synthetic CHAT files.
- `expected_outputs/transcripts_select/`: reliability selection outputs.
- `expected_outputs/transcripts_evaluate/`: reliability evaluation outputs.
- `expected_outputs/transcripts_reselect/`: reliability reselection outputs.
"""
    _write_text(project_dir / "README.md", text, force=force)


def _materialize_inputs(project_dir: Path, specs: dict[str, dict[str, Any]], *, force: bool) -> None:
    _write_readme(project_dir, specs["dataset"], force=force)
    _write_yaml(project_dir / "config" / "project.yaml", specs["project_config"], force=force)
    _write_yaml(project_dir / "config" / "advanced.yaml", specs["advanced_config"], force=force)

    obsolete_advanced_project = project_dir / "config" / "advanced_project.yaml"
    if force and obsolete_advanced_project.exists():
        obsolete_advanced_project.unlink()

    for chat in specs["chat_files"]["chat_files"]:
        _write_text(
            project_dir / "input" / "chat" / chat["filename"],
            chat["content"].rstrip() + "\n",
            force=force,
        )

    for chat in specs["reliability_chat_files"]["reliability_chat_files"]:
        _write_text(
            project_dir / "input" / "chat" / "reliability" / chat["filename"],
            chat["content"].rstrip() + "\n",
            force=force,
        )


def _metadata_fields(project_dir: Path, project_config: dict[str, Any]) -> dict[str, Any]:
    metadata_config = {
        "tiers": project_config.get("metadata_fields", {}),
        "input_dir": project_dir / project_config.get("input_dir", "input"),
    }
    return MetadataManager(metadata_config).metadata_fields


def _write_expected_transcript_table(project_dir: Path, specs: dict[str, dict[str, Any]], *, force: bool) -> Path:
    expected_dir = project_dir / "expected_outputs" / "transcripts_tabularize"
    target = expected_dir / EXPECTED_WORKBOOK
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")

    input_dir = project_dir / specs["project_config"].get("input_dir", "input")
    metadata_fields = _metadata_fields(project_dir, specs["project_config"])
    chats = read_cha_files(
        input_dir=input_dir,
        shuffle=False,
        exclude_dirnames=[specs["advanced_config"].get("reliability_dirname", "reliability")],
    )

    with _scratch_dir(project_dir) as tmpdir:
        written = tabularize_transcripts(
            metadata_fields=metadata_fields,
            chats=chats,
            output_dir=tmpdir,
            shuffle=False,
            random_seed=specs["project_config"].get("random_seed", 99),
        )
        if not written:
            raise RuntimeError("Synthetic transcript tabularization did not write a workbook.")
        expected_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(written[0], target)

    return target


def _replace_tree(source: Path, target: Path, *, force: bool) -> None:
    if target.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing directory: {target}")
        shutil.copytree(source, target, dirs_exist_ok=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)


def _write_expected_selection(project_dir: Path, specs: dict[str, dict[str, Any]], *, force: bool) -> Path:
    expected_dir = project_dir / "expected_outputs" / SELECT_OUTPUT_DIR
    input_dir = project_dir / specs["project_config"].get("input_dir", "input")
    metadata_fields = _metadata_fields(project_dir, specs["project_config"])
    chats = {
        key: value
        for key, value in read_cha_files(
            input_dir=input_dir,
            shuffle=False,
            exclude_dirnames=[
                specs["advanced_config"].get("reliability_dirname", "reliability")
            ],
        ).items()
    }

    random.seed(specs["project_config"].get("random_seed", 99))
    with _scratch_dir(project_dir) as tmpdir:
        select_transcription_reliability_samples(
            metadata_fields=metadata_fields,
            chats=chats,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            output_dir=tmpdir,
            input_dir=input_dir,
        )
        source = tmpdir / "transcription_reliability_selection"
        _replace_tree(source, expected_dir, force=force)

        reselect_input = (
            project_dir
            / specs["project_config"].get("input_dir", "input")
            / "transcription_reliability_selection"
            / "transcription_reliability_samples.xlsx"
        )
        if reselect_input.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing file: {reselect_input}")
        reselect_input.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "transcription_reliability_samples.xlsx", reselect_input)

    return expected_dir


def _write_expected_evaluation(project_dir: Path, specs: dict[str, dict[str, Any]], *, force: bool) -> Path:
    expected_dir = project_dir / "expected_outputs" / EVALUATE_OUTPUT_DIR
    metadata_fields = _metadata_fields(project_dir, specs["project_config"])

    with _scratch_dir(project_dir) as tmpdir:
        eval_input_dir = tmpdir / "input"
        shutil.copytree(project_dir / specs["project_config"].get("input_dir", "input"), eval_input_dir)

        evaluate_transcription_reliability(
            metadata_fields=metadata_fields,
            input_dir=eval_input_dir,
            output_dir=tmpdir,
            exclude_participants=specs["project_config"].get("exclude_participants", []),
            strip_clan=specs["project_config"].get("strip_clan", True),
            prefer_correction=specs["project_config"].get("prefer_correction", True),
            lowercase=specs["project_config"].get("lowercase", True),
            reliability_tag=specs["advanced_config"].get("reliability_tag", "_reliability"),
            reliability_dirname=specs["advanced_config"].get("reliability_dirname", "reliability"),
        )
        source = tmpdir / "transcription_reliability_evaluation"
        _replace_tree(source, expected_dir, force=force)

    return expected_dir


def _write_expected_reselection(project_dir: Path, specs: dict[str, dict[str, Any]], *, force: bool) -> Path:
    expected_dir = project_dir / "expected_outputs" / RESELECT_OUTPUT_DIR

    np.random.seed(specs["project_config"].get("random_seed", 99))
    with _scratch_dir(project_dir) as tmpdir:
        reselect_transcription_reliability_samples(
            input_dir=(
                project_dir
                / specs["project_config"].get("input_dir", "input")
                / "transcription_reliability_selection"
            ),
            output_dir=tmpdir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
        )
        source = tmpdir / "reselected_transcription_reliability"
        _replace_tree(source, expected_dir / "reselected_transcription_reliability", force=force)

    return expected_dir


def generate_example_files(destination: str | Path, *, force: bool = False) -> Path:
    """
    Materialize the synthetic DIAAD example project.

    Parameters
    ----------
    destination
        Directory to create or update, usually ``example_files/synthetic_project``.
    force
        Overwrite existing files when True.
    """
    project_dir = Path(destination).expanduser().resolve()
    if project_dir.exists() and any(project_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Refusing to write into non-empty directory without --force: {project_dir}"
        )

    specs = _read_specs()
    project_dir.mkdir(parents=True, exist_ok=True)
    _materialize_inputs(project_dir, specs, force=force)
    _write_expected_transcript_table(project_dir, specs, force=force)
    _write_expected_selection(project_dir, specs, force=force)
    _write_expected_evaluation(project_dir, specs, force=force)
    _write_expected_reselection(project_dir, specs, force=force)
    return project_dir
