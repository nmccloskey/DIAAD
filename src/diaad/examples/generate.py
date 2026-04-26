from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import random
import yaml

from diaad.coding.compl_utts.analysis import analyze_cu_coding
from diaad.coding.compl_utts.files import make_cu_coding_files
from diaad.coding.compl_utts.rates import calculate_cu_rates
from diaad.coding.compl_utts.rel_evaluation import evaluate_cu_reliability
from diaad.coding.compl_utts.rel_reselection import reselect_cu_rel
from diaad.coding.templates.samples import make_sample_template_files
from diaad.coding.templates.times import make_speaking_time_template_files
from diaad.coding.templates.utterances import make_utterance_template_files
from diaad.coding.word_counts.analysis import analyze_word_counts
from diaad.coding.word_counts.files import make_word_count_files
from diaad.coding.word_counts.rates import calculate_word_count_rates
from diaad.coding.word_counts.rel_evaluation import evaluate_word_count_reliability
from diaad.coding.word_counts.rel_reselection import reselect_wc_rel
from diaad.core.config import AdvancedConfig
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
TRANSCRIPTS_MODULE_DIR = "transcripts_module"
TEMPLATES_MODULE_DIR = "templates_module"
CUS_MODULE_DIR = "cus_module"
WORDS_MODULE_DIR = "words_module"
SELECT_OUTPUT_DIR = "transcripts_select"
EVALUATE_OUTPUT_DIR = "transcripts_evaluate"
RESELECT_OUTPUT_DIR = "transcripts_reselect"
TABULARIZE_OUTPUT_DIR = "transcripts_tabularize"
TEMPLATE_OUTPUT_DIRS = {
    "utterances": "templates_utterances",
    "samples": "templates_samples",
    "times": "templates_times",
}
CU_OUTPUT_DIRS = {
    "files": "cus_files",
    "evaluate": "cus_evaluate",
    "reselect": "cus_reselect",
    "analyze": "cus_analyze",
    "rates": "cus_rates",
}
WORD_OUTPUT_DIRS = {
    "files": "words_files",
    "evaluate": "words_evaluate",
    "reselect": "words_reselect",
    "analyze": "words_analyze",
    "rates": "words_rates",
}


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
- `expected_outputs/transcripts_module/`: outputs for transcript commands.
- `expected_outputs/templates_module/`: outputs for template commands.
- `expected_outputs/cus_module/`: outputs for complete-utterance coding commands.
- `expected_outputs/words_module/`: outputs for word-count commands.
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
    expected_dir = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / TABULARIZE_OUTPUT_DIR
    )
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
    source = Path(source)
    target = Path(target)

    if target.exists():
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing directory: {target}")
        _copy_tree_contents(source, target)
        return
    _copy_tree_contents(source, target)


def _long_path(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = path.resolve()
    if not (path.drive and not str(path).startswith("\\\\?\\")):
        return path
    return Path(f"\\\\?\\{path}")


def _copy_tree_contents(source: Path, target: Path) -> None:
    for file_path in source.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(source)
        target_path = target / relative
        _long_path(target_path.parent).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_long_path(file_path), _long_path(target_path))


def _cleanup_obsolete_expected_dirs(project_dir: Path, *, force: bool) -> None:
    if not force:
        return

    for dirname in (
        TABULARIZE_OUTPUT_DIR,
        SELECT_OUTPUT_DIR,
        EVALUATE_OUTPUT_DIR,
        RESELECT_OUTPUT_DIR,
        "cus_files",
        "cus_evaluate",
        "cus_reselect",
        "cus_analyze",
        "cus_rates",
        "words_files",
        "words_evaluate",
        "words_reselect",
        "words_analyze",
        "words_rates",
    ):
        shutil.rmtree(project_dir / "expected_outputs" / dirname, ignore_errors=True)


def _write_expected_selection(project_dir: Path, specs: dict[str, dict[str, Any]], *, force: bool) -> Path:
    expected_dir = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / SELECT_OUTPUT_DIR
    )
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
    expected_dir = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / EVALUATE_OUTPUT_DIR
    )
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
    expected_dir = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / RESELECT_OUTPUT_DIR
    )

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


def _prepare_template_input(tmpdir: Path, transcript_table: Path) -> Path:
    input_dir = tmpdir / "input"
    table_dir = input_dir / "transcript_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(transcript_table, table_dir / "transcript_tables.xlsx")
    return input_dir


def _template_blinding_config(specs: dict[str, dict[str, Any]]) -> AdvancedConfig:
    return AdvancedConfig(**specs["advanced_config"])


def _write_expected_utterance_templates(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir
        / "expected_outputs"
        / TEMPLATES_MODULE_DIR
        / TEMPLATE_OUTPUT_DIRS["utterances"]
    )
    transcript_table = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / TABULARIZE_OUTPUT_DIR
        / EXPECTED_WORKBOOK
    )

    with _scratch_dir(project_dir) as tmpdir:
        input_dir = _prepare_template_input(tmpdir, transcript_table)
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(specs["project_config"].get("random_seed", 99))
        make_utterance_template_files(
            input_dir=input_dir,
            output_dir=output_dir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            num_coders=specs["project_config"].get("num_coders", 0),
            stimulus_field=specs["project_config"].get("stimulus_field", ""),
            blinding_config=_template_blinding_config(specs),
            seed=specs["project_config"].get("random_seed", 99),
        )
        _replace_tree(output_dir / "coding_templates", expected_dir, force=force)

    return expected_dir


def _write_expected_sample_templates(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir
        / "expected_outputs"
        / TEMPLATES_MODULE_DIR
        / TEMPLATE_OUTPUT_DIRS["samples"]
    )
    transcript_table = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / TABULARIZE_OUTPUT_DIR
        / EXPECTED_WORKBOOK
    )

    with _scratch_dir(project_dir) as tmpdir:
        input_dir = _prepare_template_input(tmpdir, transcript_table)
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(specs["project_config"].get("random_seed", 99))
        make_sample_template_files(
            input_dir=input_dir,
            output_dir=output_dir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            num_bins=specs["project_config"].get("num_bins", 2),
            num_coders=specs["project_config"].get("num_coders", 0),
            stimulus_field=specs["project_config"].get("stimulus_field", ""),
            blinding_config=_template_blinding_config(specs),
            seed=specs["project_config"].get("random_seed", 99),
        )
        _replace_tree(output_dir / "coding_templates", expected_dir, force=force)

    return expected_dir


def _write_expected_time_templates(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir
        / "expected_outputs"
        / TEMPLATES_MODULE_DIR
        / TEMPLATE_OUTPUT_DIRS["times"]
    )
    transcript_table = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / TABULARIZE_OUTPUT_DIR
        / EXPECTED_WORKBOOK
    )

    with _scratch_dir(project_dir) as tmpdir:
        input_dir = _prepare_template_input(tmpdir, transcript_table)
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        make_speaking_time_template_files(
            input_dir=input_dir,
            output_dir=output_dir,
        )
        _replace_tree(output_dir / "coding_templates", expected_dir, force=force)

    return expected_dir


def _prepare_cu_input(tmpdir: Path, transcript_table: Path) -> Path:
    input_dir = tmpdir / "input"
    table_dir = input_dir / "transcript_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(transcript_table, table_dir / "transcript_tables.xlsx")
    return input_dir


def _cu_blinding_config(specs: dict[str, dict[str, Any]]) -> AdvancedConfig:
    return AdvancedConfig(**specs["advanced_config"])


def _fill_cu_workbook(path: Path, *, reliability: bool = False) -> None:
    df = pd.read_excel(path)
    sv_col = "sv"
    rel_col = "rel"
    if sv_col not in df.columns or rel_col not in df.columns:
        return

    df[sv_col] = df[sv_col].astype(object)
    df[rel_col] = df[rel_col].astype(object)
    if "speaker" in df.columns:
        codeable = df["speaker"].astype(str).str.upper() != "INV"
    else:
        codeable = df[sv_col].astype(str).str.upper() != "NA"
    df.loc[~codeable, [sv_col, rel_col]] = "NA"
    codeable_positions = list(df.index[codeable])
    for position, idx in enumerate(codeable_positions):
        sv_value = 1 if position % 3 != 1 else 0
        rel_value = 1 if position % 4 in {0, 1} else 0
        if reliability and position % 5 == 0:
            rel_value = 1 - rel_value
        df.at[idx, sv_col] = sv_value
        df.at[idx, rel_col] = rel_value

    df.to_excel(path, index=False)


def _write_expected_cu_files(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / CUS_MODULE_DIR / CU_OUTPUT_DIRS["files"]
    )
    transcript_table = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / TABULARIZE_OUTPUT_DIR
        / EXPECTED_WORKBOOK
    )

    metadata_fields = _metadata_fields(project_dir, specs["project_config"])
    with _scratch_dir(project_dir) as tmpdir:
        input_dir = _prepare_cu_input(tmpdir, transcript_table)
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(specs["project_config"].get("random_seed", 99))
        make_cu_coding_files(
            metadata_fields=metadata_fields,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            num_coders=specs["project_config"].get("num_coders", 0),
            input_dir=input_dir,
            output_dir=output_dir,
            cu_paradigms=specs["advanced_config"].get("cu_paradigms", []),
            exclude_participants=specs["project_config"].get("exclude_participants", []),
            stimulus_field=specs["project_config"].get("stimulus_field", ""),
            blinding_config=_cu_blinding_config(specs),
        )
        source = output_dir / "cu_coding"
        _replace_tree(source, expected_dir, force=force)

    cu_file = expected_dir / "cu_coding.xlsx"
    rel_file = expected_dir / "cu_reliability_coding.xlsx"
    _fill_cu_workbook(cu_file)
    if rel_file.exists():
        _fill_cu_workbook(rel_file, reliability=True)

    input_cu_dir = project_dir / specs["project_config"].get("input_dir", "input") / "cu_coding"
    if input_cu_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {input_cu_dir}")
    _replace_tree(expected_dir, input_cu_dir, force=force)
    return expected_dir


def _write_expected_cu_evaluation(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / CUS_MODULE_DIR / CU_OUTPUT_DIRS["evaluate"]
    )
    input_dir = project_dir / specs["project_config"].get("input_dir", "input") / "cu_coding"
    with _scratch_dir(project_dir) as tmpdir:
        evaluate_cu_reliability(
            input_dir=input_dir,
            output_dir=tmpdir,
            cu_paradigms=specs["advanced_config"].get("cu_paradigms", []),
        )
        _replace_tree(tmpdir / "cu_reliability", expected_dir, force=force)
    return expected_dir


def _write_expected_cu_reselection(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / CUS_MODULE_DIR / CU_OUTPUT_DIRS["reselect"]
    )
    input_dir = project_dir / specs["project_config"].get("input_dir", "input") / "cu_coding"
    metadata_fields = _metadata_fields(project_dir, specs["project_config"])
    with _scratch_dir(project_dir) as tmpdir:
        reselect_cu_rel(
            metadata_fields=metadata_fields,
            input_dir=input_dir,
            output_dir=tmpdir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            random_seed=specs["project_config"].get("random_seed", 99),
        )
        _replace_tree(tmpdir / "reselected_cu_coding_reliability", expected_dir, force=force)
    return expected_dir


def _write_expected_cu_analysis(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / CUS_MODULE_DIR / CU_OUTPUT_DIRS["analyze"]
    )
    input_dir = project_dir / specs["project_config"].get("input_dir", "input") / "cu_coding"
    with _scratch_dir(project_dir) as tmpdir:
        analyze_cu_coding(
            input_dir=input_dir,
            output_dir=tmpdir,
            cu_paradigms=specs["advanced_config"].get("cu_paradigms") or None,
            blinding_config=_cu_blinding_config(specs),
        )
        _replace_tree(tmpdir / "cu_coding_analysis", expected_dir, force=force)

    analysis_input_dir = (
        project_dir
        / specs["project_config"].get("input_dir", "input")
        / "cu_coding_analysis"
    )
    if analysis_input_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {analysis_input_dir}")
    _replace_tree(expected_dir, analysis_input_dir, force=force)
    return expected_dir


def _write_expected_cu_rates(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / CUS_MODULE_DIR / CU_OUTPUT_DIRS["rates"]
    )
    source_time = (
        project_dir
        / "expected_outputs"
        / TEMPLATES_MODULE_DIR
        / TEMPLATE_OUTPUT_DIRS["times"]
        / "speaking_times.xlsx"
    )
    input_dir = project_dir / specs["project_config"].get("input_dir", "input")
    time_input_dir = input_dir / "speaking_times"
    time_input_dir.mkdir(parents=True, exist_ok=True)
    time_input = time_input_dir / "speaking_times.xlsx"
    if not time_input.exists() or force:
        if source_time.resolve() != time_input.resolve():
            shutil.copyfile(source_time, time_input)
        times_df = pd.read_excel(time_input)
        if "speaking_time" in times_df.columns:
            times_df["speaking_time"] = [95, 88, 102][: len(times_df)]
        times_df.to_excel(time_input, index=False)

    with _scratch_dir(project_dir) as tmpdir:
        calculate_cu_rates(
            input_dir=input_dir,
            output_dir=tmpdir,
            cu_samples_file=specs["advanced_config"].get(
                "cu_samples_file",
                "cu_coding_by_sample_long.xlsx",
            ),
            speaking_time_file=specs["advanced_config"].get(
                "speaking_time_file",
                "speaking_times.xlsx",
            ),
            speaking_time_field=specs["advanced_config"].get(
                "speaking_time_field",
                "speaking_time",
            ),
        )
        _replace_tree(tmpdir / "cu_coding_analysis", expected_dir, force=force)
    return expected_dir


def _prepare_word_input(tmpdir: Path, transcript_table: Path) -> Path:
    input_dir = tmpdir / "input"
    table_dir = input_dir / "transcript_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(transcript_table, table_dir / "transcript_tables.xlsx")
    return input_dir


def _word_blinding_config(specs: dict[str, dict[str, Any]]) -> AdvancedConfig:
    return AdvancedConfig(**specs["advanced_config"])


def _vary_word_reliability(path: Path) -> None:
    df = pd.read_excel(path)
    if "word_count" not in df.columns:
        return

    numeric = pd.to_numeric(df["word_count"], errors="coerce")
    codeable_indices = list(df.index[numeric.notna()])
    for position, idx in enumerate(codeable_indices):
        value = int(numeric.loc[idx])
        if position % 4 == 0:
            value = max(0, value - 1)
        elif position % 5 == 0:
            value = value + 1
        df.at[idx, "word_count"] = value
    df.to_excel(path, index=False)


def _write_expected_word_files(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / WORDS_MODULE_DIR / WORD_OUTPUT_DIRS["files"]
    )
    transcript_table = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / TABULARIZE_OUTPUT_DIR
        / EXPECTED_WORKBOOK
    )

    with _scratch_dir(project_dir) as tmpdir:
        input_dir = _prepare_word_input(tmpdir, transcript_table)
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(specs["project_config"].get("random_seed", 99))
        make_word_count_files(
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            num_coders=specs["project_config"].get("num_coders", 0),
            input_dir=input_dir,
            output_dir=output_dir,
            exclude_participants=specs["project_config"].get("exclude_participants", []),
            blinding_config=_word_blinding_config(specs),
        )
        _replace_tree(output_dir / "word_counts", expected_dir, force=force)

    rel_file = expected_dir / "word_count_reliability.xlsx"
    if rel_file.exists():
        _vary_word_reliability(rel_file)

    input_word_dir = project_dir / specs["project_config"].get("input_dir", "input") / "word_counts"
    if input_word_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {input_word_dir}")
    _replace_tree(expected_dir, input_word_dir, force=force)
    return expected_dir


def _write_expected_word_evaluation(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / WORDS_MODULE_DIR / WORD_OUTPUT_DIRS["evaluate"]
    )
    input_dir = project_dir / specs["project_config"].get("input_dir", "input") / "word_counts"
    with _scratch_dir(project_dir) as tmpdir:
        evaluate_word_count_reliability(input_dir=input_dir, output_dir=tmpdir)
        _replace_tree(tmpdir / "word_count_reliability", expected_dir, force=force)
    return expected_dir


def _write_expected_word_reselection(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / WORDS_MODULE_DIR / WORD_OUTPUT_DIRS["reselect"]
    )
    input_dir = project_dir / specs["project_config"].get("input_dir", "input") / "word_counts"
    metadata_fields = _metadata_fields(project_dir, specs["project_config"])
    with _scratch_dir(project_dir) as tmpdir:
        reselect_wc_rel(
            metadata_fields=metadata_fields,
            input_dir=input_dir,
            output_dir=tmpdir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            random_seed=specs["project_config"].get("random_seed", 99),
        )
        _replace_tree(tmpdir / "reselected_word_count_reliability", expected_dir, force=force)
    return expected_dir


def _write_expected_word_analysis(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / WORDS_MODULE_DIR / WORD_OUTPUT_DIRS["analyze"]
    )
    input_dir = project_dir / specs["project_config"].get("input_dir", "input") / "word_counts"
    with _scratch_dir(project_dir) as tmpdir:
        analyze_word_counts(
            input_dir=input_dir,
            output_dir=tmpdir,
            word_count_file=specs["advanced_config"].get("word_count_file", "word_counting.xlsx"),
            word_count_field=specs["advanced_config"].get("word_count_field", "word_count"),
            blinding_config=_word_blinding_config(specs),
        )
        _replace_tree(tmpdir / "word_count_analysis", expected_dir, force=force)

    analysis_input_dir = (
        project_dir
        / specs["project_config"].get("input_dir", "input")
        / "word_count_analysis"
    )
    if analysis_input_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {analysis_input_dir}")
    _replace_tree(expected_dir, analysis_input_dir, force=force)
    return expected_dir


def _write_expected_word_rates(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / WORDS_MODULE_DIR / WORD_OUTPUT_DIRS["rates"]
    )
    source_time = (
        project_dir
        / specs["project_config"].get("input_dir", "input")
        / "speaking_times"
        / "speaking_times.xlsx"
    )
    if not source_time.exists():
        source_time = (
            project_dir
            / "expected_outputs"
            / TEMPLATES_MODULE_DIR
            / TEMPLATE_OUTPUT_DIRS["times"]
            / "speaking_times.xlsx"
        )

    input_dir = project_dir / specs["project_config"].get("input_dir", "input")
    time_input_dir = input_dir / "speaking_times"
    time_input_dir.mkdir(parents=True, exist_ok=True)
    time_input = time_input_dir / "speaking_times.xlsx"
    if not time_input.exists() or force:
        if source_time.resolve() != time_input.resolve():
            shutil.copyfile(source_time, time_input)
        times_df = pd.read_excel(time_input)
        if "speaking_time" in times_df.columns:
            times_df["speaking_time"] = [95, 88, 102][: len(times_df)]
        times_df.to_excel(time_input, index=False)

    with _scratch_dir(project_dir) as tmpdir:
        calculate_word_count_rates(
            input_dir=input_dir,
            output_dir=tmpdir,
            wc_samples_file=specs["advanced_config"].get(
                "wc_samples_file",
                "word_counting_by_sample.xlsx",
            ),
            speaking_time_file=specs["advanced_config"].get(
                "speaking_time_file",
                "speaking_times.xlsx",
            ),
            speaking_time_field=specs["advanced_config"].get(
                "speaking_time_field",
                "speaking_time",
            ),
        )
        _replace_tree(tmpdir / "word_count_analysis", expected_dir, force=force)
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
    _cleanup_obsolete_expected_dirs(project_dir, force=force)
    _write_expected_transcript_table(project_dir, specs, force=force)
    _write_expected_selection(project_dir, specs, force=force)
    _write_expected_evaluation(project_dir, specs, force=force)
    _write_expected_reselection(project_dir, specs, force=force)
    _write_expected_utterance_templates(project_dir, specs, force=force)
    _write_expected_sample_templates(project_dir, specs, force=force)
    _write_expected_time_templates(project_dir, specs, force=force)
    _write_expected_cu_files(project_dir, specs, force=force)
    _write_expected_cu_evaluation(project_dir, specs, force=force)
    _write_expected_cu_reselection(project_dir, specs, force=force)
    _write_expected_cu_analysis(project_dir, specs, force=force)
    _write_expected_cu_rates(project_dir, specs, force=force)
    _write_expected_word_files(project_dir, specs, force=force)
    _write_expected_word_evaluation(project_dir, specs, force=force)
    _write_expected_word_reselection(project_dir, specs, force=force)
    _write_expected_word_analysis(project_dir, specs, force=force)
    _write_expected_word_rates(project_dir, specs, force=force)
    return project_dir
