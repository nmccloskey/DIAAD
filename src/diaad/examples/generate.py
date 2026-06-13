from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from diaad.blinding.decode import decode_blinding
from diaad.blinding.encode import encode_blinding
from diaad.cli.commands import VALID_COMMANDS
from diaad.coding.compl_utts.analysis import analyze_cu_coding
from diaad.coding.compl_utts.files import make_cu_coding_files
from diaad.coding.compl_utts.rates import calculate_cu_rates
from diaad.coding.compl_utts.rel_evaluation import evaluate_cu_reliability
from diaad.coding.compl_utts.rel_reselection import reselect_cu_rel
from diaad.coding.powers.analysis import analyze_powers_coding
from diaad.coding.powers.files import make_powers_coding_files
from diaad.coding.powers.rates import calculate_powers_rates
from diaad.coding.powers.rel_evaluation import evaluate_powers_reliability
from diaad.coding.powers.rel_reselection import reselect_powers_rel
from diaad.coding.convo_turns.analysis import analyze_digital_convo_turns
from diaad.coding.convo_turns.files import make_digital_convo_turn_files
from diaad.coding.convo_turns.rel_evaluation import evaluate_digital_convo_turns_reliability
from diaad.coding.convo_turns.rel_reselection import reselect_digital_convo_turns_rel
from diaad.coding.target_vocab.analysis import run_target_vocab
from diaad.coding.target_vocab.files import (
    check_target_vocab_resources,
    make_target_vocab_file,
)
from diaad.coding.target_vocab.rates import calculate_target_vocab_rates
from diaad.coding.templates.samples import make_sample_template_files
from diaad.coding.templates.subset import make_sample_subset_file
from diaad.coding.templates.times import make_speaking_time_template_files
from diaad.coding.templates.utterances import make_utterance_template_files
from diaad.coding.word_counts.analysis import analyze_word_counts
from diaad.coding.word_counts.files import make_word_count_files
from diaad.coding.word_counts.rates import calculate_word_count_rates
from diaad.coding.word_counts.rel_evaluation import evaluate_word_count_reliability
from diaad.coding.word_counts.rel_reselection import reselect_wc_rel
from diaad.core.config import AdvancedConfig
from diaad.transcripts.cha_files import read_cha_files
from diaad.transcripts.detabularization import detabularize_transcripts
from diaad.transcripts.transcript_tables import tabularize_transcripts
from diaad.metadata.unblinding import unblind_dataframe
from diaad.transcripts.transcription_reliability_evaluation import (
    evaluate_transcription_reliability,
)
from diaad.transcripts.transcription_reliability_selection import (
    reselect_transcription_reliability_samples,
    select_transcription_reliability_samples,
)
from psair.examples import (
    ExampleAssets,
    replace_tree,
    scratch_dir as psair_scratch_dir,
    write_json,
    write_text,
    write_yaml,
)
from psair.metadata.metadata_fields import MetadataManager


SPEC_PACKAGE = "diaad.examples"
SPEC_ROOT = ("assets", "spec")
DEFAULT_TRANSCRIPT_TABLE_FILENAME = "transcript_tables.xlsx"
EXPECTED_WORKBOOK = DEFAULT_TRANSCRIPT_TABLE_FILENAME
TRANSCRIPTS_MODULE_DIR = "transcripts_module"
BLINDING_MODULE_DIR = "blinding_module"
TEMPLATES_MODULE_DIR = "templates_module"
CUS_MODULE_DIR = "cus_module"
WORDS_MODULE_DIR = "words_module"
POWERS_MODULE_DIR = "powers_module"
VOCAB_MODULE_DIR = "vocab_module"
TURNS_MODULE_DIR = "turns_module"
SELECT_OUTPUT_DIR = "transcripts_select"
EVALUATE_OUTPUT_DIR = "transcripts_evaluate"
RESELECT_OUTPUT_DIR = "transcripts_reselect"
TABULARIZE_OUTPUT_DIR = "transcripts_tabularize"
BLINDING_OUTPUT_DIRS = {
    "encode": "blinding_encode",
    "decode": "blinding_decode",
}
TEMPLATE_OUTPUT_DIRS = {
    "utterances": "templates_utterances",
    "samples": "templates_samples",
    "times": "templates_times",
    "subset": "templates_subset",
    "resubset": "templates_resubset",
}
TEMPLATE_SUBSET_INPUT_DIRS = {
    "subset": "sample_subset",
    "resubset": "sample_resubset",
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
POWERS_OUTPUT_DIRS = {
    "files": "powers_files",
    "evaluate": "powers_evaluate",
    "reselect": "powers_reselect",
    "analyze": "powers_analyze",
    "rates": "powers_rates",
}
VOCAB_OUTPUT_DIRS = {
    "file": "vocab_file",
    "check": "vocab_check",
    "analyze": "vocab_analyze",
    "rates": "vocab_rates",
}
TURNS_OUTPUT_DIRS = {
    "files": "turns_files",
    "evaluate": "turns_evaluate",
    "reselect": "turns_reselect",
    "analyze": "turns_analyze",
}
EXAMPLE_PACKAGE_PREFIX = "example_files_"
FULL_DATASET_SLUG = "full_dataset"


@dataclass(frozen=True)
class ExampleCapability:
    """Reusable input artifact that can satisfy one or more example commands."""

    name: str
    materialize: Callable[["ExampleBuildContext"], None]


@dataclass(frozen=True)
class ExampleCommandPlan:
    """Command-specific example plan with declared input capabilities."""

    command: str
    required_capabilities: tuple[str, ...]
    build_output: Callable[["ExampleBuildContext"], None]
    output_capabilities: tuple[str, ...] = ()


@dataclass
class ExampleBuildContext:
    """Mutable state shared while building a command-specific example package."""

    package_dir: Path
    specs: dict[str, dict[str, Any]]
    force: bool
    materialized_capabilities: set[str] = field(default_factory=set)

    @property
    def project_config(self) -> dict[str, Any]:
        project = dict(self.specs["project_config"])
        project["input_dir"] = "example_input"
        project["output_dir"] = "example_output"
        return project

    @property
    def example_config_dir(self) -> Path:
        return self.package_dir / "example_config"

    @property
    def example_input_dir(self) -> Path:
        return self.package_dir / self.project_config["input_dir"]

    @property
    def example_output_dir(self) -> Path:
        return self.package_dir / self.project_config["output_dir"]

    @property
    def example_logs_dir(self) -> Path:
        return self.package_dir / "example_logs"


def _transcript_table_filename(specs: dict[str, dict[str, Any]]) -> str:
    return specs["advanced_config"].get(
        "transcript_table_filename",
        DEFAULT_TRANSCRIPT_TABLE_FILENAME,
    )


def _metadata_source_filename(specs: dict[str, dict[str, Any]]) -> str:
    return specs["advanced_config"].get(
        "metadata_source",
        DEFAULT_TRANSCRIPT_TABLE_FILENAME,
    )


def _expected_transcript_workbook(project_dir: Path, specs: dict[str, dict[str, Any]]) -> Path:
    return (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / TABULARIZE_OUTPUT_DIR
        / _transcript_table_filename(specs)
    )


example_assets = ExampleAssets(SPEC_PACKAGE)


@contextmanager
def scratch_dir(parent):
    """
    Create example-generation scratch space outside the final package.

    The PSAIR helper creates ``_dx_*`` directories under the supplied parent.
    Keeping those temporary directories outside ``project_dir`` prevents
    transient cleanup failures from leaking scratch artifacts into user-facing
    example packages.
    """
    del parent
    with tempfile.TemporaryDirectory(
        prefix="diaad_examples_",
        ignore_cleanup_errors=True,
    ) as scratch_parent:
        with psair_scratch_dir(scratch_parent) as path:
            yield path


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
        "dataset": example_assets.read_yaml_mapping(*SPEC_ROOT, "dataset.yaml"),
        "project_config": example_assets.read_yaml_mapping(*SPEC_ROOT, "configs", "project.yaml"),
        "advanced_config": example_assets.read_yaml_mapping(*SPEC_ROOT, "configs", "advanced.yaml"),
        "vocab_resource": example_assets.read_yaml_mapping(*SPEC_ROOT, "vocab", "picnic_resource.yaml"),
        "template_subsets": example_assets.read_yaml_mapping(
            *SPEC_ROOT,
            "templates",
            "sample_subsets.yaml",
        ),
        "turns_sessions": example_assets.read_yaml_mapping(*SPEC_ROOT, "turns", "sessions.yaml"),
        "chat_files": example_assets.read_yaml_mapping(*SPEC_ROOT, "transcripts", "chat_files.yaml"),
        "reliability_chat_files": example_assets.read_yaml_mapping(
            *SPEC_ROOT,
            "transcripts",
            "reliability_chat_files.yaml",
        ),
        "expected_tables": example_assets.read_yaml_mapping(*SPEC_ROOT, "transcripts", "expected_tables.yaml"),
    }
    _validate_chat_spec(specs["chat_files"])
    _validate_reliability_spec(specs["chat_files"], specs["reliability_chat_files"])
    _validate_expected_tables(specs["chat_files"], specs["expected_tables"])
    return specs


def _ensure_writable_package(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(
            f"Refusing to write into non-empty directory without --force: {path}"
        )


def _normalize_example_commands(commands: Iterable[str] | str) -> tuple[str, ...]:
    if isinstance(commands, str):
        commands = [commands]

    normalized: list[str] = []
    seen: set[str] = set()
    for command in commands:
        value = command.strip().lower()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)

    invalid = [command for command in normalized if command not in VALID_COMMANDS]
    if invalid:
        raise ValueError(
            "Unknown DIAAD command(s) for examples: " + ", ".join(invalid)
        )

    unsupported = [
        command
        for command in normalized
        if command not in EXAMPLE_COMMAND_PLANS
    ]
    if unsupported:
        available = ", ".join(sorted(EXAMPLE_COMMAND_PLANS))
        raise ValueError(
            "DIAAD example plans are not yet available for: "
            + ", ".join(unsupported)
            + f". Available example commands: {available}"
        )

    if "templates subset" in normalized and len(normalized) > 1:
        raise ValueError(
            "The 'templates subset' command example must be generated by itself "
            "because the command expects exactly one Excel workbook under "
            "example_input."
        )

    return tuple(normalized)


def example_package_slug(commands: Iterable[str] | str | None = None) -> str:
    """Return the stable package slug for a full or command-specific example."""
    if commands is None:
        return FULL_DATASET_SLUG
    normalized = _normalize_example_commands(commands)
    if not normalized:
        return FULL_DATASET_SLUG
    return "_".join(command.replace(" ", "_") for command in normalized)


def example_package_name(commands: Iterable[str] | str | None = None) -> str:
    """Return the stable package directory name for the requested examples."""
    return f"{EXAMPLE_PACKAGE_PREFIX}{example_package_slug(commands)}"


def _required_capabilities_for_commands(commands: Iterable[str] | str) -> tuple[str, ...]:
    normalized = _normalize_example_commands(commands)
    required: list[str] = []
    seen: set[str] = set()
    available_from_prior_outputs: set[str] = set()
    for command in normalized:
        plan = EXAMPLE_COMMAND_PLANS[command]
        for capability_name in plan.required_capabilities:
            if capability_name in available_from_prior_outputs:
                continue
            if capability_name not in seen:
                required.append(capability_name)
                seen.add(capability_name)
        available_from_prior_outputs.update(plan.output_capabilities)
    return tuple(required)


def _materialize_capability(ctx: ExampleBuildContext, capability_name: str) -> None:
    if capability_name in ctx.materialized_capabilities:
        return
    capability = EXAMPLE_CAPABILITIES[capability_name]
    capability.materialize(ctx)
    ctx.materialized_capabilities.add(capability_name)


def _materialize_required_capabilities(
    ctx: ExampleBuildContext,
    commands: Iterable[str] | str,
) -> None:
    for capability_name in _required_capabilities_for_commands(commands):
        _materialize_capability(ctx, capability_name)


def _write_command_example_config(ctx: ExampleBuildContext) -> None:
    write_yaml(
        ctx.example_config_dir / "project.yaml",
        ctx.project_config,
        force=ctx.force,
    )
    write_yaml(
        ctx.example_config_dir / "advanced.yaml",
        ctx.specs["advanced_config"],
        force=ctx.force,
    )


def _command_list_text(commands: Iterable[str]) -> str:
    return "\n".join(f"- `{command}`" for command in commands)


def _write_command_example_readme(
    ctx: ExampleBuildContext,
    commands: tuple[str, ...],
) -> None:
    commands_text = _command_list_text(commands)
    example_command = ", ".join(commands)
    text = f"""# DIAAD Command Example Files

This package contains synthetic example input and output files for:

{commands_text}

Run from this directory with:

```powershell
diaad {example_command} --config example_config
```

Key folders:

- `example_config/`: runnable DIAAD configuration for this example package.
- `example_input/`: synthetic files needed to run the command(s).
- `example_output/`: representative output produced by the command(s).
- `example_logs/`: illustrative log output for this example package.
"""
    write_text(ctx.package_dir / "README.md", text, force=ctx.force)


def _write_command_example_logs(
    ctx: ExampleBuildContext,
    commands: tuple[str, ...],
) -> None:
    ctx.example_logs_dir.mkdir(parents=True, exist_ok=True)
    log_text = (
        "DIAAD example log\n"
        "=================\n\n"
        "This illustrative log accompanies a generated command-specific example "
        "package. Full `diaad examples` runs also write real run logs in the "
        "timestamped DIAAD output directory.\n\n"
        "Example command(s): " + ", ".join(commands) + "\n"
    )
    write_text(ctx.example_logs_dir / "diaad_example.log", log_text, force=ctx.force)


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
- `expected_outputs/blinding_module/`: outputs for standalone blinding commands.
- `expected_outputs/transcripts_module/`: outputs for transcript commands.
- `expected_outputs/templates_module/`: outputs for template commands.
- `expected_outputs/cus_module/`: outputs for complete-utterance coding commands.
- `expected_outputs/words_module/`: outputs for word-count commands.
- `expected_outputs/powers_module/`: outputs for POWERS commands.
- `expected_outputs/vocab_module/`: outputs for target-vocabulary commands.
- `expected_outputs/turns_module/`: outputs for digital conversation-turn commands.
"""
    write_text(project_dir / "README.md", text, force=force)


def _materialize_inputs(project_dir: Path, specs: dict[str, dict[str, Any]], *, force: bool) -> None:
    _write_readme(project_dir, specs["dataset"], force=force)
    write_yaml(project_dir / "config" / "project.yaml", specs["project_config"], force=force)
    write_yaml(project_dir / "config" / "advanced.yaml", specs["advanced_config"], force=force)

    obsolete_advanced_project = project_dir / "config" / "advanced_project.yaml"
    if force and obsolete_advanced_project.exists():
        obsolete_advanced_project.unlink()

    for chat in specs["chat_files"]["chat_files"]:
        write_text(
            project_dir / "input" / "chat" / chat["filename"],
            chat["content"].rstrip() + "\n",
            force=force,
        )

    for chat in specs["reliability_chat_files"]["reliability_chat_files"]:
        write_text(
            project_dir / "input" / "chat" / "reliability" / chat["filename"],
            chat["content"].rstrip() + "\n",
            force=force,
        )

    _write_sample_subset_inputs(project_dir, specs, force=force)


def _write_original_chat_inputs(
    input_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> None:
    for chat in specs["chat_files"]["chat_files"]:
        write_text(
            input_dir / "chat" / chat["filename"],
            chat["content"].rstrip() + "\n",
            force=force,
        )


def _write_reliability_chat_inputs(
    input_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> None:
    _write_original_chat_inputs(input_dir, specs, force=force)
    for chat in specs["reliability_chat_files"]["reliability_chat_files"]:
        write_text(
            input_dir / "chat" / "reliability" / chat["filename"],
            chat["content"].rstrip() + "\n",
            force=force,
        )


def _build_transcript_table_from_chat(
    *,
    input_dir: Path,
    output_dir: Path,
    specs: dict[str, dict[str, Any]],
) -> Path:
    metadata_config = {
        "tiers": specs["project_config"].get("metadata_fields", {}),
        "input_dir": input_dir,
    }
    metadata_fields = MetadataManager(metadata_config).metadata_fields
    chats = read_cha_files(
        input_dir=input_dir,
        shuffle=False,
        exclude_dirnames=[specs["advanced_config"].get("reliability_dirname", "reliability")],
    )
    written = tabularize_transcripts(
        metadata_fields=metadata_fields,
        chats=chats,
        output_dir=output_dir,
        shuffle=False,
        random_seed=specs["project_config"].get("random_seed", 99),
        transcript_table_filename=_transcript_table_filename(specs),
    )
    if not written:
        raise RuntimeError("Synthetic transcript tabularization did not write a workbook.")
    return Path(written[0])


def _materialize_chat_input_capability(ctx: ExampleBuildContext) -> None:
    _write_original_chat_inputs(ctx.example_input_dir, ctx.specs, force=ctx.force)


def _materialize_reliability_chat_input_capability(ctx: ExampleBuildContext) -> None:
    _write_reliability_chat_inputs(ctx.example_input_dir, ctx.specs, force=ctx.force)


def _materialize_transcript_table_capability(ctx: ExampleBuildContext) -> None:
    target_dir = ctx.example_input_dir / "transcript_tables"
    target = target_dir / _transcript_table_filename(ctx.specs)
    if target.exists() and not ctx.force:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")

    with scratch_dir(ctx.package_dir) as tmpdir:
        source_input_dir = tmpdir / "source_input"
        _write_original_chat_inputs(source_input_dir, ctx.specs, force=True)
        transcript_table = _build_transcript_table_from_chat(
            input_dir=source_input_dir,
            output_dir=tmpdir / "transcript_output",
            specs=ctx.specs,
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(transcript_table, target)


def _materialize_transcription_selection_capability(ctx: ExampleBuildContext) -> None:
    target_dir = ctx.example_input_dir / "transcription_reliability_selection"
    if target_dir.exists() and not ctx.force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {target_dir}")

    with scratch_dir(ctx.package_dir) as tmpdir:
        source_input_dir = tmpdir / "source_input"
        _write_original_chat_inputs(source_input_dir, ctx.specs, force=True)
        transcript_table = _build_transcript_table_from_chat(
            input_dir=source_input_dir,
            output_dir=tmpdir / "transcript_output",
            specs=ctx.specs,
        )
        table_dir = source_input_dir / "transcript_tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(transcript_table, table_dir / _transcript_table_filename(ctx.specs))
        _select_transcription_reliability_to_output(
            input_dir=source_input_dir,
            output_dir=tmpdir / "selection_output",
            specs=ctx.specs,
        )
        replace_tree(
            tmpdir / "selection_output" / "transcription_reliability_selection",
            target_dir,
            force=ctx.force,
        )


def _materialize_cu_coding_capability(ctx: ExampleBuildContext) -> None:
    target_dir = ctx.example_input_dir / "cu_coding"
    if target_dir.exists() and not ctx.force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {target_dir}")

    with scratch_dir(ctx.package_dir) as tmpdir:
        source_input_dir = tmpdir / "source_input"
        _write_original_chat_inputs(source_input_dir, ctx.specs, force=True)
        transcript_table = _build_transcript_table_from_chat(
            input_dir=source_input_dir,
            output_dir=tmpdir / "transcript_output",
            specs=ctx.specs,
        )
        cu_input_dir = _prepare_cu_input(
            tmpdir,
            transcript_table,
            _transcript_table_filename(ctx.specs),
        )
        cu_output_dir = tmpdir / "cu_output"
        cu_output_dir.mkdir(parents=True, exist_ok=True)
        make_cu_coding_files(
            metadata_fields=_metadata_fields_for_input(ctx.specs, cu_input_dir),
            frac=ctx.specs["project_config"].get("reliability_fraction", 0.34),
            num_coders=ctx.specs["project_config"].get("num_coders", 0),
            input_dir=cu_input_dir,
            output_dir=cu_output_dir,
            cu_paradigms=ctx.specs["advanced_config"].get("cu_paradigms", []),
            exclude_speakers=ctx.specs["project_config"].get("exclude_speakers", []),
            stimulus_field=ctx.specs["project_config"].get("stimulus_column", ""),
            blinding_config=_cu_blinding_config(ctx.specs),
            transcript_table_filename=_transcript_table_filename(ctx.specs),
        )
        replace_tree(cu_output_dir / "cu_coding", target_dir, force=ctx.force)

    cu_file = target_dir / "cu_coding.xlsx"
    rel_file = target_dir / "cu_reliability_coding.xlsx"
    _fill_cu_workbook(cu_file)
    if rel_file.exists():
        _fill_cu_workbook(rel_file, reliability=True)


def _materialize_word_count_capability(ctx: ExampleBuildContext) -> None:
    target_dir = ctx.example_input_dir / "word_counts"
    if target_dir.exists() and not ctx.force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {target_dir}")

    with scratch_dir(ctx.package_dir) as tmpdir:
        source_input_dir = tmpdir / "source_input"
        _write_original_chat_inputs(source_input_dir, ctx.specs, force=True)
        transcript_table = _build_transcript_table_from_chat(
            input_dir=source_input_dir,
            output_dir=tmpdir / "transcript_output",
            specs=ctx.specs,
        )
        word_input_dir = _prepare_word_input(
            tmpdir,
            transcript_table,
            _transcript_table_filename(ctx.specs),
        )
        word_output_dir = tmpdir / "word_output"
        word_output_dir.mkdir(parents=True, exist_ok=True)
        make_word_count_files(
            frac=ctx.specs["project_config"].get("reliability_fraction", 0.34),
            num_coders=ctx.specs["project_config"].get("num_coders", 0),
            input_dir=word_input_dir,
            output_dir=word_output_dir,
            exclude_speakers=ctx.specs["project_config"].get("exclude_speakers", []),
            blinding_config=_word_blinding_config(ctx.specs),
            transcript_table_filename=_transcript_table_filename(ctx.specs),
        )
        replace_tree(word_output_dir / "word_counts", target_dir, force=ctx.force)

    rel_file = target_dir / "word_count_reliability.xlsx"
    if rel_file.exists():
        _vary_word_reliability(rel_file)


def _materialize_sample_subset_input_capability(ctx: ExampleBuildContext) -> None:
    path = ctx.example_input_dir / _sample_subset_input_filename(ctx.specs, "subset")
    if path.exists() and not ctx.force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    ctx.example_input_dir.mkdir(parents=True, exist_ok=True)
    _sample_subset_dataframe(ctx.specs, "subset").to_excel(
        path,
        sheet_name="samples",
        index=False,
    )


def _materialize_speaking_time_capability(ctx: ExampleBuildContext) -> None:
    target_dir = ctx.example_input_dir / "speaking_times"
    if target_dir.exists() and not ctx.force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {target_dir}")
    with scratch_dir(ctx.package_dir) as tmpdir:
        source_input_dir = tmpdir / "source_input"
        _write_original_chat_inputs(source_input_dir, ctx.specs, force=True)
        transcript_table = _build_transcript_table_from_chat(
            input_dir=source_input_dir,
            output_dir=tmpdir / "transcript_output",
            specs=ctx.specs,
        )
        table_dir = source_input_dir / "transcript_tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(transcript_table, table_dir / _transcript_table_filename(ctx.specs))
        make_speaking_time_template_files(
            input_dir=source_input_dir,
            output_dir=tmpdir / "time_output",
            sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
            transcript_table_filename=_transcript_table_filename(ctx.specs),
        )
        replace_tree(tmpdir / "time_output" / "coding_templates", target_dir, force=ctx.force)
    time_input = target_dir / "speaking_times.xlsx"
    times_df = pd.read_excel(time_input)
    if "speaking_time" in times_df.columns:
        times_df["speaking_time"] = [95, 88, 102][: len(times_df)]
    times_df.to_excel(time_input, index=False)


def _materialize_cu_analysis_capability(ctx: ExampleBuildContext) -> None:
    _materialize_capability(ctx, "cu_coding_workbooks")
    target_dir = ctx.example_input_dir / "cu_coding_analysis"
    if target_dir.exists() and not ctx.force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {target_dir}")
    with scratch_dir(ctx.package_dir) as tmpdir:
        analyze_cu_coding(
            input_dir=ctx.example_input_dir,
            output_dir=tmpdir,
            cu_paradigms=ctx.specs["advanced_config"].get("cu_paradigms") or None,
            blinding_config=_cu_blinding_config(ctx.specs),
            exclude_speakers=ctx.specs["project_config"].get("exclude_speakers", []),
        )
        replace_tree(tmpdir / "cu_coding_analysis", target_dir, force=ctx.force)
    _unblind_sample_summary_if_needed(
        target_dir / ctx.specs["advanced_config"].get(
            "cu_samples_filename",
            "cu_coding_by_sample_long.xlsx",
        ),
        target_dir / "cu_analysis_blind_codebook.xlsx",
    )


def _materialize_word_count_analysis_capability(ctx: ExampleBuildContext) -> None:
    _materialize_capability(ctx, "word_count_workbooks")
    target_dir = ctx.example_input_dir / "word_count_analysis"
    if target_dir.exists() and not ctx.force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {target_dir}")
    with scratch_dir(ctx.package_dir) as tmpdir:
        analyze_word_counts(
            input_dir=ctx.example_input_dir,
            output_dir=tmpdir,
            word_count_file=ctx.specs["advanced_config"].get(
                "word_count_filename",
                "word_counting.xlsx",
            ),
            word_count_field=ctx.specs["advanced_config"].get(
                "word_count_column",
                "word_count",
            ),
            blinding_config=_word_blinding_config(ctx.specs),
            exclude_speakers=ctx.specs["project_config"].get("exclude_speakers", []),
        )
        replace_tree(tmpdir / "word_count_analysis", target_dir, force=ctx.force)
    _unblind_sample_summary_if_needed(
        target_dir / ctx.specs["advanced_config"].get(
            "wc_samples_filename",
            "word_counting_by_sample.xlsx",
        ),
        target_dir / "word_count_analysis_blind_codebook.xlsx",
    )


def _unblind_sample_summary_if_needed(summary_path: Path, codebook_path: Path) -> None:
    if not summary_path.exists() or not codebook_path.exists():
        return
    summary_df = pd.read_excel(summary_path)
    if "sample_id" in summary_df.columns or "sample_id_blinded" not in summary_df.columns:
        return
    codebook_df = pd.read_excel(codebook_path)
    unblind_dataframe(summary_df, codebook_df).to_excel(summary_path, index=False)


def _metadata_fields_for_input(
    specs: dict[str, dict[str, Any]],
    input_dir: Path,
) -> dict[str, Any]:
    metadata_config = {
        "tiers": specs["project_config"].get("metadata_fields", {}),
        "input_dir": input_dir,
    }
    return MetadataManager(metadata_config).metadata_fields


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
    transcript_table_filename = _transcript_table_filename(specs)
    target = expected_dir / transcript_table_filename
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")

    input_dir = project_dir / specs["project_config"].get("input_dir", "input")
    metadata_fields = _metadata_fields(project_dir, specs["project_config"])
    chats = read_cha_files(
        input_dir=input_dir,
        shuffle=False,
        exclude_dirnames=[specs["advanced_config"].get("reliability_dirname", "reliability")],
    )

    with scratch_dir(project_dir) as tmpdir:
        written = tabularize_transcripts(
            metadata_fields=metadata_fields,
            chats=chats,
            output_dir=tmpdir,
            shuffle=False,
            random_seed=specs["project_config"].get("random_seed", 99),
            transcript_table_filename=transcript_table_filename,
        )
        if not written:
            raise RuntimeError("Synthetic transcript tabularization did not write a workbook.")
        expected_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(written[0], target)

    return target


def _write_provided_transcript_table(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    transcript_table: Path,
    *,
    force: bool,
) -> Path:
    input_dir = project_dir / specs["project_config"].get("input_dir", "input")
    target = input_dir / "transcript_tables" / _transcript_table_filename(specs)
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(transcript_table, target)
    metadata_source = input_dir / "transcript_tables" / _metadata_source_filename(specs)
    if metadata_source != target:
        if metadata_source.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing file: {metadata_source}")
        shutil.copyfile(transcript_table, metadata_source)
    return target


def _sample_subset_input_dir(project_dir: Path, specs: dict[str, dict[str, Any]], mode: str) -> Path:
    return (
        project_dir
        / specs["project_config"].get("input_dir", "input")
        / TEMPLATE_SUBSET_INPUT_DIRS[mode]
    )


def _sample_subset_input_filename(specs: dict[str, dict[str, Any]], mode: str) -> str:
    return specs["template_subsets"][mode].get(
        "filename",
        f"sample_{mode}_input.xlsx",
    )


def _sample_subset_dataframe(specs: dict[str, dict[str, Any]], mode: str) -> pd.DataFrame:
    rows = specs["template_subsets"][mode].get("rows", [])
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Template sample subset spec for {mode!r} has no rows.")
    return df


def _write_sample_subset_input_workbook(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    mode: str,
    *,
    force: bool,
) -> Path:
    input_dir = _sample_subset_input_dir(project_dir, specs, mode)
    path = input_dir / _sample_subset_input_filename(specs, mode)
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    input_dir.mkdir(parents=True, exist_ok=True)
    _sample_subset_dataframe(specs, mode).to_excel(
        path,
        sheet_name="samples",
        index=False,
    )
    return path


def _write_sample_subset_inputs(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> None:
    for mode in ("subset", "resubset"):
        _write_sample_subset_input_workbook(project_dir, specs, mode, force=force)


def _cleanup_obsolete_expected_dirs(project_dir: Path, *, force: bool) -> None:
    if not force:
        return

    for dirname in (
        TABULARIZE_OUTPUT_DIR,
        SELECT_OUTPUT_DIR,
        EVALUATE_OUTPUT_DIR,
        RESELECT_OUTPUT_DIR,
        "blinding_encode",
        "blinding_decode",
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
        "powers_files",
        "powers_evaluate",
        "powers_reselect",
        "powers_analyze",
        "powers_rates",
        "vocab_file",
        "vocab_check",
        "vocab_analyze",
        "vocab_rates",
        "turns_files",
        "turns_evaluate",
        "turns_reselect",
        "turns_analyze",
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
    with scratch_dir(project_dir) as tmpdir:
        select_transcription_reliability_samples(
            metadata_fields=metadata_fields,
            chats=chats,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            output_dir=tmpdir,
            input_dir=input_dir,
            transcript_table_filename=_transcript_table_filename(specs),
        )
        source = tmpdir / "transcription_reliability_selection"
        replace_tree(source, expected_dir, force=force)

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


def _select_transcription_reliability_to_output(
    *,
    input_dir: Path,
    output_dir: Path,
    specs: dict[str, dict[str, Any]],
) -> None:
    metadata_fields = _metadata_fields_for_input(specs, input_dir)
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
    select_transcription_reliability_samples(
        metadata_fields=metadata_fields,
        chats=chats,
        frac=specs["project_config"].get("reliability_fraction", 0.34),
        output_dir=output_dir,
        input_dir=input_dir,
        transcript_table_filename=_transcript_table_filename(specs),
    )


def _write_expected_evaluation(project_dir: Path, specs: dict[str, dict[str, Any]], *, force: bool) -> Path:
    expected_dir = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / EVALUATE_OUTPUT_DIR
    )
    metadata_fields = _metadata_fields(project_dir, specs["project_config"])

    with scratch_dir(project_dir) as tmpdir:
        eval_input_dir = tmpdir / "input"
        shutil.copytree(project_dir / specs["project_config"].get("input_dir", "input"), eval_input_dir)

        evaluate_transcription_reliability(
            metadata_fields=metadata_fields,
            input_dir=eval_input_dir,
            output_dir=tmpdir,
            exclude_speakers=specs["project_config"].get("exclude_speakers", []),
            strip_clan=specs["project_config"].get("strip_clan", True),
            prefer_correction=specs["project_config"].get("prefer_correction", True),
            lowercase=specs["project_config"].get("lowercase", True),
            reliability_tag=specs["advanced_config"].get("reliability_tag", "_reliability"),
            reliability_dirname=specs["advanced_config"].get("reliability_dirname", "reliability"),
        )
        source = tmpdir / "transcription_reliability_evaluation"
        replace_tree(source, expected_dir, force=force)

    return expected_dir


def _write_expected_reselection(project_dir: Path, specs: dict[str, dict[str, Any]], *, force: bool) -> Path:
    expected_dir = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / RESELECT_OUTPUT_DIR
    )

    np.random.seed(specs["project_config"].get("random_seed", 99))
    with scratch_dir(project_dir) as tmpdir:
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
        replace_tree(source, expected_dir / "reselected_transcription_reliability", force=force)

    return expected_dir


def _prepare_template_input(
    tmpdir: Path,
    transcript_table: Path,
    transcript_table_filename: str,
) -> Path:
    input_dir = tmpdir / "input"
    table_dir = input_dir / "transcript_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(transcript_table, table_dir / transcript_table_filename)
    return input_dir


def _template_blinding_config(specs: dict[str, dict[str, Any]]) -> AdvancedConfig:
    return AdvancedConfig(**specs["advanced_config"])


def _build_utterance_templates_output(
    ctx: ExampleBuildContext,
    *,
    output_dir: Path,
) -> None:
    make_utterance_template_files(
        input_dir=ctx.example_input_dir,
        output_dir=output_dir,
        frac=ctx.specs["project_config"].get("reliability_fraction", 0.34),
        num_coders=ctx.specs["project_config"].get("num_coders", 0),
        stimulus_field=ctx.specs["project_config"].get("stimulus_column", ""),
        blinding_config=_template_blinding_config(ctx.specs),
        seed=ctx.specs["project_config"].get("random_seed", 99),
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
        utterance_id_field=ctx.specs["advanced_config"].get("utterance_id_column", "utterance_id"),
        transcript_table_filename=_transcript_table_filename(ctx.specs),
    )


def _build_sample_templates_output(
    ctx: ExampleBuildContext,
    *,
    output_dir: Path,
) -> None:
    make_sample_template_files(
        input_dir=ctx.example_input_dir,
        output_dir=output_dir,
        frac=ctx.specs["project_config"].get("reliability_fraction", 0.34),
        num_bins=ctx.specs["project_config"].get("num_bins", 2),
        num_coders=ctx.specs["project_config"].get("num_coders", 0),
        stimulus_field=ctx.specs["project_config"].get("stimulus_column", ""),
        blinding_config=_template_blinding_config(ctx.specs),
        seed=ctx.specs["project_config"].get("random_seed", 99),
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
        transcript_table_filename=_transcript_table_filename(ctx.specs),
    )


def _build_time_templates_output(
    ctx: ExampleBuildContext,
    *,
    output_dir: Path,
) -> None:
    make_speaking_time_template_files(
        input_dir=ctx.example_input_dir,
        output_dir=output_dir,
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
        transcript_table_filename=_transcript_table_filename(ctx.specs),
    )


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
        / _transcript_table_filename(specs)
    )

    with scratch_dir(project_dir) as tmpdir:
        input_dir = _prepare_template_input(
            tmpdir,
            transcript_table,
            _transcript_table_filename(specs),
        )
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(specs["project_config"].get("random_seed", 99))
        make_utterance_template_files(
            input_dir=input_dir,
            output_dir=output_dir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            num_coders=specs["project_config"].get("num_coders", 0),
            stimulus_field=specs["project_config"].get("stimulus_column", ""),
            blinding_config=_template_blinding_config(specs),
            seed=specs["project_config"].get("random_seed", 99),
            transcript_table_filename=_transcript_table_filename(specs),
        )
        replace_tree(output_dir / "coding_templates", expected_dir, force=force)

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
        / _transcript_table_filename(specs)
    )

    with scratch_dir(project_dir) as tmpdir:
        input_dir = _prepare_template_input(
            tmpdir,
            transcript_table,
            _transcript_table_filename(specs),
        )
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(specs["project_config"].get("random_seed", 99))
        make_sample_template_files(
            input_dir=input_dir,
            output_dir=output_dir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            num_bins=specs["project_config"].get("num_bins", 2),
            num_coders=specs["project_config"].get("num_coders", 0),
            stimulus_field=specs["project_config"].get("stimulus_column", ""),
            blinding_config=_template_blinding_config(specs),
            seed=specs["project_config"].get("random_seed", 99),
            transcript_table_filename=_transcript_table_filename(specs),
        )
        replace_tree(output_dir / "coding_templates", expected_dir, force=force)

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
        / _transcript_table_filename(specs)
    )

    with scratch_dir(project_dir) as tmpdir:
        input_dir = _prepare_template_input(
            tmpdir,
            transcript_table,
            _transcript_table_filename(specs),
        )
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        make_speaking_time_template_files(
            input_dir=input_dir,
            output_dir=output_dir,
            transcript_table_filename=_transcript_table_filename(specs),
        )
        replace_tree(output_dir / "coding_templates", expected_dir, force=force)

    return expected_dir


def _write_expected_sample_subset(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
    mode: str,
) -> Path:
    expected_dir = (
        project_dir
        / "expected_outputs"
        / TEMPLATES_MODULE_DIR
        / TEMPLATE_OUTPUT_DIRS[mode]
    )

    with scratch_dir(project_dir) as tmpdir:
        input_dir = tmpdir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        _sample_subset_dataframe(specs, mode).to_excel(
            input_dir / _sample_subset_input_filename(specs, mode),
            sheet_name="samples",
            index=False,
        )
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        make_sample_subset_file(
            input_dir=input_dir,
            output_dir=output_dir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            sample_id_field=specs["advanced_config"].get("sample_id_column", "sample_id"),
            seed=specs["project_config"].get("random_seed", 99),
        )
        replace_tree(output_dir / "coding_templates", expected_dir, force=force)

    return expected_dir


def _blinding_command_config(specs: dict[str, dict[str, Any]]) -> AdvancedConfig:
    return AdvancedConfig(
        auto_blind=True,
        blind_columns=["sample_id"],
        metadata_source=_metadata_source_filename(specs),
        id_columns=specs["advanced_config"].get("id_columns"),
    )


def _write_expected_blinding_encode(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir
        / "expected_outputs"
        / BLINDING_MODULE_DIR
        / BLINDING_OUTPUT_DIRS["encode"]
    )
    source = (
        project_dir
        / specs["project_config"].get("input_dir", "input")
        / "powers_coding"
        / "powers_coding.xlsx"
    )

    with scratch_dir(project_dir) as tmpdir:
        input_dir = tmpdir / "input" / "powers_coding"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, input_dir / source.name)
        encode_blinding(
            input_dir=input_dir,
            output_dir=tmpdir,
            blinding_config=_blinding_command_config(specs),
            seed=specs["project_config"].get("random_seed", 99),
        )
        replace_tree(tmpdir / "blinding", expected_dir, force=force)

    return expected_dir


def _write_expected_blinding_decode(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir
        / "expected_outputs"
        / BLINDING_MODULE_DIR
        / BLINDING_OUTPUT_DIRS["decode"]
    )
    input_root = project_dir / specs["project_config"].get("input_dir", "input")
    source = input_root / "cu_coding" / "cu_coding.xlsx"
    codebook = input_root / "cu_coding" / "cu_blind_codebook.xlsx"

    with scratch_dir(project_dir) as tmpdir:
        input_dir = tmpdir / "input" / "cu_coding"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, input_dir / source.name)
        shutil.copyfile(codebook, input_dir / codebook.name)
        decode_blinding(
            input_dir=input_dir,
            output_dir=tmpdir,
            blinding_config=_blinding_command_config(specs),
        )
        replace_tree(tmpdir / "blinding", expected_dir, force=force)

    return expected_dir


def _prepare_cu_input(
    tmpdir: Path,
    transcript_table: Path,
    transcript_table_filename: str,
) -> Path:
    input_dir = tmpdir / "input"
    table_dir = input_dir / "transcript_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(transcript_table, table_dir / transcript_table_filename)
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
        / _transcript_table_filename(specs)
    )

    metadata_fields = _metadata_fields(project_dir, specs["project_config"])
    with scratch_dir(project_dir) as tmpdir:
        input_dir = _prepare_cu_input(
            tmpdir,
            transcript_table,
            _transcript_table_filename(specs),
        )
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
            exclude_speakers=specs["project_config"].get("exclude_speakers", []),
            stimulus_field=specs["project_config"].get("stimulus_column", ""),
            blinding_config=_cu_blinding_config(specs),
            transcript_table_filename=_transcript_table_filename(specs),
        )
        source = output_dir / "cu_coding"
        replace_tree(source, expected_dir, force=force)

    cu_file = expected_dir / "cu_coding.xlsx"
    rel_file = expected_dir / "cu_reliability_coding.xlsx"
    _fill_cu_workbook(cu_file)
    if rel_file.exists():
        _fill_cu_workbook(rel_file, reliability=True)

    input_cu_dir = project_dir / specs["project_config"].get("input_dir", "input") / "cu_coding"
    if input_cu_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {input_cu_dir}")
    replace_tree(expected_dir, input_cu_dir, force=force)
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
    with scratch_dir(project_dir) as tmpdir:
        evaluate_cu_reliability(
            input_dir=input_dir,
            output_dir=tmpdir,
            cu_paradigms=specs["advanced_config"].get("cu_paradigms", []),
        )
        replace_tree(tmpdir / "cu_reliability", expected_dir, force=force)
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
    with scratch_dir(project_dir) as tmpdir:
        reselect_cu_rel(
            metadata_fields=metadata_fields,
            input_dir=input_dir,
            output_dir=tmpdir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            random_seed=specs["project_config"].get("random_seed", 99),
        )
        replace_tree(tmpdir / "reselected_cu_coding_reliability", expected_dir, force=force)
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
    with scratch_dir(project_dir) as tmpdir:
        analyze_cu_coding(
            input_dir=input_dir,
            output_dir=tmpdir,
            cu_paradigms=specs["advanced_config"].get("cu_paradigms") or None,
            blinding_config=_cu_blinding_config(specs),
            exclude_speakers=specs["project_config"].get("exclude_speakers", []),
        )
        replace_tree(tmpdir / "cu_coding_analysis", expected_dir, force=force)

    analysis_input_dir = (
        project_dir
        / specs["project_config"].get("input_dir", "input")
        / "cu_coding_analysis"
    )
    if analysis_input_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {analysis_input_dir}")
    replace_tree(expected_dir, analysis_input_dir, force=force)
    return expected_dir


def _build_transcripts_tabularize_example_output(ctx: ExampleBuildContext) -> None:
    _build_transcript_table_from_chat(
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        specs=ctx.specs,
    )


def _build_transcripts_chats_example_output(ctx: ExampleBuildContext) -> None:
    detabularize_transcripts(
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
        transcript_table_filename=_transcript_table_filename(ctx.specs),
    )


def _build_transcripts_select_example_output(ctx: ExampleBuildContext) -> None:
    _select_transcription_reliability_to_output(
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        specs=ctx.specs,
    )


def _build_transcripts_reselect_example_output(ctx: ExampleBuildContext) -> None:
    with scratch_dir(ctx.package_dir) as tmpdir:
        reselect_transcription_reliability_samples(
            input_dir=ctx.example_input_dir,
            output_dir=tmpdir,
            frac=ctx.specs["project_config"].get("reliability_fraction", 0.34),
        )
        replace_tree(
            tmpdir / "reselected_transcription_reliability",
            ctx.example_output_dir / "reselected_transcription_reliability",
            force=ctx.force,
        )


def _build_transcripts_evaluate_example_output(ctx: ExampleBuildContext) -> None:
    evaluate_transcription_reliability(
        metadata_fields=_metadata_fields_for_input(ctx.specs, ctx.example_input_dir),
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        exclude_speakers=ctx.specs["project_config"].get("exclude_speakers", []),
        strip_clan=ctx.specs["project_config"].get("strip_clan", True),
        prefer_correction=ctx.specs["project_config"].get("prefer_correction", True),
        lowercase=ctx.specs["project_config"].get("lowercase", True),
        reliability_tag=ctx.specs["advanced_config"].get("reliability_tag", "_reliability"),
        reliability_dirname=ctx.specs["advanced_config"].get("reliability_dirname", "reliability"),
    )


def _build_templates_utterances_example_output(ctx: ExampleBuildContext) -> None:
    _build_utterance_templates_output(ctx, output_dir=ctx.example_output_dir)


def _build_templates_samples_example_output(ctx: ExampleBuildContext) -> None:
    _build_sample_templates_output(ctx, output_dir=ctx.example_output_dir)


def _build_templates_times_example_output(ctx: ExampleBuildContext) -> None:
    _build_time_templates_output(ctx, output_dir=ctx.example_output_dir)


def _build_templates_subset_example_output(ctx: ExampleBuildContext) -> None:
    make_sample_subset_file(
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        frac=ctx.specs["project_config"].get("reliability_fraction", 0.34),
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
        seed=ctx.specs["project_config"].get("random_seed", 99),
    )


def _build_cu_files_example_output(ctx: ExampleBuildContext) -> None:
    make_cu_coding_files(
        metadata_fields=_metadata_fields_for_input(ctx.specs, ctx.example_input_dir),
        frac=ctx.specs["project_config"].get("reliability_fraction", 0.34),
        num_coders=ctx.specs["project_config"].get("num_coders", 0),
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        cu_paradigms=ctx.specs["advanced_config"].get("cu_paradigms", []),
        exclude_speakers=ctx.specs["project_config"].get("exclude_speakers", []),
        stimulus_field=ctx.specs["project_config"].get("stimulus_column", ""),
        blinding_config=_cu_blinding_config(ctx.specs),
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
        transcript_table_filename=_transcript_table_filename(ctx.specs),
    )


def _build_cu_reselect_example_output(ctx: ExampleBuildContext) -> None:
    reselect_cu_rel(
        metadata_fields=_metadata_fields_for_input(ctx.specs, ctx.example_input_dir),
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        frac=ctx.specs["project_config"].get("reliability_fraction", 0.34),
        random_seed=ctx.specs["project_config"].get("random_seed", 99),
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
    )


def _build_cu_evaluate_example_output(ctx: ExampleBuildContext) -> None:
    evaluate_cu_reliability(
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        cu_paradigms=ctx.specs["advanced_config"].get("cu_paradigms", []),
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
        utterance_id_field=ctx.specs["advanced_config"].get("utterance_id_column", "utterance_id"),
    )


def _build_cu_analyze_example_output(ctx: ExampleBuildContext) -> None:
    analyze_cu_coding(
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        cu_paradigms=ctx.specs["advanced_config"].get("cu_paradigms") or None,
        blinding_config=_cu_blinding_config(ctx.specs),
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
        exclude_speakers=ctx.specs["project_config"].get("exclude_speakers", []),
    )


def _build_cu_rates_example_output(ctx: ExampleBuildContext) -> None:
    calculate_cu_rates(
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        cu_samples_file=ctx.specs["advanced_config"].get(
            "cu_samples_filename",
            "cu_coding_by_sample_long.xlsx",
        ),
        speaking_time_file=ctx.specs["advanced_config"].get(
            "speaking_time_filename",
            "speaking_times.xlsx",
        ),
        speaking_time_field=ctx.specs["advanced_config"].get(
            "speaking_time_column",
            "speaking_time",
        ),
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
    )


def _build_word_files_example_output(ctx: ExampleBuildContext) -> None:
    make_word_count_files(
        frac=ctx.specs["project_config"].get("reliability_fraction", 0.34),
        num_coders=ctx.specs["project_config"].get("num_coders", 0),
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        exclude_speakers=ctx.specs["project_config"].get("exclude_speakers", []),
        blinding_config=_word_blinding_config(ctx.specs),
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
        utterance_id_field=ctx.specs["advanced_config"].get("utterance_id_column", "utterance_id"),
        transcript_table_filename=_transcript_table_filename(ctx.specs),
    )


def _build_word_evaluate_example_output(ctx: ExampleBuildContext) -> None:
    evaluate_word_count_reliability(
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
        utterance_id_field=ctx.specs["advanced_config"].get("utterance_id_column", "utterance_id"),
    )


def _build_word_reselect_example_output(ctx: ExampleBuildContext) -> None:
    reselect_wc_rel(
        metadata_fields=_metadata_fields_for_input(ctx.specs, ctx.example_input_dir),
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        frac=ctx.specs["project_config"].get("reliability_fraction", 0.34),
        random_seed=ctx.specs["project_config"].get("random_seed", 99),
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
    )


def _build_word_analyze_example_output(ctx: ExampleBuildContext) -> None:
    analyze_word_counts(
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        word_count_file=ctx.specs["advanced_config"].get(
            "word_count_filename",
            "word_counting.xlsx",
        ),
        word_count_field=ctx.specs["advanced_config"].get(
            "word_count_column",
            "word_count",
        ),
        blinding_config=_word_blinding_config(ctx.specs),
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
        exclude_speakers=ctx.specs["project_config"].get("exclude_speakers", []),
    )


def _build_word_rates_example_output(ctx: ExampleBuildContext) -> None:
    calculate_word_count_rates(
        input_dir=ctx.example_input_dir,
        output_dir=ctx.example_output_dir,
        wc_samples_file=ctx.specs["advanced_config"].get(
            "wc_samples_filename",
            "word_counting_by_sample.xlsx",
        ),
        speaking_time_file=ctx.specs["advanced_config"].get(
            "speaking_time_filename",
            "speaking_times.xlsx",
        ),
        speaking_time_field=ctx.specs["advanced_config"].get(
            "speaking_time_column",
            "speaking_time",
        ),
        sample_id_field=ctx.specs["advanced_config"].get("sample_id_column", "sample_id"),
    )


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
    cu_summary_input = (
        input_dir
        / "cu_coding_analysis"
        / specs["advanced_config"].get("cu_samples_filename", "cu_coding_by_sample_long.xlsx")
    )
    cu_analysis_codebook = input_dir / "cu_coding_analysis" / "cu_analysis_blind_codebook.xlsx"
    if cu_summary_input.exists() and cu_analysis_codebook.exists():
        cu_summary_df = pd.read_excel(cu_summary_input)
        if "sample_id" not in cu_summary_df.columns and "sample_id_blinded" in cu_summary_df.columns:
            codebook_df = pd.read_excel(cu_analysis_codebook)
            unblind_dataframe(cu_summary_df, codebook_df).to_excel(cu_summary_input, index=False)

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

    with scratch_dir(project_dir) as tmpdir:
        calculate_cu_rates(
            input_dir=input_dir,
            output_dir=tmpdir,
            cu_samples_file=specs["advanced_config"].get(
                "cu_samples_filename",
                "cu_coding_by_sample_long.xlsx",
            ),
            speaking_time_file=specs["advanced_config"].get(
                "speaking_time_filename",
                "speaking_times.xlsx",
            ),
            speaking_time_field=specs["advanced_config"].get(
                "speaking_time_column",
                "speaking_time",
            ),
        )
        replace_tree(tmpdir / "cu_coding_analysis", expected_dir, force=force)
    return expected_dir


def _prepare_word_input(
    tmpdir: Path,
    transcript_table: Path,
    transcript_table_filename: str,
) -> Path:
    input_dir = tmpdir / "input"
    table_dir = input_dir / "transcript_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(transcript_table, table_dir / transcript_table_filename)
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
        / _transcript_table_filename(specs)
    )

    with scratch_dir(project_dir) as tmpdir:
        input_dir = _prepare_word_input(
            tmpdir,
            transcript_table,
            _transcript_table_filename(specs),
        )
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(specs["project_config"].get("random_seed", 99))
        make_word_count_files(
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            num_coders=specs["project_config"].get("num_coders", 0),
            input_dir=input_dir,
            output_dir=output_dir,
            exclude_speakers=specs["project_config"].get("exclude_speakers", []),
            blinding_config=_word_blinding_config(specs),
            transcript_table_filename=_transcript_table_filename(specs),
        )
        replace_tree(output_dir / "word_counts", expected_dir, force=force)

    rel_file = expected_dir / "word_count_reliability.xlsx"
    if rel_file.exists():
        _vary_word_reliability(rel_file)

    input_word_dir = project_dir / specs["project_config"].get("input_dir", "input") / "word_counts"
    if input_word_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {input_word_dir}")
    replace_tree(expected_dir, input_word_dir, force=force)
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
    with scratch_dir(project_dir) as tmpdir:
        evaluate_word_count_reliability(input_dir=input_dir, output_dir=tmpdir)
        replace_tree(tmpdir / "word_count_reliability", expected_dir, force=force)
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
    with scratch_dir(project_dir) as tmpdir:
        reselect_wc_rel(
            metadata_fields=metadata_fields,
            input_dir=input_dir,
            output_dir=tmpdir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            random_seed=specs["project_config"].get("random_seed", 99),
        )
        replace_tree(tmpdir / "reselected_word_count_reliability", expected_dir, force=force)
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
    with scratch_dir(project_dir) as tmpdir:
        analyze_word_counts(
            input_dir=input_dir,
            output_dir=tmpdir,
            word_count_file=specs["advanced_config"].get("word_count_filename", "word_counting.xlsx"),
            word_count_field=specs["advanced_config"].get("word_count_column", "word_count"),
            blinding_config=_word_blinding_config(specs),
            exclude_speakers=specs["project_config"].get("exclude_speakers", []),
        )
        replace_tree(tmpdir / "word_count_analysis", expected_dir, force=force)

    analysis_input_dir = (
        project_dir
        / specs["project_config"].get("input_dir", "input")
        / "word_count_analysis"
    )
    if analysis_input_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {analysis_input_dir}")
    replace_tree(expected_dir, analysis_input_dir, force=force)
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
    wc_summary_input = (
        input_dir
        / "word_count_analysis"
        / specs["advanced_config"].get("wc_samples_filename", "word_counting_by_sample.xlsx")
    )
    wc_analysis_codebook = (
        input_dir / "word_count_analysis" / "word_count_analysis_blind_codebook.xlsx"
    )
    if wc_summary_input.exists() and wc_analysis_codebook.exists():
        wc_summary_df = pd.read_excel(wc_summary_input)
        if "sample_id" not in wc_summary_df.columns and "sample_id_blinded" in wc_summary_df.columns:
            codebook_df = pd.read_excel(wc_analysis_codebook)
            unblind_dataframe(wc_summary_df, codebook_df).to_excel(wc_summary_input, index=False)

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

    with scratch_dir(project_dir) as tmpdir:
        calculate_word_count_rates(
            input_dir=input_dir,
            output_dir=tmpdir,
            wc_samples_file=specs["advanced_config"].get(
                "wc_samples_filename",
                "word_counting_by_sample.xlsx",
            ),
            speaking_time_file=specs["advanced_config"].get(
                "speaking_time_filename",
                "speaking_times.xlsx",
            ),
            speaking_time_field=specs["advanced_config"].get(
                "speaking_time_column",
                "speaking_time",
            ),
        )
        replace_tree(tmpdir / "word_count_analysis", expected_dir, force=force)
    return expected_dir


def _prepare_powers_input(
    tmpdir: Path,
    transcript_table: Path,
    transcript_table_filename: str,
) -> Path:
    input_dir = tmpdir / "input"
    table_dir = input_dir / "transcript_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(transcript_table, table_dir / transcript_table_filename)
    return input_dir


def _powers_blinding_config(specs: dict[str, dict[str, Any]]) -> AdvancedConfig:
    return AdvancedConfig(**specs["advanced_config"])


def _fill_powers_utterance_values(df: pd.DataFrame, *, reliability: bool = False) -> pd.DataFrame:
    df = df.copy()
    for col in [
        "POWERS_comment",
        "turn_type",
        "comments",
        "collab_repair",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("object")
    turn_types = ["T", "T", "ST", "MT", "T", "NV"]
    repair_values = ["", "", "repair_1", "", "", ""]

    for pos, idx in enumerate(df.index):
        speaker = str(df.at[idx, "speaker"]) if "speaker" in df.columns else ""
        utterance = str(df.at[idx, "utterance"]) if "utterance" in df.columns else ""
        words = [token for token in utterance.replace(".", "").replace("?", "").split() if token]
        base_count = max(1, len(words))
        if speaker == "INV":
            base_count = max(1, base_count - 1)

        df.at[idx, "speech_units"] = 1 + (pos % 2)
        df.at[idx, "turn_type"] = turn_types[pos % len(turn_types)]
        df.at[idx, "content_words"] = base_count
        df.at[idx, "num_nouns"] = max(0, min(3, base_count // 2))
        df.at[idx, "circumlocutions"] = 1 + (pos % 3)
        df.at[idx, "sem_paras"] = 1 + (pos % 4)
        df.at[idx, "phon_errs"] = 1 + (pos % 2)
        df.at[idx, "neologisms"] = 1 + (pos % 5)
        df.at[idx, "comments"] = ""
        df.at[idx, "lg_pauses"] = 1 + (pos % 3)
        df.at[idx, "filled_pauses"] = 1 + (pos % 2)
        df.at[idx, "collab_repair"] = repair_values[pos % len(repair_values)]

        if reliability:
            if pos % 5 == 0:
                df.at[idx, "content_words"] = max(0, int(df.at[idx, "content_words"]) - 1)
            if pos % 6 == 0:
                df.at[idx, "turn_type"] = "T"
            if pos % 8 == 0:
                df.at[idx, "filled_pauses"] = int(df.at[idx, "filled_pauses"]) + 1

    return df


def _fill_powers_section_e_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["type_of_day", "other_notes"]:
        if col in df.columns:
            df[col] = df[col].astype("object")
    days = ["weekday", "weekday", "weekend"]
    enjoyment = [4, 3, 4]
    difficulty = [2, 2, 1]
    for pos, idx in enumerate(df.index):
        df.at[idx, "type_of_day"] = days[pos % len(days)]
        df.at[idx, "amount_of_enjoyment"] = enjoyment[pos % len(enjoyment)]
        df.at[idx, "degree_of_difficulty"] = difficulty[pos % len(difficulty)]
        df.at[idx, "other_notes"] = ""
    return df


def _fill_powers_coding_workbook(path: Path) -> None:
    utterances = pd.read_excel(path, sheet_name="utterance_coding")
    section_e = pd.read_excel(path, sheet_name="section_e")
    utterances = _fill_powers_utterance_values(utterances)
    section_e = _fill_powers_section_e_values(section_e)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        utterances.to_excel(writer, sheet_name="utterance_coding", index=False)
        section_e.to_excel(writer, sheet_name="section_e", index=False)


def _fill_powers_reliability_workbook(path: Path) -> None:
    df = pd.read_excel(path)
    df = _fill_powers_utterance_values(df, reliability=True)
    df.to_excel(path, index=False)


def _write_expected_powers_files(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / POWERS_MODULE_DIR / POWERS_OUTPUT_DIRS["files"]
    )
    transcript_table = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / TABULARIZE_OUTPUT_DIR
        / _transcript_table_filename(specs)
    )

    with scratch_dir(project_dir) as tmpdir:
        input_dir = _prepare_powers_input(
            tmpdir,
            transcript_table,
            _transcript_table_filename(specs),
        )
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(specs["project_config"].get("random_seed", 99))
        make_powers_coding_files(
            metadata_fields=_metadata_fields(project_dir, specs["project_config"]),
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            num_coders=specs["project_config"].get("num_coders", 0),
            input_dir=input_dir,
            output_dir=output_dir,
            exclude_speakers=specs["project_config"].get("exclude_speakers", []),
            automate_powers=specs["project_config"].get("automate_powers", False),
            blinding_config=_powers_blinding_config(specs),
            transcript_table_filename=_transcript_table_filename(specs),
        )
        replace_tree(output_dir / "powers_coding", expected_dir, force=force)

    coding_file = next(expected_dir.rglob("powers_coding.xlsx"), None)
    reliability_file = next(expected_dir.rglob("powers_reliability_coding.xlsx"), None)
    if coding_file is not None:
        _fill_powers_coding_workbook(coding_file)
    if reliability_file is not None:
        _fill_powers_reliability_workbook(reliability_file)

    input_powers_dir = project_dir / specs["project_config"].get("input_dir", "input") / "powers_coding"
    if input_powers_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {input_powers_dir}")
    replace_tree(expected_dir, input_powers_dir, force=force)
    return expected_dir


def _write_expected_powers_evaluation(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / POWERS_MODULE_DIR / POWERS_OUTPUT_DIRS["evaluate"]
    )
    input_dir = project_dir / specs["project_config"].get("input_dir", "input") / "powers_coding"
    with scratch_dir(project_dir) as tmpdir:
        evaluate_powers_reliability(input_dir=input_dir, output_dir=tmpdir)
        replace_tree(tmpdir / "powers_reliability", expected_dir, force=force)
    return expected_dir


def _write_expected_powers_reselection(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / POWERS_MODULE_DIR / POWERS_OUTPUT_DIRS["reselect"]
    )
    input_dir = project_dir / specs["project_config"].get("input_dir", "input") / "powers_coding"
    metadata_fields = _metadata_fields(project_dir, specs["project_config"])
    with scratch_dir(project_dir) as tmpdir:
        reselect_powers_rel(
            metadata_fields=metadata_fields,
            input_dir=input_dir,
            output_dir=tmpdir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            random_seed=specs["project_config"].get("random_seed", 99),
            automate_powers=specs["project_config"].get("automate_powers", False),
        )
        replace_tree(tmpdir / "reselected_powers_reliability", expected_dir, force=force)
    return expected_dir


def _write_expected_powers_analysis(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / POWERS_MODULE_DIR / POWERS_OUTPUT_DIRS["analyze"]
    )
    input_dir = project_dir / specs["project_config"].get("input_dir", "input") / "powers_coding"
    with scratch_dir(project_dir) as tmpdir:
        analyze_powers_coding(
            input_dir=input_dir,
            output_dir=tmpdir,
            blinding_config=_powers_blinding_config(specs),
        )
        replace_tree(tmpdir / "powers_coding_analysis", expected_dir, force=force)

    analysis_input_dir = (
        project_dir
        / specs["project_config"].get("input_dir", "input")
        / "powers_coding_analysis"
    )
    if analysis_input_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {analysis_input_dir}")
    replace_tree(expected_dir, analysis_input_dir, force=force)
    return expected_dir


def _write_expected_powers_rates(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / POWERS_MODULE_DIR / POWERS_OUTPUT_DIRS["rates"]
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

    with scratch_dir(project_dir) as tmpdir:
        calculate_powers_rates(
            input_dir=input_dir,
            output_dir=tmpdir,
            speaking_time_file=specs["advanced_config"].get(
                "speaking_time_filename",
                "speaking_times.xlsx",
            ),
            speaking_time_field=specs["advanced_config"].get(
                "speaking_time_column",
                "speaking_time",
            ),
        )
        replace_tree(tmpdir / "powers_coding_analysis", expected_dir, force=force)
    return expected_dir


def _vocab_resource_path(project_dir: Path, specs: dict[str, dict[str, Any]]) -> Path:
    configured = specs["advanced_config"].get(
        "target_vocabulary_resource_path",
        "input/target_vocab/resources/picnic_target_vocab.json",
    )
    path = Path(configured)
    if path.is_absolute():
        return path
    return project_dir / path


def _write_vocab_resource(project_dir: Path, specs: dict[str, dict[str, Any]], *, force: bool) -> Path:
    resource_path = _vocab_resource_path(project_dir, specs)
    if not resource_path.exists() or force:
        write_json(resource_path, specs["vocab_resource"], force=force)
    return resource_path


def _write_vocab_unblind_input(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    workbook = _expected_transcript_workbook(project_dir, specs)
    utt_df = pd.read_excel(workbook, sheet_name="utterances")
    samples_df = pd.read_excel(workbook, sheet_name="samples")
    sample_metadata_cols = [
        col
        for col in ["sample_id", "participant_id", "stimulus", "timepoint"]
        if col in samples_df.columns
    ]
    if sample_metadata_cols and set(sample_metadata_cols) != {"sample_id"}:
        utt_df = utt_df.merge(
            samples_df[sample_metadata_cols].drop_duplicates(),
            on="sample_id",
            how="left",
        )
    exclude = set(specs["project_config"].get("exclude_speakers", []))
    if "speaker" in utt_df.columns and exclude:
        utt_df = utt_df[~utt_df["speaker"].isin(exclude)].copy()

    speaking_times = {
        sample_id: value
        for sample_id, value in zip(
            sorted(utt_df["sample_id"].dropna().unique()),
            [95, 88, 102],
        )
    }
    utt_df["speaking_time"] = utt_df["sample_id"].map(speaking_times)
    utt_df["word_count"] = (
        utt_df["utterance"]
        .fillna("")
        .astype(str)
        .str.replace(r"[^\w\s']", " ", regex=True)
        .str.split()
        .str.len()
    )

    keep_cols = [
        col
        for col in [
            "sample_id",
            "stimulus",
            "timepoint",
            "utterance_id",
            "speaker",
            "utterance",
            "speaking_time",
            "word_count",
        ]
        if col in utt_df.columns
    ]
    out_path = (
        project_dir
        / specs["project_config"].get("input_dir", "input")
        / "target_vocab"
        / "unblind_utterance_data.xlsx"
    )
    if out_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    utt_df[keep_cols].to_excel(out_path, index=False)
    return out_path


def _write_expected_vocab_file(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    del specs
    expected_dir = (
        project_dir / "expected_outputs" / VOCAB_MODULE_DIR / VOCAB_OUTPUT_DIRS["file"]
    )
    with scratch_dir(project_dir) as tmpdir:
        make_target_vocab_file(input_dir=tmpdir / "input", output_dir=tmpdir)
        replace_tree(tmpdir / "target_vocab", expected_dir, force=force)
    return expected_dir


def _write_expected_vocab_check(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / VOCAB_MODULE_DIR / VOCAB_OUTPUT_DIRS["check"]
    )
    resource_path = _write_vocab_resource(project_dir, specs, force=force)

    with scratch_dir(project_dir) as tmpdir:
        check_target_vocab_resources(resource_path=resource_path, output_dir=tmpdir)
        replace_tree(tmpdir / "target_vocab", expected_dir, force=force)

    return expected_dir


def _stable_target_vocab_analysis_name(path: Path) -> Path:
    return path.with_name("target_vocab_data_260101_0000.xlsx")


def _write_expected_vocab_analysis(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / VOCAB_MODULE_DIR / VOCAB_OUTPUT_DIRS["analyze"]
    )
    resource_path = _write_vocab_resource(project_dir, specs, force=force)
    _write_vocab_unblind_input(project_dir, specs, force=force)
    input_dir = project_dir / specs["project_config"].get("input_dir", "input")
    metadata_fields = _metadata_fields(project_dir, specs["project_config"])

    with scratch_dir(project_dir) as tmpdir:
        run_target_vocab(
            metadata_fields=metadata_fields,
            input_dir=input_dir,
            output_dir=tmpdir,
            exclude_speakers=specs["project_config"].get("exclude_speakers", []),
            stimulus_field=specs["project_config"].get("stimulus_column", "stimulus"),
            resource_path=resource_path,
            transcript_table_filename=_transcript_table_filename(specs),
        )
        output_dir = tmpdir / "target_vocab"
        generated = sorted(output_dir.glob("target_vocab_data_*.xlsx"))
        if not generated:
            raise FileNotFoundError("Target vocabulary analysis did not produce a workbook.")
        stable = _stable_target_vocab_analysis_name(generated[0])
        generated[0].replace(stable)
        replace_tree(output_dir, expected_dir, force=force)

    analysis_input_dir = input_dir / "target_vocab_analysis"
    if analysis_input_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing directory: {analysis_input_dir}")
    replace_tree(expected_dir, analysis_input_dir, force=force)
    return expected_dir


def _write_expected_vocab_rates(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / VOCAB_MODULE_DIR / VOCAB_OUTPUT_DIRS["rates"]
    )
    input_dir = project_dir / specs["project_config"].get("input_dir", "input")
    with scratch_dir(project_dir) as tmpdir:
        calculate_target_vocab_rates(input_dir=input_dir, output_dir=tmpdir)
        replace_tree(tmpdir / "target_vocab", expected_dir, force=force)
    return expected_dir


def _turns_input_dir(project_dir: Path, specs: dict[str, dict[str, Any]]) -> Path:
    return (
        project_dir
        / specs["project_config"].get("input_dir", "input")
        / "conversation_turns"
    )


def _turns_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["sample_id", "session", "bin", "turns"])
    if df.empty:
        raise ValueError("Turns example rows must not be empty.")
    required = {"sample_id", "session", "bin", "turns"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Turns example rows are missing required columns: {sorted(missing)}")
    return df


def _write_turns_coding_inputs(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    turns_dir = _turns_input_dir(project_dir, specs)
    turns_dir.mkdir(parents=True, exist_ok=True)
    primary_path = turns_dir / "conversation_turns_template.xlsx"
    reliability_path = turns_dir / "conversation_turns_reliability_template.xlsx"

    if not force and (primary_path.exists() or reliability_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing turns input files in: {turns_dir}")

    primary_df = _turns_dataframe(specs["turns_sessions"].get("primary_rows", []))
    reliability_df = _turns_dataframe(specs["turns_sessions"].get("reliability_rows", []))
    primary_df.to_excel(primary_path, index=False)
    reliability_df.to_excel(reliability_path, index=False)
    return turns_dir


def _write_expected_turn_files(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / TURNS_MODULE_DIR / TURNS_OUTPUT_DIRS["files"]
    )
    transcript_table = (
        project_dir
        / "expected_outputs"
        / TRANSCRIPTS_MODULE_DIR
        / TABULARIZE_OUTPUT_DIR
        / _transcript_table_filename(specs)
    )

    with scratch_dir(project_dir) as tmpdir:
        input_dir = _prepare_template_input(
            tmpdir,
            transcript_table,
            _transcript_table_filename(specs),
        )
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        random.seed(specs["project_config"].get("random_seed", 99))
        make_digital_convo_turn_files(
            input_dir=input_dir,
            output_dir=output_dir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            num_bins=specs["project_config"].get("num_bins", 2),
            num_coders=specs["project_config"].get("num_coders", 0),
            blinding_config=_template_blinding_config(specs),
            seed=specs["project_config"].get("random_seed", 99),
            transcript_table_filename=_transcript_table_filename(specs),
        )
        replace_tree(output_dir / "coding_templates", expected_dir, force=force)

    return expected_dir


def _write_expected_turn_evaluation(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / TURNS_MODULE_DIR / TURNS_OUTPUT_DIRS["evaluate"]
    )
    input_dir = _write_turns_coding_inputs(project_dir, specs, force=force)
    metadata_fields = _metadata_fields(project_dir, specs["project_config"])

    with scratch_dir(project_dir) as tmpdir:
        evaluate_digital_convo_turns_reliability(
            metadata_fields=metadata_fields,
            input_dir=input_dir,
            output_dir=tmpdir,
        )
        replace_tree(tmpdir / "turns_reliability", expected_dir, force=force)

    return expected_dir


def _write_expected_turn_reselection(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / TURNS_MODULE_DIR / TURNS_OUTPUT_DIRS["reselect"]
    )
    input_dir = _turns_input_dir(project_dir, specs)
    metadata_fields = _metadata_fields(project_dir, specs["project_config"])

    with scratch_dir(project_dir) as tmpdir:
        reselect_digital_convo_turns_rel(
            metadata_fields=metadata_fields,
            input_dir=input_dir,
            output_dir=tmpdir,
            frac=specs["project_config"].get("reliability_fraction", 0.34),
            random_seed=specs["project_config"].get("random_seed", 99),
        )
        replace_tree(tmpdir / "reselected_turns_reliability", expected_dir, force=force)

    return expected_dir


def _write_expected_turn_analysis(
    project_dir: Path,
    specs: dict[str, dict[str, Any]],
    *,
    force: bool,
) -> Path:
    expected_dir = (
        project_dir / "expected_outputs" / TURNS_MODULE_DIR / TURNS_OUTPUT_DIRS["analyze"]
    )
    primary_path = _turns_input_dir(project_dir, specs) / "conversation_turns_template.xlsx"

    with scratch_dir(project_dir) as tmpdir:
        input_dir = tmpdir / "input" / "conversation_turns"
        input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(primary_path, input_dir / primary_path.name)
        output_dir = tmpdir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        analyze_digital_convo_turns(input_dir=input_dir, output_dir=output_dir)
        replace_tree(output_dir, expected_dir, force=force)

    return expected_dir


EXAMPLE_CAPABILITIES: dict[str, ExampleCapability] = {
    "synthetic_chat_files": ExampleCapability(
        name="synthetic_chat_files",
        materialize=_materialize_chat_input_capability,
    ),
    "synthetic_reliability_chat_files": ExampleCapability(
        name="synthetic_reliability_chat_files",
        materialize=_materialize_reliability_chat_input_capability,
    ),
    "transcript_table_workbook": ExampleCapability(
        name="transcript_table_workbook",
        materialize=_materialize_transcript_table_capability,
    ),
    "transcription_reliability_selection": ExampleCapability(
        name="transcription_reliability_selection",
        materialize=_materialize_transcription_selection_capability,
    ),
    "cu_coding_workbooks": ExampleCapability(
        name="cu_coding_workbooks",
        materialize=_materialize_cu_coding_capability,
    ),
    "cu_coding_analysis": ExampleCapability(
        name="cu_coding_analysis",
        materialize=_materialize_cu_analysis_capability,
    ),
    "word_count_workbooks": ExampleCapability(
        name="word_count_workbooks",
        materialize=_materialize_word_count_capability,
    ),
    "word_count_analysis": ExampleCapability(
        name="word_count_analysis",
        materialize=_materialize_word_count_analysis_capability,
    ),
    "sample_subset_input_workbook": ExampleCapability(
        name="sample_subset_input_workbook",
        materialize=_materialize_sample_subset_input_capability,
    ),
    "speaking_time_workbook": ExampleCapability(
        name="speaking_time_workbook",
        materialize=_materialize_speaking_time_capability,
    ),
}


EXAMPLE_COMMAND_PLANS: dict[str, ExampleCommandPlan] = {
    # Transcripts
    "transcripts tabularize": ExampleCommandPlan(
        command="transcripts tabularize",
        required_capabilities=("synthetic_chat_files",),
        build_output=_build_transcripts_tabularize_example_output,
        output_capabilities=("transcript_table_workbook",),
    ),
    "transcripts chats": ExampleCommandPlan(
        command="transcripts chats",
        required_capabilities=("transcript_table_workbook",),
        build_output=_build_transcripts_chats_example_output,
    ),
    "transcripts select": ExampleCommandPlan(
        command="transcripts select",
        required_capabilities=("synthetic_chat_files",),
        build_output=_build_transcripts_select_example_output,
    ),
    "transcripts reselect": ExampleCommandPlan(
        command="transcripts reselect",
        required_capabilities=("transcription_reliability_selection",),
        build_output=_build_transcripts_reselect_example_output,
    ),
    "transcripts evaluate": ExampleCommandPlan(
        command="transcripts evaluate",
        required_capabilities=("synthetic_reliability_chat_files",),
        build_output=_build_transcripts_evaluate_example_output,
    ),

    # Generic coding templates
    "templates utterances": ExampleCommandPlan(
        command="templates utterances",
        required_capabilities=("transcript_table_workbook",),
        build_output=_build_templates_utterances_example_output,
    ),
    "templates samples": ExampleCommandPlan(
        command="templates samples",
        required_capabilities=("transcript_table_workbook",),
        build_output=_build_templates_samples_example_output,
    ),
    "templates times": ExampleCommandPlan(
        command="templates times",
        required_capabilities=("transcript_table_workbook",),
        build_output=_build_templates_times_example_output,
    ),
    "templates subset": ExampleCommandPlan(
        command="templates subset",
        required_capabilities=("sample_subset_input_workbook",),
        build_output=_build_templates_subset_example_output,
    ),

    # Complete Utterance coding
    "cus files": ExampleCommandPlan(
        command="cus files",
        required_capabilities=("transcript_table_workbook",),
        build_output=_build_cu_files_example_output,
    ),
    "cus reselect": ExampleCommandPlan(
        command="cus reselect",
        required_capabilities=("cu_coding_workbooks",),
        build_output=_build_cu_reselect_example_output,
    ),
    "cus evaluate": ExampleCommandPlan(
        command="cus evaluate",
        required_capabilities=("cu_coding_workbooks",),
        build_output=_build_cu_evaluate_example_output,
    ),
    "cus analyze": ExampleCommandPlan(
        command="cus analyze",
        required_capabilities=("cu_coding_workbooks",),
        build_output=_build_cu_analyze_example_output,
    ),
    "cus rates": ExampleCommandPlan(
        command="cus rates",
        required_capabilities=("cu_coding_analysis", "speaking_time_workbook"),
        build_output=_build_cu_rates_example_output,
    ),

    # Manual word counting
    "words files": ExampleCommandPlan(
        command="words files",
        required_capabilities=("transcript_table_workbook",),
        build_output=_build_word_files_example_output,
    ),
    "words reselect": ExampleCommandPlan(
        command="words reselect",
        required_capabilities=("word_count_workbooks",),
        build_output=_build_word_reselect_example_output,
    ),
    "words evaluate": ExampleCommandPlan(
        command="words evaluate",
        required_capabilities=("word_count_workbooks",),
        build_output=_build_word_evaluate_example_output,
    ),
    "words analyze": ExampleCommandPlan(
        command="words analyze",
        required_capabilities=("word_count_workbooks",),
        build_output=_build_word_analyze_example_output,
    ),
    "words rates": ExampleCommandPlan(
        command="words rates",
        required_capabilities=("word_count_analysis", "speaking_time_workbook"),
        build_output=_build_word_rates_example_output,
    ),
}


def _generate_full_example_files(destination: str | Path, *, force: bool) -> Path:
    project_dir = Path(destination).expanduser().resolve()
    _ensure_writable_package(project_dir, force=force)

    specs = _read_specs()
    project_dir.mkdir(parents=True, exist_ok=True)
    _materialize_inputs(project_dir, specs, force=force)
    _cleanup_obsolete_expected_dirs(project_dir, force=force)
    transcript_table = _write_expected_transcript_table(project_dir, specs, force=force)
    _write_provided_transcript_table(project_dir, specs, transcript_table, force=force)
    _write_expected_selection(project_dir, specs, force=force)
    _write_expected_evaluation(project_dir, specs, force=force)
    _write_expected_reselection(project_dir, specs, force=force)
    _write_expected_utterance_templates(project_dir, specs, force=force)
    _write_expected_sample_templates(project_dir, specs, force=force)
    _write_expected_time_templates(project_dir, specs, force=force)
    _write_expected_sample_subset(project_dir, specs, force=force, mode="subset")
    _write_expected_sample_subset(project_dir, specs, force=force, mode="resubset")
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
    _write_expected_powers_files(project_dir, specs, force=force)
    _write_expected_powers_evaluation(project_dir, specs, force=force)
    _write_expected_powers_reselection(project_dir, specs, force=force)
    _write_expected_powers_analysis(project_dir, specs, force=force)
    _write_expected_powers_rates(project_dir, specs, force=force)
    _write_expected_blinding_encode(project_dir, specs, force=force)
    _write_expected_blinding_decode(project_dir, specs, force=force)
    _write_expected_vocab_file(project_dir, specs, force=force)
    _write_expected_vocab_check(project_dir, specs, force=force)
    _write_expected_vocab_analysis(project_dir, specs, force=force)
    _write_expected_vocab_rates(project_dir, specs, force=force)
    _write_expected_turn_files(project_dir, specs, force=force)
    _write_expected_turn_evaluation(project_dir, specs, force=force)
    _write_expected_turn_reselection(project_dir, specs, force=force)
    _write_expected_turn_analysis(project_dir, specs, force=force)
    return project_dir


def _generate_command_example_files(
    destination: str | Path,
    *,
    commands: Iterable[str] | str,
    force: bool,
) -> Path:
    normalized = _normalize_example_commands(commands)
    package_dir = (
        Path(destination).expanduser().resolve()
        / example_package_name(normalized)
    )
    _ensure_writable_package(package_dir, force=force)

    specs = _read_specs()
    package_dir.mkdir(parents=True, exist_ok=True)
    ctx = ExampleBuildContext(package_dir=package_dir, specs=specs, force=force)
    _write_command_example_readme(ctx, normalized)
    _write_command_example_config(ctx)
    _materialize_required_capabilities(ctx, normalized)

    for command in normalized:
        EXAMPLE_COMMAND_PLANS[command].build_output(ctx)

    _write_command_example_logs(ctx, normalized)
    return package_dir


def generate_example_files(
    destination: str | Path,
    *,
    force: bool = False,
    commands: Iterable[str] | str | None = None,
) -> Path:
    """
    Materialize the synthetic DIAAD example project.

    Parameters
    ----------
    destination
        For full-dataset examples, directory to create or update. For
        command-specific examples, base directory under which
        ``example_files_<slug>`` is created.
    force
        Overwrite existing files when True.
    commands
        Optional canonical DIAAD command or commands for command-specific
        example files. When omitted, the full synthetic dataset is generated.
    """
    if commands is None:
        return _generate_full_example_files(destination, force=force)
    return _generate_command_example_files(
        destination,
        commands=commands,
        force=force,
    )
