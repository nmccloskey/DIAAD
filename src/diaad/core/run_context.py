from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import random
import numpy as np

from diaad import __version__
from psair.metadata.metadata_fields import MetadataManager
from psair.core.logger import logger

from diaad.core.config import ConfigManager
from diaad.metadata.discovery import find_transcript_table


@dataclass
class RunContext:
    """
    Execution context for a single DIAAD run.

    RunContext serves as the boundary object between user configuration
    and executable program modules. It owns normalized configuration,
    resolved runtime paths, metadata field state, and optional cached inputs such
    as loaded CHAT files.

    It is intended to be initialized once in main.py and then passed to
    dispatch/wrapper layers.
    """

    config_dir: str | Path | None
    project_root: str | Path
    start_time: datetime
    config_overrides: dict[str, Any] | None = None
    create_output_dir: bool = True

    # ------------------------------------------------------------------
    # Core managers / config-derived state
    # ------------------------------------------------------------------
    config: ConfigManager = field(init=False)
    metadata_manager: MetadataManager = field(init=False)
    metadata_fields: dict[str, Any] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------
    # Resolved runtime paths
    # ------------------------------------------------------------------
    input_dir: Path = field(init=False)
    base_output_dir: Path = field(init=False)
    out_dir: Path = field(init=False)

    # ------------------------------------------------------------------
    # Cached optional runtime state
    # ------------------------------------------------------------------
    chats: Any = field(default=None, init=False)
    commands: list[str] = field(default_factory=list, init=False)
    start_snapshot: dict[str, Any] | None = field(default=None, init=False)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).expanduser().resolve()
        self.config_dir = self._resolve_config_source(self.config_dir)
        self.config = ConfigManager(self.config_dir, config_overrides=self.config_overrides)
        self._resolve_directories()
        self._seed_rngs()
        self._build_metadata_field_state()

    # ------------------------------------------------------------------
    # Basic metadata / convenience properties
    # ------------------------------------------------------------------
    @property
    def version(self) -> str:
        """Return the installed DIAAD version."""
        return __version__

    @property
    def timestamp(self) -> str:
        """Return the run timestamp string used in output folder naming."""
        return self.start_time.strftime("%y%m%d_%H%M")

    # ------------------------------------------------------------------
    # Frequently used normalized config values
    # ------------------------------------------------------------------
    @property
    def random_seed(self) -> int:
        return self.config.random_seed

    @property
    def reliability_fraction(self) -> float:
        return self.config.reliability_fraction

    @property
    def shuffle_samples(self) -> bool:
        return self.config.shuffle_samples

    @property
    def num_coders(self) -> int:
        return self.config.num_coders

    @property
    def num_bins(self) -> int:
        return self.config.num_bins

    @property
    def cu_paradigms(self) -> list[str]:
        return self.config.cu_paradigms

    @property
    def stimulus_field(self) -> str:
        return self.config.stimulus_field

    @property
    def exclude_speakers(self) -> list[str]:
        return self.config.exclude_speakers

    @property
    def strip_clan(self) -> bool:
        return self.config.strip_clan

    @property
    def prefer_correction(self) -> bool:
        return self.config.prefer_correction

    @property
    def lowercase(self) -> bool:
        return self.config.lowercase

    @property
    def automate_powers(self) -> bool:
        return self.config.automate_powers

    @property
    def auto_tabularize(self) -> bool:
        return self.config.auto_tabularize

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _resolve_project_path(self, path: str | Path) -> Path:
        """
        Resolve a project-defined path relative to the DIAAD project root.

        Absolute paths are preserved. Relative paths are interpreted
        relative to the project root.
        """
        p = Path(path).expanduser()
        if p.is_absolute():
            return p.resolve()
        return (self.project_root / p).resolve()

    def _resolve_config_source(self, config_source: str | Path | None) -> Path | None:
        """
        Resolve the selected config source.

        When omitted, DIAAD uses a conventional project-local config directory
        if present, otherwise ConfigManager falls back to packaged defaults.
        Explicit sources are resolved here and validated by ConfigManager.
        """
        if config_source is not None:
            path = Path(config_source).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            resolved = path.resolve()
            return resolved

        default_config_dir = self.project_root / "config"
        if default_config_dir.exists():
            return default_config_dir.resolve()
        return None

    def _resolve_directories(self) -> None:
        """
        Resolve configured input/output directories and create the
        timestamped run output directory.
        """
        self.input_dir = self._resolve_project_path(self.config.input_dir)
        self.base_output_dir = self._resolve_project_path(self.config.output_dir)

        self.out_dir = (self.base_output_dir / f"diaad_{self.timestamp}").resolve()
        if self.create_output_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def _seed_rngs(self) -> None:
        """Seed Python and NumPy random generators from config."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        logger.info("Random seed set to %s", self.random_seed)

    def _build_metadata_field_state(self) -> None:
        """
        Build MetadataManager and metadata fields from configuration.
        """
        metadata_config = {
            **self.config.metadata_fields_config,
            "input_dir": self.input_dir,
        }
        self.metadata_manager = MetadataManager(metadata_config)
        self.metadata_fields = self.metadata_manager.metadata_fields

        logger.info(
            f"Successfully parsed {len(self.metadata_fields)} metadata fields for DIAAD."
        )

    # ------------------------------------------------------------------
    # Mutable runtime state
    # ------------------------------------------------------------------
    def set_commands(self, commands: Iterable[str]) -> None:
        """Store parsed command strings for this run."""
        self.commands = list(commands)

    # ------------------------------------------------------------------
    # CHAT / transcript preparation
    # ------------------------------------------------------------------
    def load_chats(self, *, force: bool = False) -> Any:
        """
        Load CHAT files if not already loaded, unless force=True.
        """
        if self.chats is not None and not force:
            return self.chats

        from diaad.transcripts.cha_files import read_cha_files

        self.chats = read_cha_files(
            input_dir=self.input_dir,
            shuffle=self.shuffle_samples,
            exclude_dirnames=[self.config.reliability_dirname],
        )
        return self.chats

    def find_transcript_tables(self) -> list[Path]:
        """
        Return transcript tables present in the input directory or current run
        output directory.
        """
        transcript_table = find_transcript_table(
            directories=[self.input_dir, self.out_dir],
            filename=self.config.transcript_table_filename,
            required=False,
        )
        return [transcript_table] if transcript_table is not None else []

    def transcript_tables_exist(self) -> bool:
        """
        Return True if transcript tables are already present in either the
        input directory or current run output directory.
        """
        return bool(self.find_transcript_tables())

    def ensure_transcript_tables(self) -> None:
        """
        Create transcript tables automatically if required tables are not
        already available.
        """
        transcript_tables = self.find_transcript_tables()
        if transcript_tables:
            logger.info(
                "Transcript tables already available; using %s.",
                ", ".join(str(path) for path in transcript_tables),
            )
            return

        message = (
            "Commands requested for this run require transcript tables, but DIAAD "
            f"did not find any transcript_tables files in {self.input_dir} or "
            f"{self.out_dir}. Provide transcript tables in the input directory, "
            "run 'diaad transcripts tabularize' first, or set "
            "'auto_tabularize: true' in config/project.yaml to let DIAAD create "
            "transcript tables from input .cha transcripts automatically."
        )
        if not self.auto_tabularize:
            logger.error(message)
            raise RuntimeError(message)

        logger.info(
            "No transcript tables detected in %s or %s; auto_tabularize is true, "
            "so DIAAD will create transcript tables from input .cha transcripts "
            "in %s.",
            self.input_dir,
            self.out_dir,
            self.out_dir,
        )

        chats = self.load_chats()

        from diaad.transcripts.transcript_tables import tabularize_transcripts

        written = tabularize_transcripts(
            metadata_fields=self.metadata_fields,
            chats=chats,
            output_dir=self.out_dir,
            shuffle=self.shuffle_samples,
            random_seed=self.random_seed,
            sample_id_field=self.config.sample_id_field,
            utterance_id_field=self.config.utterance_id_field,
            transcript_table_filename=self.config.transcript_table_filename,
        )
        if written:
            logger.info(
                "Auto-generated transcript table(s): %s.",
                ", ".join(str(path) for path in written),
            )

    # ------------------------------------------------------------------
    # Logging / termination helpers
    # ------------------------------------------------------------------
    def termination_kwargs(self) -> dict[str, Any]:
        """
        Return a standardized payload for terminate_logger().
        """
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "config_path": self.config_dir,
            "config": self.config.to_dict(),
            "start_time": self.start_time,
            "program_name": "DIAAD",
            "version": self.version,
        }

    def run_paths(self) -> dict[str, str]:
        """Return standard run paths for provenance and dry-run payloads."""
        return {
            "project_root": str(Path(self.project_root).resolve()),
            "config_source": str(self.config_dir) if self.config_dir is not None else None,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.base_output_dir),
            "run_output_dir": str(self.out_dir),
        }

    # ------------------------------------------------------------------
    # Keyword-argument builders
    # ------------------------------------------------------------------
    def kwargs_io(self) -> dict[str, Any]:
        """
        Return the standard input/output directory payload for modules
        that do not require metadata fields.
        """
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
        }

    def kwargs_encode_blinding(self) -> dict[str, Any]:
        """Return kwargs for standalone blinding encode."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "blinding_config": self.config.blinding,
            "seed": self.random_seed,
        }

    def kwargs_decode_blinding(self) -> dict[str, Any]:
        """Return kwargs for standalone blinding decode."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "blinding_config": self.config.blinding,
        }

    def kwargs_metadata_field_io(self) -> dict[str, Any]:
        """
        Return the standard metadata field + input/output payload for modules
        that operate on metadata-aware transcript tables.
        """
        return {
            "metadata_fields": self.metadata_fields,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
        }

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------
    def kwargs_tabularize_transcripts(self) -> dict[str, Any]:
        """Return kwargs for transcript tabularization."""
        if self.chats is None:
            raise RuntimeError("CHAT files have not been loaded for this run.")
        return {
            "metadata_fields": self.metadata_fields,
            "chats": self.chats,
            "output_dir": self.out_dir,
            "shuffle": self.shuffle_samples,
            "random_seed": self.random_seed,
            "sample_id_field": self.config.sample_id_field,
            "utterance_id_field": self.config.utterance_id_field,
            "transcript_table_filename": self.config.transcript_table_filename,
        }

    def kwargs_detabularize_transcripts(self) -> dict[str, Any]:
        """Return kwargs for transcript de-tabularization."""
        return {
            **self.kwargs_io(),
            "sample_id_field": self.config.sample_id_field,
            "transcript_table_filename": self.config.transcript_table_filename,
        }

    def kwargs_select_transcription_reliability_samples(self) -> dict[str, Any]:
        """Return kwargs for transcription reliability sample selection."""
        if self.chats is None:
            raise RuntimeError("CHAT files have not been loaded for this run.")
        return {
            "metadata_fields": self.metadata_fields,
            "chats": self.chats,
            "frac": self.reliability_fraction,
            "output_dir": self.out_dir,
            "input_dir": self.input_dir,
        }

    def kwargs_reselect_transcription_reliability_samples(self) -> dict[str, Any]:
        """Return kwargs for transcription reliability reselection."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "frac": self.reliability_fraction,
        }

    def kwargs_evaluate_transcription_reliability(self) -> dict[str, Any]:
        """Return kwargs for transcription reliability evaluation."""
        return {
            "metadata_fields": self.metadata_fields,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "exclude_speakers": self.exclude_speakers,
            "strip_clan": self.strip_clan,
            "prefer_correction": self.prefer_correction,
            "lowercase": self.lowercase,
            "reliability_tag": self.config.advanced.reliability_tag,
            "reliability_dirname": self.config.advanced.reliability_dirname,
        }

    # ------------------------------------------------------------------
    # Complete Utterance coding
    # ------------------------------------------------------------------
    def kwargs_make_cu_coding_files(self) -> dict[str, Any]:
        """Return kwargs for CU coding file creation."""
        return {
            "metadata_fields": self.metadata_fields,
            "frac": self.reliability_fraction,
            "num_coders": self.num_coders,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "cu_paradigms": self.cu_paradigms,
            "exclude_speakers": self.exclude_speakers,
            "stimulus_field": self.stimulus_field,
            "blinding_config": self.config.blinding,
            "sample_id_field": self.config.sample_id_field,
        }
    
    def kwargs_evaluate_cu_reliability(self) -> dict[str, Any]:
        """Return kwargs for CU reliability evaluation."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "cu_paradigms": self.cu_paradigms,
            "sample_id_field": self.config.sample_id_field,
            "utterance_id_field": self.config.utterance_id_field,
        }

    def kwargs_reselect_cu_rel(self) -> dict[str, Any]:
        """Return kwargs for CU reliability reselection."""
        return {
            "metadata_fields": self.metadata_fields,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "frac": self.reliability_fraction,
            "random_seed": self.random_seed,
            "sample_id_field": self.config.sample_id_field,
        }

    def kwargs_cu_analysis(self) -> dict[str, Any]:
        """
        Return shared kwargs for CU reliability evaluation and finalized
        CU analysis.
        """
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "cu_paradigms": self.cu_paradigms,
            "blinding_config": self.config.blinding,
            "sample_id_field": self.config.sample_id_field,
            "exclude_speakers": self.exclude_speakers,
        }
    
    def kwargs_cu_rates(self) -> dict[str, Any]:
        """
        Return shared kwargs for CU rate calculation.
        """
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "cu_samples_file": self.config.advanced.cu_samples_filename,
            "speaking_time_file": self.config.advanced.speaking_time_filename,
            "speaking_time_field": self.config.advanced.speaking_time_column,
            "sample_id_field": self.config.sample_id_field,
        }

    # ------------------------------------------------------------------
    # Manual word counting
    # ------------------------------------------------------------------
    def kwargs_make_word_count_files(self) -> dict[str, Any]:
        """Return kwargs for manual word-count file creation."""
        return {
            "frac": self.reliability_fraction,
            "num_coders": self.num_coders,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "exclude_speakers": self.exclude_speakers,
            "blinding_config": self.config.blinding,
            "sample_id_field": self.config.sample_id_field,
            "utterance_id_field": self.config.utterance_id_field,
        }

    def kwargs_reselect_wc_rel(self) -> dict[str, Any]:
        """Return kwargs for word-count reliability reselection."""
        return {
            "metadata_fields": self.metadata_fields,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "frac": self.reliability_fraction,
            "random_seed": self.random_seed,
            "sample_id_field": self.config.sample_id_field,
        }

    def kwargs_evaluate_word_count_reliability(self) -> dict[str, Any]:
        """Return kwargs for word-count reliability evaluation."""
        return {
            **self.kwargs_io(),
            "sample_id_field": self.config.sample_id_field,
            "utterance_id_field": self.config.utterance_id_field,
        }
    
    def kwargs_analyze_word_counts(self) -> dict[str, Any]:
        """Return kwargs for word-count analysis."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "word_count_file": self.config.advanced.word_count_filename,
            "word_count_field": self.config.advanced.word_count_column,
            "blinding_config": self.config.blinding,
            "sample_id_field": self.config.sample_id_field,
            "exclude_speakers": self.exclude_speakers,
        }

    def kwargs_wc_rates(self) -> dict[str, Any]:
        """
        Return shared kwargs for word count rate calculation.
        """
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "wc_samples_file": self.config.advanced.wc_samples_filename,
            "speaking_time_file": self.config.advanced.speaking_time_filename,
            "speaking_time_field": self.config.advanced.speaking_time_column,
            "sample_id_field": self.config.sample_id_field,
        }

    # ------------------------------------------------------------------
    # Target vocabulary coverage
    # ------------------------------------------------------------------
    def kwargs_target_vocab(self) -> dict[str, Any]:
        """Return kwargs for target vocabulary coverage analysis."""
        return {
            "metadata_fields": self.metadata_fields,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "exclude_speakers": self.exclude_speakers,
            "stimulus_field": self.stimulus_field,
            "resource_path": self.config.advanced.target_vocabulary_resource_path,
            "sample_id_field": self.config.sample_id_field,
        }

    def kwargs_target_vocab_check(self) -> dict[str, Any]:
        """Return kwargs for target vocabulary resource validation."""
        return {
            "resource_path": self.config.advanced.target_vocabulary_resource_path,
            "output_dir": self.out_dir,
        }

    def kwargs_target_vocab_file(self) -> dict[str, Any]:
        """Return kwargs for target vocabulary template generation."""
        return self.kwargs_io()

    def kwargs_target_vocab_rates(self) -> dict[str, Any]:
        """Return kwargs for target vocabulary rate calculation."""
        return {
            **self.kwargs_io(),
            "sample_id_field": self.config.sample_id_field,
        }

    # ------------------------------------------------------------------
    # Digital Conversation Turns
    # ------------------------------------------------------------------
    def kwargs_make_digital_convo_turn_files(self) -> dict[str, Any]:
        """Return kwargs for digital conversation turn template generation."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "frac": self.reliability_fraction,
            "num_bins": self.num_bins,
            "num_coders": self.num_coders,
            "blinding_config": self.config.blinding,
            "seed": self.random_seed,
            "sample_id_field": self.config.sample_id_field,
        }

    def kwargs_digital_convo_turns_reliability(self) -> dict[str, Any]:
        """Return kwargs for digital conversation turn reliability evaluation."""
        return {
            **self.kwargs_metadata_field_io(),
            "sample_id_field": self.config.sample_id_field,
        }

    def kwargs_reselect_digital_convo_turns(self) -> dict[str, Any]:
        """Return kwargs for digital conversation turn reliability reselection."""
        return {
            "metadata_fields": self.metadata_fields,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "frac": self.reliability_fraction,
            "random_seed": self.random_seed,
            "sample_id_field": self.config.sample_id_field,
        }

    def kwargs_digital_convo_turns(self) -> dict[str, Any]:
        """Return kwargs for digital conversation turn analysis."""
        return {
            **self.kwargs_io(),
            "sample_id_field": self.config.sample_id_field,
        }

    # ------------------------------------------------------------------
    # Generic coding templates
    # ------------------------------------------------------------------
    def kwargs_make_utterance_templates(self) -> dict[str, Any]:
        """Return kwargs for utterance template generation."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "frac": self.reliability_fraction,
            "num_coders": self.num_coders,
            "stimulus_field": self.stimulus_field,
            "blinding_config": self.config.blinding,
            "seed": self.random_seed,
            "sample_id_field": self.config.sample_id_field,
            "utterance_id_field": self.config.utterance_id_field,
        }

    def kwargs_make_sample_templates(self) -> dict[str, Any]:
        """Return kwargs for sample template generation."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "frac": self.reliability_fraction,
            "num_bins": self.num_bins,
            "num_coders": self.num_coders,
            "stimulus_field": self.stimulus_field,
            "blinding_config": self.config.blinding,
            "seed": self.random_seed,
            "sample_id_field": self.config.sample_id_field,
        }

    def kwargs_make_speaking_time_templates(self) -> dict[str, Any]:
        """Return kwargs for speaking-time template generation."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "sample_id_field": self.config.sample_id_field,
        }

    # ------------------------------------------------------------------
    # POWERS coding workflow
    # ------------------------------------------------------------------
    def kwargs_make_powers_coding_files(self) -> dict[str, Any]:
        """Return kwargs for POWERS coding file creation."""
        return {
            "metadata_fields": self.metadata_fields,
            "frac": self.reliability_fraction,
            "num_coders": self.num_coders,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "exclude_speakers": self.exclude_speakers,
            "automate_powers": self.automate_powers,
            "blinding_config": self.config.blinding,
            "powers_coding_file": self.config.powers_coding_filename,
            "powers_reliability_file": self.config.powers_reliability_filename,
            "spacy_model_name": self.config.spacy_model_name,
            "sample_id_field": self.config.sample_id_field,
            "utterance_id_field": self.config.utterance_id_field,
        }

    def kwargs_analyze_powers_coding(self) -> dict[str, Any]:
        """Return kwargs for POWERS analysis."""
        return {
            **self.kwargs_io(),
            "powers_coding_file": self.config.powers_coding_filename,
            "blinding_config": self.config.blinding,
            "sample_id_field": self.config.sample_id_field,
        }

    def kwargs_powers_rates(self) -> dict[str, Any]:
        """Return shared kwargs for POWERS rate calculation."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "speaking_time_file": self.config.advanced.speaking_time_filename,
            "speaking_time_field": self.config.advanced.speaking_time_column,
            "sample_id_field": self.config.sample_id_field,
        }

    def kwargs_evaluate_powers_reliability(self) -> dict[str, Any]:
        """Return kwargs for POWERS reliability evaluation."""
        return {
            **self.kwargs_io(),
            "powers_coding_file": self.config.powers_coding_filename,
            "powers_reliability_file": self.config.powers_reliability_filename,
            "sample_id_field": self.config.sample_id_field,
            "utterance_id_field": self.config.utterance_id_field,
        }

    def kwargs_reselect_powers_reliability(self) -> dict[str, Any]:
        """Return kwargs for POWERS reliability reselection."""
        return {
            "metadata_fields": self.metadata_fields,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "frac": self.reliability_fraction,
            "random_seed": self.random_seed,
            "automate_powers": self.automate_powers,
            "powers_coding_file": self.config.powers_coding_filename,
            "powers_reliability_file": self.config.powers_reliability_filename,
            "spacy_model_name": self.config.spacy_model_name,
            "sample_id_field": self.config.sample_id_field,
        }
