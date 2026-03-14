from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import random
import numpy as np

from diaad import __version__
from diaad.io.discovery import find_matching_files
from diaad.utils.logger import logger, get_root

from diaad.utils.config import ConfigManager


@dataclass
class RunContext:
    """
    Execution context for a single DIAAD run.

    RunContext serves as the boundary object between user configuration
    and executable program modules. It owns normalized configuration,
    resolved runtime paths, tier state, and optional cached inputs such
    as loaded CHAT files.

    It is intended to be initialized once in main.py and then passed to
    dispatch/wrapper layers.
    """

    config_path: str | Path
    start_time: datetime

    # ------------------------------------------------------------------
    # Core managers / config-derived state
    # ------------------------------------------------------------------
    config: ConfigManager = field(init=False)
    tier_manager: Any = field(default=None, init=False)
    tiers: dict[str, Any] = field(default_factory=dict, init=False)

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

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.config_path = Path(self.config_path).resolve()
        self.config = ConfigManager(self.config_path)
        self._resolve_directories()
        self._seed_rngs()
        self._build_tier_state()

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
    def coders(self) -> list[Any]:
        return self.config.coders

    @property
    def cu_paradigms(self) -> list[str]:
        return self.config.cu_paradigms

    @property
    def narrative_field(self) -> str:
        return self.config.narrative_field

    @property
    def exclude_participants(self) -> list[str]:
        return self.config.exclude_participants

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
    def just_c2_powers(self) -> bool:
        return self.config.just_c2_powers

    @property
    def stratify_by(self) -> list[str]:
        return self.config.stratify_by

    @property
    def num_strata(self) -> int:
        return self.config.num_strata

    @property
    def selection_table(self) -> str:
        return self.config.selection_table

    @property
    def stratum_numbers(self) -> list[int]:
        return self.config.stratum_numbers

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _resolve_directories(self) -> None:
        """
        Resolve configured input/output directories and create the
        timestamped run output directory.
        """
        self.input_dir = cwd_path(self.config.input_dir)
        self.base_output_dir = cwd_path(self.config.output_dir)

        if not self.input_dir.is_relative_to(get_root()):
            logger.warning(
                "Input directory %s is outside the project root.",
                self.input_dir,
            )

        self.out_dir = (self.base_output_dir / f"diaad_output_{self.timestamp}").resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _seed_rngs(self) -> None:
        """Seed Python and NumPy random generators from config."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        logger.info("Random seed set to %s", self.random_seed)

    def _build_tier_state(self) -> None:
        """
        Build TierManager and adapted DIAAD tiers from configuration.
        """
        from diaad.utils.diaad_tier_adapter import adapt_tiers_for_diaad
        from diaad.utils.tiers import TierManager

        self.tier_manager = TierManager(self.config.tiers_config)
        self.tiers = adapt_tiers_for_diaad(self.tier_manager) or {}

        if self.tiers:
            logger.info("Successfully parsed and adapted tiers for DIAAD.")
        else:
            logger.warning("Adapted tiers are empty or malformed.")

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

        from diaad.utils.cha_files import read_cha_files

        self.chats = read_cha_files(
            input_dir=self.input_dir,
            shuffle=self.shuffle_samples,
        )
        return self.chats

    def transcript_tables_exist(self) -> bool:
        """
        Return True if transcript tables are already present in either the
        input directory or current run output directory.
        """
        transcript_tables = find_matching_files(
            directories=[self.input_dir, self.out_dir],
            search_base="transcript_tables",
        )
        return bool(transcript_tables)

    def ensure_transcript_tables(self) -> None:
        """
        Create transcript tables automatically if required tables are not
        already available.
        """
        if self.transcript_tables_exist():
            return

        logger.info(
            "No input transcript tables detected - creating them automatically."
        )

        chats = self.load_chats()

        from diaad.transcripts.transcript_tables import tabularize_transcripts

        tabularize_transcripts(
            tiers=self.tiers,
            chats=chats,
            output_dir=self.out_dir,
            shuffle=self.shuffle_samples,
            random_seed=self.random_seed,
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
            "config_path": self.config_path,
            "config": self.config.as_termination_dict(),
            "start_time": self.start_time,
            "program_name": "DIAAD",
            "version": self.version,
        }

    # ------------------------------------------------------------------
    # Keyword-argument builders
    # ------------------------------------------------------------------
    def kwargs_io(self) -> dict[str, Any]:
        """
        Return the standard input/output directory payload for modules
        that do not require tiers.
        """
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
        }

    def kwargs_tiered_io(self) -> dict[str, Any]:
        """
        Return the standard tiers + input/output payload for modules
        that operate on tier-aware transcript tables.
        """
        return {
            "tiers": self.tiers,
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
            "tiers": self.tiers,
            "chats": self.chats,
            "output_dir": self.out_dir,
            "shuffle": self.shuffle_samples,
            "random_seed": self.random_seed,
        }

    def kwargs_select_transcription_reliability_samples(self) -> dict[str, Any]:
        """Return kwargs for transcription reliability sample selection."""
        if self.chats is None:
            raise RuntimeError("CHAT files have not been loaded for this run.")
        return {
            "tiers": self.tiers,
            "chats": self.chats,
            "frac": self.reliability_fraction,
            "output_dir": self.out_dir,
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
            "tiers": self.tiers,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "exclude_participants": self.exclude_participants,
            "strip_clan": self.strip_clan,
            "prefer_correction": self.prefer_correction,
            "lowercase": self.lowercase,
        }

    # ------------------------------------------------------------------
    # Complete Utterance coding
    # ------------------------------------------------------------------
    def kwargs_make_cu_coding_files(self) -> dict[str, Any]:
        """Return kwargs for CU coding file creation."""
        return {
            "tiers": self.tiers,
            "frac": self.reliability_fraction,
            "coders": self.coders,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "cu_paradigms": self.cu_paradigms,
            "exclude_participants": self.exclude_participants,
            "narrative_field": self.narrative_field,
        }

    def kwargs_reselect_cu_rel(self) -> dict[str, Any]:
        """Return kwargs for CU reliability reselection."""
        return {
            "tiers": self.tiers,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "frac": self.reliability_fraction,
            "random_seed": self.random_seed,
        }

    def kwargs_cu_analysis(self) -> dict[str, Any]:
        """
        Return shared kwargs for CU reliability evaluation and finalized
        CU analysis.
        """
        return {
            "tiers": self.tiers,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "cu_paradigms": self.cu_paradigms,
        }

    # ------------------------------------------------------------------
    # Manual word counting
    # ------------------------------------------------------------------
    def kwargs_make_word_count_files(self) -> dict[str, Any]:
        """Return kwargs for manual word-count file creation."""
        return {
            "tiers": self.tiers,
            "frac": self.reliability_fraction,
            "coders": self.coders,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
        }

    def kwargs_reselect_wc_rel(self) -> dict[str, Any]:
        """Return kwargs for word-count reliability reselection."""
        return {
            "tiers": self.tiers,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "frac": self.reliability_fraction,
            "random_seed": self.random_seed,
        }

    # ------------------------------------------------------------------
    # CoreLex
    # ------------------------------------------------------------------
    def kwargs_corelex(self) -> dict[str, Any]:
        """Return kwargs for CoreLex analysis."""
        return {
            "tiers": self.tiers,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "exclude_participants": self.exclude_participants,
            "narrative_field": self.narrative_field,
        }

    # ------------------------------------------------------------------
    # Digital Conversation Turns
    # ------------------------------------------------------------------
    def kwargs_digital_convo_turns(self) -> dict[str, Any]:
        """Return kwargs for digital conversation turn analysis."""
        return self.kwargs_io()

    # ------------------------------------------------------------------
    # POWERS coding workflow
    # ------------------------------------------------------------------
    def kwargs_make_powers_coding_files(self) -> dict[str, Any]:
        """Return kwargs for POWERS coding file creation."""
        return {
            "tiers": self.tiers,
            "frac": self.reliability_fraction,
            "coders": self.coders,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "exclude_participants": self.exclude_participants,
            "automate_powers": self.automate_powers,
        }

    def kwargs_analyze_powers_coding(
        self,
        *,
        reliability: bool = False,
        just_c2_powers: bool | None = None,
    ) -> dict[str, Any]:
        """
        Return kwargs for POWERS analysis.

        Parameters
        ----------
        reliability:
            Whether the analysis is running in reliability mode.
        just_c2_powers:
            Override for the configured just_c2_powers setting. If None,
            the configured value is used.
        """
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "reliability": reliability,
            "just_c2_powers": (
                self.just_c2_powers if just_c2_powers is None else just_c2_powers
            ),
            "exclude_participants": self.exclude_participants,
        }

    def kwargs_reselect_powers_reliability(self) -> dict[str, Any]:
        """Return kwargs for POWERS reliability reselection."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "frac": self.reliability_fraction,
            "exclude_participants": self.exclude_participants,
            "automate_powers": self.automate_powers,
        }

    # ------------------------------------------------------------------
    # POWERS automation validation
    # ------------------------------------------------------------------
    def kwargs_select_for_validation(self) -> dict[str, Any]:
        """Return base kwargs for POWERS validation sample selection."""
        return {
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "stratify_by": self.stratify_by,
            "num_strata": self.num_strata,
            "random_seed": self.random_seed,
        }

    def kwargs_validate_automation(self) -> dict[str, Any]:
        """Return base kwargs for POWERS automation validation."""
        return {
            "selection_table": self.selection_table,
            "stratum_numbers": self.stratum_numbers,
            "input_dir": self.input_dir,
            "output_dir": self.out_dir,
            "exclude_participants": self.exclude_participants,
        }
