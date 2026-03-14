from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import random
import numpy as np

from diaad import __version__
from diaad.utils.auxiliary import find_files, project_path
from diaad.utils.logger import logger
from diaad.utils.auxiliary import get_root

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
        self.input_dir = project_path(self.config.input_dir)
        self.base_output_dir = project_path(self.config.output_dir)

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
        transcript_tables = find_files(
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
