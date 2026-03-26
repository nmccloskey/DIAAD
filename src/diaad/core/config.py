from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import yaml

from diaad.core.logger import logger


# ------------------------------------------------------------------
# Section dataclasses
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectConfig:
    """Normalized project-level configuration."""

    input_dir: str = "diaad_data/input"
    output_dir: str = "diaad_data/output"

    random_seed: int = 99
    reliability_fraction: float = 0.2
    shuffle_samples: bool = True

    strip_clan: bool = True
    prefer_correction: bool = True
    lowercase: bool = True

    reliability_tag: str = "_reliability"
    reliability_dirname: str = "reliability"

    exclude_participants: list[str] | None = None
    num_coders: int = 0
    narrative_field: str = ""

    cu_paradigms: list[str] | None = None
    cu_samples_file: Path | str = "cu_coding_by_sample_long.xlsx"
    cu_utts_file: Path | str = "cu_coding_by_utterance.xlsx"

    automate_powers: bool = True
    just_c2_powers: bool = False

    speaking_time_file: Path | str = "speaking_times.xlsx"
    speaking_time_field: str = "speaking_time"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exclude_participants",
            list(self.exclude_participants or []),
        )
        object.__setattr__(self, "cu_paradigms", list(self.cu_paradigms or []))


TierSpec: TypeAlias = str | list[str]

@dataclass(frozen=True)
class TiersConfig:
    """Normalized tier-definition configuration."""
    tiers: dict[str, TierSpec]


@dataclass(frozen=True)
class BlindingConfig:
    """Normalized blinding configuration."""

    blind_files: bool = True
    blind_analysis: bool = True
    metadata_source: str = "transcript_tables"

    coding_blind_cols: list[str] | None = None
    analysis_blind_cols: list[str] | None = None
    id_cols: list[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coding_blind_cols",
            list(self.coding_blind_cols or ["sample_id"]),
        )
        object.__setattr__(
            self,
            "analysis_blind_cols",
            list(self.analysis_blind_cols or ["sample_id"]),
        )
        object.__setattr__(
            self,
            "id_cols",
            list(self.id_cols or ["sample_id", "utterance_id"]),
        )

    @property
    def blinded_suffix(self) -> str:
        """Suffix used for blinded output columns."""
        return "_blinded"

    def get_blind_cols(self, mode: str) -> list[str]:
        """
        Return columns configured for a blinding mode.

        Parameters
        ----------
        mode : str
            Either "coding" or "analysis".

        Returns
        -------
        list[str]
            Columns configured for blinding in that mode.
        """
        if mode == "coding":
            return self.coding_blind_cols
        if mode == "analysis":
            return self.analysis_blind_cols
        raise ValueError(f"Unknown blinding mode: {mode}")

    def should_blind(self, mode: str) -> bool:
        """
        Return whether blinding is enabled for a given mode.

        Parameters
        ----------
        mode : str
            Either "coding" or "analysis".

        Returns
        -------
        bool
            True if blinding is enabled for that mode.
        """
        if mode == "coding":
            return self.blind_files
        if mode == "analysis":
            return self.blind_analysis
        raise ValueError(f"Unknown blinding mode: {mode}")


@dataclass(frozen=True)
class ValidationConfig:
    """Normalized POWERS validation configuration."""

    stratify_by: list[str]
    num_strata: int
    selection_table: str
    stratum_numbers: list[int]


# ------------------------------------------------------------------
# Config manager
# ------------------------------------------------------------------

class ConfigManager:
    """
    Read, validate, normalize, and expose DIAAD configuration.

    Expected config directory contents:
        project.yaml
        tiers.yaml
        blinding.yaml
        validation.yaml
    """

    REQUIRED_FILES = (
        "project.yaml",
        "tiers.yaml",
        "blinding.yaml",
        "validation.yaml",
    )

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir).expanduser().resolve()
        self._validate_config_dir()

        self._raw_project = self._read_yaml("project.yaml")
        self._raw_tiers = self._read_yaml("tiers.yaml")
        self._raw_blinding = self._read_yaml("blinding.yaml")
        self._raw_validation = self._read_yaml("validation.yaml")

        self.project = self._parse_project(self._raw_project)
        self.tiers_section = self._parse_tiers(self._raw_tiers)
        self.blinding = self._parse_blinding(self._raw_blinding)
        self.validation = self._parse_validation(self._raw_validation)

        logger.info("Loaded configuration from %s", self.config_dir)

    # ------------------------------------------------------------------
    # Public convenience properties
    # ------------------------------------------------------------------

    @property
    def input_dir(self) -> str:
        return self.project.input_dir

    @property
    def output_dir(self) -> str:
        return self.project.output_dir

    @property
    def random_seed(self) -> int:
        return self.project.random_seed

    @property
    def reliability_fraction(self) -> float:
        return self.project.reliability_fraction

    @property
    def shuffle_samples(self) -> bool:
        return self.project.shuffle_samples

    @property
    def strip_clan(self) -> bool:
        return self.project.strip_clan

    @property
    def prefer_correction(self) -> bool:
        return self.project.prefer_correction

    @property
    def lowercase(self) -> bool:
        return self.project.lowercase

    @property
    def exclude_participants(self) -> list[str]:
        return self.project.exclude_participants

    @property
    def num_coders(self) -> list[Any]:
        return self.project.num_coders

    @property
    def narrative_field(self) -> str:
        return self.project.narrative_field

    @property
    def cu_paradigms(self) -> list[str]:
        return self.project.cu_paradigms

    @property
    def cu_utts_file(self) -> str:
        return self.project.cu_utts_file

    @property
    def cu_samples_file(self) -> str:
        return self.project.cu_samples_file

    @property
    def automate_powers(self) -> bool:
        return self.project.automate_powers

    @property
    def just_c2_powers(self) -> bool:
        return self.project.just_c2_powers

    @property
    def speaking_time_file(self) -> str:
        return self.project.speaking_time_file

    @property
    def speaking_time_field(self) -> str:
        return self.project.speaking_time_field

    @property
    def tiers_config(self) -> dict[str, Any]:
        """
        Return the normalized tier configuration in a shape compatible
        with TierManager.
        """
        return {
            "tiers": self.tiers_section.tiers,
        }

    @property
    def id_cols(self) -> list[str]:
        return self.blinding.id_cols
    
    @property
    def blind_files(self) -> bool:
        return self.blinding.blind_files
    
    @property
    def blinded_suffix(self) -> str:
        return self.blinding.blinded_suffix

    @property
    def blind_analysis(self) -> bool:
        return self.blinding.blind_analysis
    
    @property
    def coding_blind_cols(self) -> list[str]:
        return self.blinding.get_blind_cols(mode="coding")

    @property
    def analysis_blind_cols(self) -> list[str]:
        return self.blinding.get_blind_cols(mode="analysis")

    @property
    def stratify_by(self) -> list[str]:
        return self.validation.stratify_by

    @property
    def num_strata(self) -> int:
        return self.validation.num_strata

    @property
    def selection_table(self) -> str:
        return self.validation.selection_table

    @property
    def stratum_numbers(self) -> list[int]:
        return self.validation.stratum_numbers

    @property
    def metadata_source(self) -> str:
        return self.blinding.metadata_source
    
    @property
    def metadata_path(self) -> Path:
        return self.input_dir / self.metadata_source

    # ------------------------------------------------------------------
    # Public serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return normalized configuration as a nested dictionary."""
        return {
            "project": {
                "input_dir": self.project.input_dir,
                "output_dir": self.project.output_dir,
                "random_seed": self.project.random_seed,
                "reliability_fraction": self.project.reliability_fraction,
                "shuffle_samples": self.project.shuffle_samples,
                "strip_clan": self.project.strip_clan,
                "prefer_correction": self.project.prefer_correction,
                "lowercase": self.project.lowercase,
                "reliability_tag": self.project.reliability_tag,
                "reliability_dirname": self.project.reliability_dirname,
                "exclude_participants": self.project.exclude_participants,
                "num_coders": self.project.num_coders,
                "narrative_field": self.project.narrative_field,
                "cu_paradigms": self.project.cu_paradigms,
                "cu_samples_file": self.project.cu_samples_file,
                "cu_utts_file": self.project.cu_utts_file,
                "automate_powers": self.project.automate_powers,
                "just_c2_powers": self.project.just_c2_powers,
                "speaking_time_file": self.project.speaking_time_file,
                "speaking_time_field": self.project.speaking_time_field,
            },
            "tiers": self.tiers_section.tiers,
            "blinding": {
                "default_strategy": self.blinding.default_strategy,
                "strategies": self.blinding.strategies,
                "default_id_cols": self.blinding.default_id_cols,
                "code_prefixes": self.blinding.code_prefixes,
            },
            "validation": {
                "stratify_by": self.validation.stratify_by,
                "num_strata": self.validation.num_strata,
                "selection_table": self.validation.selection_table,
                "stratum_numbers": self.validation.stratum_numbers,
            },
        }

    def as_termination_dict(self) -> dict[str, Any]:
        """
        Return a compact config payload suitable for logging and
        run-finalization metadata.
        """
        return {
            "config_dir": str(self.config_dir),
            "project": {
                "input_dir": self.input_dir,
                "output_dir": self.output_dir,
                "random_seed": self.random_seed,
                "reliability_fraction": self.reliability_fraction,
                "shuffle_samples": self.shuffle_samples,
                "strip_clan": self.strip_clan,
                "prefer_correction": self.prefer_correction,
                "lowercase": self.lowercase,
                "exclude_participants": self.exclude_participants,
                "num_coders": self.num_coders,
                "narrative_field": self.narrative_field,
                "cu_paradigms": self.cu_paradigms,
                "cu_samples_file": self.cu_samples_file,
                "cu_utts_file": self.cu_utts_file,
                "automate_powers": self.automate_powers,
                "just_c2_powers": self.just_c2_powers,
                "speaking_time_file": self.speaking_time_file,
                "speaking_time_field": self.speaking_time_field,
            },
            "tiers": self.tiers_config,
            "blinding": {
                "blind_files": self.blind_files,
                "blind_analysis": self.blind_analysis,
                "metadata_source": self.metadata_source,
                "coding_blind_cols": self.coding_blind_cols,
                "analysis_blind_cols": self.analysis_blind_cols,
                "id_cols": self.id_cols,
                "blinded_suffix": self.blinded_suffix,
            },
            "validation": {
                "stratify_by": self.stratify_by,
                "num_strata": self.num_strata,
                "selection_table": self.selection_table,
                "stratum_numbers": self.stratum_numbers,
            },
        }

    # ------------------------------------------------------------------
    # Internal file loading
    # ------------------------------------------------------------------

    def _validate_config_dir(self) -> None:
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {self.config_dir}")
        if not self.config_dir.is_dir():
            raise NotADirectoryError(f"Config path is not a directory: {self.config_dir}")

        missing = [name for name in self.REQUIRED_FILES if not (self.config_dir / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required config file(s) in {self.config_dir}: {', '.join(missing)}"
            )

    def _read_yaml(self, filename: str) -> dict[str, Any]:
        path = self.config_dir / filename
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise TypeError(f"{filename} must parse to a dictionary-like object.")
        return data

    # ------------------------------------------------------------------
    # Project parsing
    # ------------------------------------------------------------------

    def _parse_project(self, data: dict[str, Any]) -> ProjectConfig:
        project = ProjectConfig(
            input_dir=self._as_str(data.get("input_dir"), default="diaad_data/input"),
            output_dir=self._as_str(data.get("output_dir"), default="diaad_data/output"),
            random_seed=self._as_int(data.get("random_seed"), default=99),
            reliability_fraction=self._as_float(
                data.get("reliability_fraction"),
                default=0.2,
            ),
            shuffle_samples=self._as_bool(data.get("shuffle_samples"), default=True),
            strip_clan=self._as_bool(data.get("strip_clan"), default=True),
            prefer_correction=self._as_bool(
                data.get("prefer_correction"),
                default=True,
            ),
            reliability_tag=self._as_str(data.get("reliability_tag"), default="_reliability"),
            reliability_dirname=self._as_str(data.get("reliability_dirname"), default="reliability"),
            lowercase=self._as_bool(data.get("lowercase"), default=True),
            exclude_participants=self._as_str_list(
                data.get("exclude_participants"),
                default=[],
            ),
            num_coders=self._as_int(data.get("num_coders"), default=0),
            narrative_field=self._as_str(data.get("narrative_field"), default=""),
            cu_paradigms=self._as_str_list(data.get("cu_paradigms"), default=[]),
            cu_samples_file=self._as_str(data.get("cu_samples_file"), default="cu_coding_by_sample_long.xlsx"),
            cu_utts_file=self._as_str(data.get("cu_utts_file"), default="cu_coding_by_utterance.xlsx"),
            automate_powers=self._as_bool(data.get("automate_powers"), default=True),
            just_c2_powers=self._as_bool(data.get("just_c2_powers"), default=False),
            speaking_time_file=self._as_str(
                data.get("speaking_time_file"),
                default="speaking_times.xlsx",
            ),
            speaking_time_field=self._as_str(
                data.get("speaking_time_field"),
                default="speaking_time",
            ),
        )

        if not 0 < project.reliability_fraction <= 1:
            raise ValueError(
                f"reliability_fraction must be > 0 and <= 1; got {project.reliability_fraction}"
            )

        return project

    # ------------------------------------------------------------------
    # Tiers parsing
    # ------------------------------------------------------------------

    def _parse_tiers(self, data: dict[str, Any]) -> TiersConfig:
        tiers = data.get("tiers", {})

        if not isinstance(tiers, dict):
            raise TypeError("tiers.yaml: 'tiers' must be a dictionary.")

        normalized_tiers: dict[str, Any] = {}

        for tier_name, tier_spec in tiers.items():
            tier_name = str(tier_name)

            if isinstance(tier_spec, str):
                if not tier_spec.strip():
                    raise ValueError(f"Tier '{tier_name}' regex string must be non-empty.")
                normalized_tiers[tier_name] = tier_spec

            elif isinstance(tier_spec, list):
                if not all(isinstance(v, str) for v in tier_spec):
                    raise TypeError(
                        f"Tier '{tier_name}' values must be a list of strings."
                    )
                normalized_tiers[tier_name] = list(tier_spec)

            else:
                raise TypeError(
                    f"Tier '{tier_name}' must be either a regex string or a list[str]."
                )

        return TiersConfig(tiers=normalized_tiers)

    # ------------------------------------------------------------------
    # Blinding parsing
    # ------------------------------------------------------------------

    def _parse_blinding(self, data: dict[str, Any]) -> BlindingConfig:
        """Parse blinding.yaml into a normalized BlindingConfig."""
        blind_files = self._as_bool(
            data.get("blind_files"),
            default=True,
        )
        blind_analysis = self._as_bool(
            data.get("blind_analysis"),
            default=True,
        )
        metadata_source = self._as_str(
            data.get("metadata_source"),
            default="transcript_tables",
        )

        coding_blind_cols = self._as_str_list(
            data.get("coding_blind_cols"),
            default=None,
        )
        analysis_blind_cols = self._as_str_list(
            data.get("analysis_blind_cols"),
            default=None,
        )
        id_cols = self._as_str_list(
            data.get("id_cols"),
            default=None,
        )

        return BlindingConfig(
            blind_files=blind_files,
            blind_analysis=blind_analysis,
            metadata_source=metadata_source,
            coding_blind_cols=coding_blind_cols,
            analysis_blind_cols=analysis_blind_cols,
            id_cols=id_cols,
        )

    # ------------------------------------------------------------------
    # Validation parsing
    # ------------------------------------------------------------------

    def _parse_validation(self, data: dict[str, Any]) -> ValidationConfig:
        validation = ValidationConfig(
            stratify_by=self._as_str_list(data.get("stratify_by"), default=[]),
            num_strata=self._as_int(data.get("num_strata"), default=0),
            selection_table=self._as_str(data.get("selection_table"), default=""),
            stratum_numbers=self._as_int_list(
                data.get("stratum_numbers"),
                default=[],
            ),
        )

        if validation.num_strata < 0:
            raise ValueError(f"num_strata must be >= 0; got {validation.num_strata}")

        return validation

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _as_bool(value: Any, default: bool | None = None) -> bool:
        if value is None:
            if default is None:
                raise TypeError("Expected bool-compatible value, got None.")
            return default

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "y", "1"}:
                return True
            if lowered in {"false", "no", "n", "0"}:
                return False

        if isinstance(value, int) and value in {0, 1}:
            return bool(value)

        raise TypeError(f"Could not interpret {value!r} as bool.")

    @staticmethod
    def _as_int(value: Any, default: int | None = None) -> int:
        if value is None:
            if default is None:
                raise TypeError("Expected int-compatible value, got None.")
            return default

        if isinstance(value, bool):
            raise TypeError(f"Boolean value {value!r} is not accepted as int.")

        if isinstance(value, int):
            return value

        if isinstance(value, float) and value.is_integer():
            return int(value)

        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
                return int(stripped)

        raise TypeError(f"Could not interpret {value!r} as int.")

    @staticmethod
    def _as_float(value: Any, default: float | None = None) -> float:
        if value is None:
            if default is None:
                raise TypeError("Expected float-compatible value, got None.")
            return default

        if isinstance(value, bool):
            raise TypeError(f"Boolean value {value!r} is not accepted as float.")

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError as e:
                raise TypeError(f"Could not interpret {value!r} as float.") from e

        raise TypeError(f"Could not interpret {value!r} as float.")

    @staticmethod
    def _as_str(value: Any, default: str | None = None) -> str:
        if value is None:
            if default is None:
                raise TypeError("Expected string-compatible value, got None.")
            return default

        if isinstance(value, str):
            return value

        return str(value)

    @staticmethod
    def _as_list(value: Any, default: list[Any] | None = None) -> list[Any]:
        if value is None:
            return list(default or [])
        if isinstance(value, list):
            return list(value)
        raise TypeError(f"Expected list-compatible value, got {type(value).__name__}.")

    def _as_str_list(
        self,
        value: Any,
        default: list[str] | None = None,
    ) -> list[str]:
        items = self._as_list(value, default=default)
        return [self._as_str(item) for item in items]

    def _as_int_list(
        self,
        value: Any,
        default: list[int] | None = None,
    ) -> list[int]:
        items = self._as_list(value, default=default)
        return [self._as_int(item) for item in items]
