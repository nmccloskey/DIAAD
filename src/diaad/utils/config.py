from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from diaad.utils.logger import logger


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

    exclude_participants: list[str] | None = None
    coders: list[Any] | None = None
    narrative_field: str = ""

    cu_paradigms: list[str] | None = None

    automate_powers: bool = True
    just_c2_powers: bool = False

    compute_rates: bool = True
    speaking_time_field: str = "speaking_time"
    rate_unit: str = "minute"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exclude_participants",
            list(self.exclude_participants or []),
        )
        object.__setattr__(self, "coders", list(self.coders or []))
        object.__setattr__(self, "cu_paradigms", list(self.cu_paradigms or []))


@dataclass(frozen=True)
class TiersConfig:
    """Normalized tier-definition configuration."""

    tiers: dict[str, dict[str, Any]]
    tier_groups: dict[str, list[str]]


@dataclass(frozen=True)
class BlindingConfig:
    """Normalized blinding configuration."""

    default_strategy: str
    strategies: dict[str, dict[str, Any]]
    default_id_cols: list[str]
    code_prefixes: dict[str, str]


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

        self._validate_cross_section_consistency()

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
    def coders(self) -> list[Any]:
        return self.project.coders

    @property
    def narrative_field(self) -> str:
        return self.project.narrative_field

    @property
    def cu_paradigms(self) -> list[str]:
        return self.project.cu_paradigms

    @property
    def automate_powers(self) -> bool:
        return self.project.automate_powers

    @property
    def just_c2_powers(self) -> bool:
        return self.project.just_c2_powers

    @property
    def compute_rates(self) -> bool:
        return self.project.compute_rates

    @property
    def speaking_time_field(self) -> str:
        return self.project.speaking_time_field

    @property
    def rate_unit(self) -> str:
        return self.project.rate_unit

    @property
    def tiers_config(self) -> dict[str, Any]:
        """
        Return the normalized tier configuration in a shape compatible
        with TierManager.
        """
        return {
            "tiers": self.tiers_section.tiers,
            "tier_groups": self.tiers_section.tier_groups,
        }

    @property
    def default_blinding_strategy(self) -> str:
        return self.blinding.default_strategy

    @property
    def blinding_strategies(self) -> dict[str, dict[str, Any]]:
        return self.blinding.strategies

    @property
    def default_id_cols(self) -> list[str]:
        return self.blinding.default_id_cols

    @property
    def code_prefixes(self) -> dict[str, str]:
        return self.blinding.code_prefixes

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
                "exclude_participants": self.project.exclude_participants,
                "coders": self.project.coders,
                "narrative_field": self.project.narrative_field,
                "cu_paradigms": self.project.cu_paradigms,
                "automate_powers": self.project.automate_powers,
                "just_c2_powers": self.project.just_c2_powers,
                "compute_rates": self.project.compute_rates,
                "speaking_time_field": self.project.speaking_time_field,
                "rate_unit": self.project.rate_unit,
            },
            "tiers": {
                "tiers": self.tiers_section.tiers,
                "tier_groups": self.tiers_section.tier_groups,
            },
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
                "coders": self.coders,
                "narrative_field": self.narrative_field,
                "cu_paradigms": self.cu_paradigms,
                "automate_powers": self.automate_powers,
                "just_c2_powers": self.just_c2_powers,
                "compute_rates": self.compute_rates,
                "speaking_time_field": self.speaking_time_field,
                "rate_unit": self.rate_unit,
            },
            "tiers": self.tiers_config,
            "blinding": {
                "default_strategy": self.default_blinding_strategy,
                "default_id_cols": self.default_id_cols,
                "code_prefixes": self.code_prefixes,
                "strategies": self.blinding_strategies,
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
            lowercase=self._as_bool(data.get("lowercase"), default=True),
            exclude_participants=self._as_str_list(
                data.get("exclude_participants"),
                default=[],
            ),
            coders=self._as_list(data.get("coders"), default=[]),
            narrative_field=self._as_str(data.get("narrative_field"), default=""),
            cu_paradigms=self._as_str_list(data.get("cu_paradigms"), default=[]),
            automate_powers=self._as_bool(data.get("automate_powers"), default=True),
            just_c2_powers=self._as_bool(data.get("just_c2_powers"), default=False),
            compute_rates=self._as_bool(data.get("compute_rates"), default=True),
            speaking_time_field=self._as_str(
                data.get("speaking_time_field"),
                default="speaking_time",
            ),
            rate_unit=self._as_str(data.get("rate_unit"), default="minute"),
        )

        if not 0 < project.reliability_fraction <= 1:
            raise ValueError(
                f"reliability_fraction must be > 0 and <= 1; got {project.reliability_fraction}"
            )

        allowed_rate_units = {"second", "minute", "hour"}
        if project.rate_unit not in allowed_rate_units:
            raise ValueError(
                f"rate_unit must be one of {sorted(allowed_rate_units)}; got {project.rate_unit!r}"
            )

        return project

    # ------------------------------------------------------------------
    # Tiers parsing
    # ------------------------------------------------------------------

    def _parse_tiers(self, data: dict[str, Any]) -> TiersConfig:
        tiers = data.get("tiers", {})
        tier_groups = data.get("tier_groups", {})

        if not isinstance(tiers, dict):
            raise TypeError("tiers.yaml: 'tiers' must be a dictionary.")
        if not isinstance(tier_groups, dict):
            raise TypeError("tiers.yaml: 'tier_groups' must be a dictionary.")

        normalized_tiers: dict[str, dict[str, Any]] = {}
        orders: list[int] = []

        for tier_name, tier_spec in tiers.items():
            if not isinstance(tier_spec, dict):
                raise TypeError(f"Tier '{tier_name}' must map to a dictionary.")

            order = tier_spec.get("order")
            if order is None:
                raise ValueError(f"Tier '{tier_name}' is missing required key 'order'.")
            order = self._as_int(order)
            orders.append(order)

            normalized_spec = dict(tier_spec)
            normalized_spec["order"] = order
            normalized_tiers[str(tier_name)] = normalized_spec

        if len(orders) != len(set(orders)):
            logger.warning("Duplicate tier order values detected in tiers.yaml.")

        normalized_groups: dict[str, list[str]] = {}
        for group_name, members in tier_groups.items():
            normalized_groups[str(group_name)] = self._as_str_list(members, default=[])

        return TiersConfig(
            tiers=normalized_tiers,
            tier_groups=normalized_groups,
        )

    # ------------------------------------------------------------------
    # Blinding parsing
    # ------------------------------------------------------------------

    def _parse_blinding(self, data: dict[str, Any]) -> BlindingConfig:
        default_strategy = self._as_str(
            data.get("default_strategy"),
            default="analysis",
        )
        strategies = data.get("strategies", {})
        default_id_cols = self._as_str_list(
            data.get("default_id_cols"),
            default=["sample_id", "utterance_id"],
        )
        code_prefixes = data.get("code_prefixes", {})

        if not isinstance(strategies, dict):
            raise TypeError("blinding.yaml: 'strategies' must be a dictionary.")
        if not isinstance(code_prefixes, dict):
            raise TypeError("blinding.yaml: 'code_prefixes' must be a dictionary.")

        if default_strategy not in strategies:
            raise ValueError(
                f"default_strategy {default_strategy!r} is not defined under 'strategies'."
            )

        normalized_strategies: dict[str, dict[str, Any]] = {}
        for name, spec in strategies.items():
            if not isinstance(spec, dict):
                raise TypeError(f"Blinding strategy '{name}' must map to a dictionary.")

            normalized = dict(spec)
            normalized["append"] = self._as_bool(spec.get("append"), default=True)
            normalized["suffix"] = self._as_str(spec.get("suffix"), default="_blind")
            normalized["preserve_unmapped"] = self._as_bool(
                spec.get("preserve_unmapped"),
                default=True,
            )
            normalized["drop_recovered_source_cols"] = self._as_bool(
                spec.get("drop_recovered_source_cols"),
                default=False,
            )
            normalized["include_na"] = self._as_bool(
                spec.get("include_na"),
                default=False,
            )
            normalized["metadata_source"] = self._as_str(
                spec.get("metadata_source"),
                default="transcript_tables",
            )
            normalized["width"] = self._as_int(spec.get("width"), default=3)

            normalized_strategies[str(name)] = normalized

        normalized_prefixes = {
            str(k): self._as_str(v)
            for k, v in code_prefixes.items()
        }

        return BlindingConfig(
            default_strategy=default_strategy,
            strategies=normalized_strategies,
            default_id_cols=default_id_cols,
            code_prefixes=normalized_prefixes,
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
    # Cross-section validation
    # ------------------------------------------------------------------

    def _validate_cross_section_consistency(self) -> None:
        tier_names = set(self.tiers_section.tiers.keys())

        for group_name, members in self.tiers_section.tier_groups.items():
            unknown = [m for m in members if m not in tier_names]
            if unknown:
                logger.warning(
                    "Tier group '%s' references unknown tiers: %s",
                    group_name,
                    ", ".join(unknown),
                )

        for field in self.validation.stratify_by:
            if field not in tier_names:
                logger.warning(
                    "Validation stratify field '%s' is not defined in tiers.yaml",
                    field,
                )

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
