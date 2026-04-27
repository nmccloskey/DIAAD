from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import yaml

from psair.core.logger import logger


# ------------------------------------------------------------------
# Section dataclasses
# ------------------------------------------------------------------

MetadataFieldSpec: TypeAlias = str | list[str]


@dataclass(frozen=True)
class ProjectConfig:
    """Normalized user-facing project configuration."""

    input_dir: str = "diaad_data/input"
    output_dir: str = "diaad_data/output"

    random_seed: int = 99
    reliability_fraction: float = 0.2
    shuffle_samples: bool = True

    strip_clan: bool = True
    prefer_correction: bool = True
    lowercase: bool = True
    exclude_participants: list[str] | None = None

    num_bins: int = 4
    num_coders: int = 0
    stimulus_field: str = ""
    automate_powers: bool = True

    metadata_fields: dict[str, MetadataFieldSpec] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exclude_participants",
            list(self.exclude_participants or []),
        )
        object.__setattr__(
            self,
            "metadata_fields",
            dict(self.metadata_fields or {}),
        )


@dataclass(frozen=True)
class AdvancedConfig:
    """Normalized advanced configuration."""

    reliability_tag: str = "_reliability"
    reliability_dirname: str = "reliability"

    cu_paradigms: list[str] | None = None
    cu_samples_file: str = "cu_coding_by_sample_long.xlsx"
    cu_utts_file: str = "cu_coding_by_utterance.xlsx"

    word_count_file: str = "word_counting.xlsx"
    word_count_field: str = "word_count"
    wc_samples_file: str = "word_counting_by_sample.xlsx"

    speaking_time_file: str = "speaking_times.xlsx"
    speaking_time_field: str = "speaking_time"

    target_vocabulary_resource_path: str = ""

    auto_blind: bool = False
    blind_cols: list[str] | None = None
    metadata_source: str = "transcript_tables"
    codebook_filename: str = ""

    # Deprecated aliases retained so older config files and call sites still work.
    coding_blind_cols: list[str] | None = None
    analysis_blind_cols: list[str] | None = None
    id_cols: list[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cu_paradigms", list(self.cu_paradigms or []))
        configured_blind_cols = self.blind_cols
        if configured_blind_cols is None:
            configured_blind_cols = self.coding_blind_cols or self.analysis_blind_cols
        object.__setattr__(
            self,
            "blind_cols",
            list(configured_blind_cols or ["sample_id"]),
        )
        object.__setattr__(
            self,
            "coding_blind_cols",
            list(self.blind_cols),
        )
        object.__setattr__(
            self,
            "analysis_blind_cols",
            list(self.blind_cols),
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
        """Return columns configured for a blinding mode."""
        if mode in {"coding", "analysis"}:
            return self.blind_cols
        raise ValueError(f"Unknown blinding mode: {mode}")

    def should_blind(self, mode: str) -> bool:
        """Return True when a blinding mode has configured columns."""
        return bool(self.auto_blind and self.get_blind_cols(mode))


# ------------------------------------------------------------------
# Config manager
# ------------------------------------------------------------------

class ConfigManager:
    """
    Read, validate, normalize, and expose DIAAD configuration.

    Expected config directory contents:
        project.yaml
        advanced.yaml
    """

    REQUIRED_FILES = (
        "project.yaml",
        "advanced.yaml",
    )

    def __init__(self, config_dir: str | Path) -> None:
        self.config_dir = Path(config_dir).expanduser().resolve()
        self._validate_config_dir()

        self._raw_project = self._read_yaml("project.yaml")
        self._raw_advanced = self._read_yaml("advanced.yaml")

        self.project = self._parse_project(self._raw_project)
        self.advanced = self._parse_advanced(self._raw_advanced)

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
    def num_coders(self) -> int:
        return self.project.num_coders

    @property
    def num_bins(self) -> int:
        return self.project.num_bins

    @property
    def stimulus_field(self) -> str:
        return self.project.stimulus_field

    @property
    def automate_powers(self) -> bool:
        return self.project.automate_powers

    @property
    def reliability_tag(self) -> str:
        return self.advanced.reliability_tag

    @property
    def reliability_dirname(self) -> str:
        return self.advanced.reliability_dirname

    @property
    def target_vocabulary_resource_path(self) -> str:
        return self.advanced.target_vocabulary_resource_path

    @property
    def cu_paradigms(self) -> list[str]:
        return self.advanced.cu_paradigms

    @property
    def cu_utts_file(self) -> str:
        return self.advanced.cu_utts_file

    @property
    def cu_samples_file(self) -> str:
        return self.advanced.cu_samples_file

    @property
    def word_count_file(self) -> str:
        return self.advanced.word_count_file

    @property
    def word_count_field(self) -> str:
        return self.advanced.word_count_field

    @property
    def wc_samples_file(self) -> str:
        return self.advanced.wc_samples_file

    @property
    def speaking_time_file(self) -> str:
        return self.advanced.speaking_time_file

    @property
    def speaking_time_field(self) -> str:
        return self.advanced.speaking_time_field

    @property
    def metadata_fields_config(self) -> dict[str, Any]:
        """
        Return normalized metadata field definitions in the shape expected
        by MetadataManager.
        """
        return {
            "tiers": self.project.metadata_fields,
        }

    @property
    def blinding(self) -> AdvancedConfig:
        """
        Return advanced config for modules that expect blinding settings.

        Blinding settings now live in advanced.yaml, but downstream code only
        needs the blinding-related attributes on this object.
        """
        return self.advanced

    @property
    def id_cols(self) -> list[str]:
        return self.advanced.id_cols

    @property
    def auto_blind(self) -> bool:
        return self.advanced.auto_blind

    @property
    def blind_cols(self) -> list[str]:
        return self.advanced.blind_cols

    @property
    def blinded_suffix(self) -> str:
        return self.advanced.blinded_suffix

    @property
    def coding_blind_cols(self) -> list[str]:
        return self.advanced.get_blind_cols(mode="coding")

    @property
    def analysis_blind_cols(self) -> list[str]:
        return self.advanced.get_blind_cols(mode="analysis")

    @property
    def metadata_source(self) -> str:
        return self.advanced.metadata_source

    @property
    def codebook_filename(self) -> str:
        return self.advanced.codebook_filename

    @property
    def metadata_path(self) -> Path:
        return Path(self.input_dir) / self.metadata_source

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
                "num_bins": self.project.num_bins,
                "num_coders": self.project.num_coders,
                "stimulus_field": self.project.stimulus_field,
                "automate_powers": self.project.automate_powers,
                "metadata_fields": self.project.metadata_fields,
            },
            "advanced": {
                "reliability_tag": self.advanced.reliability_tag,
                "reliability_dirname": self.advanced.reliability_dirname,
                "cu_paradigms": self.advanced.cu_paradigms,
                "cu_samples_file": self.advanced.cu_samples_file,
                "cu_utts_file": self.advanced.cu_utts_file,
                "word_count_file": self.advanced.word_count_file,
                "word_count_field": self.advanced.word_count_field,
                "wc_samples_file": self.advanced.wc_samples_file,
                "speaking_time_file": self.advanced.speaking_time_file,
                "speaking_time_field": self.advanced.speaking_time_field,
                "target_vocabulary_resource_path": (
                    self.advanced.target_vocabulary_resource_path
                ),
                "auto_blind": self.advanced.auto_blind,
                "blind_cols": self.advanced.blind_cols,
                "metadata_source": self.advanced.metadata_source,
                "id_cols": self.advanced.id_cols,
                "codebook_filename": self.advanced.codebook_filename,
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
    # Section parsing
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
            num_bins=self._as_int(data.get("num_bins"), default=4),
            num_coders=self._as_int(data.get("num_coders"), default=0),
            stimulus_field=self._as_str(data.get("stimulus_field"), default=""),
            automate_powers=self._as_bool(data.get("automate_powers"), default=True),
            metadata_fields=self._parse_metadata_fields(data),
        )

        if not 0 < project.reliability_fraction <= 1:
            raise ValueError(
                f"reliability_fraction must be > 0 and <= 1; got {project.reliability_fraction}"
            )
        if project.num_bins < 1:
            raise ValueError(f"num_bins must be >= 1; got {project.num_bins}")

        return project

    def _parse_advanced(self, data: dict[str, Any]) -> AdvancedConfig:
        return AdvancedConfig(
            reliability_tag=self._as_str(
                data.get("reliability_tag"),
                default="_reliability",
            ),
            reliability_dirname=self._as_str(
                data.get("reliability_dirname"),
                default="reliability",
            ),
            cu_paradigms=self._as_str_list(data.get("cu_paradigms"), default=[]),
            cu_samples_file=self._as_str(
                data.get("cu_samples_file"),
                default="cu_coding_by_sample_long.xlsx",
            ),
            cu_utts_file=self._as_str(
                data.get("cu_utts_file"),
                default="cu_coding_by_utterance.xlsx",
            ),
            word_count_file=self._as_str(
                data.get("word_count_file"),
                default="word_counting.xlsx",
            ),
            word_count_field=self._as_str(
                data.get("word_count_field"),
                default="word_count",
            ),
            wc_samples_file=self._as_str(
                data.get("wc_samples_file"),
                default="word_counting_by_sample.xlsx",
            ),
            speaking_time_file=self._as_str(
                data.get("speaking_time_file"),
                default="speaking_times.xlsx",
            ),
            speaking_time_field=self._as_str(
                data.get("speaking_time_field"),
                default="speaking_time",
            ),
            target_vocabulary_resource_path=self._as_str(
                data.get("target_vocabulary_resource_path"),
                default="",
            ),
            metadata_source=self._as_str(
                data.get("metadata_source"),
                default="transcript_tables",
            ),
            auto_blind=self._as_bool(data.get("auto_blind"), default=False),
            blind_cols=self._as_optional_str_list(
                data.get(
                    "blind_cols",
                    data.get("coding_blind_cols", data.get("analysis_blind_cols")),
                ),
            ),
            coding_blind_cols=self._as_optional_str_list(
                data.get("coding_blind_cols"),
            ),
            analysis_blind_cols=self._as_optional_str_list(
                data.get("analysis_blind_cols"),
            ),
            id_cols=self._as_optional_str_list(data.get("id_cols")),
            codebook_filename=self._as_str(
                data.get("codebook_filename"),
                default="",
            ),
        )

    def _parse_metadata_fields(
        self,
        data: dict[str, Any],
    ) -> dict[str, MetadataFieldSpec]:
        fields = data.get("metadata_fields", data.get("tiers", {}))

        if fields is None:
            return {}
        if not isinstance(fields, dict):
            raise TypeError("project.yaml: 'metadata_fields' must be a dictionary.")

        normalized_fields: dict[str, MetadataFieldSpec] = {}

        for field_name, field_spec in fields.items():
            field_name = str(field_name)

            if isinstance(field_spec, str):
                if not field_spec.strip():
                    raise ValueError(
                        f"Metadata field '{field_name}' regex string must be non-empty."
                    )
                normalized_fields[field_name] = field_spec

            elif isinstance(field_spec, list):
                if not all(isinstance(v, str) for v in field_spec):
                    raise TypeError(
                        f"Metadata field '{field_name}' values must be a list of strings."
                    )
                normalized_fields[field_name] = list(field_spec)

            else:
                raise TypeError(
                    f"Metadata field '{field_name}' must be either a regex string or a list[str]."
                )

        return normalized_fields

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

    def _as_optional_str_list(self, value: Any) -> list[str] | None:
        if value is None:
            return None
        return self._as_str_list(value)

    def _as_int_list(
        self,
        value: Any,
        default: list[int] | None = None,
    ) -> list[int]:
        items = self._as_list(value, default=default)
        return [self._as_int(item) for item in items]
