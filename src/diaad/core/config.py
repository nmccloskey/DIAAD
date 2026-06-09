from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any, TypeAlias

from psair.core.config_files import (
    build_config_source_metadata,
    load_sectioned_config,
    load_yaml_mapping,
    merge_defaults,
)
from psair.core.logger import logger
from psair.core.provenance import diff_config_values

from diaad.core.config_overrides import apply_config_overrides


# ------------------------------------------------------------------
# Section dataclasses
# ------------------------------------------------------------------

MetadataFieldSpec: TypeAlias = str | list[str]

CONFIG_SECTIONS = ("project", "advanced")
DEFAULT_CONFIG_PACKAGE = "diaad.config"
DEFAULT_CONFIG_FILENAME = "default_config.yaml"


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
    exclude_speakers: list[str] | None = None

    auto_tabularize: bool = False

    num_bins: int = 4
    num_coders: int = 0
    stimulus_column: str = ""
    automate_powers: bool = True

    metadata_fields: dict[str, MetadataFieldSpec] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exclude_speakers",
            list(self.exclude_speakers or []),
        )
        object.__setattr__(
            self,
            "metadata_fields",
            dict(self.metadata_fields or {}),
        )


@dataclass(frozen=True)
class AdvancedConfig:
    """Normalized advanced configuration."""

    transcript_table_filename: str = "transcript_tables.xlsx"

    sample_id_column: str = "sample_id"
    utterance_id_column: str = "utterance_id"

    reliability_tag: str = "_reliability"
    reliability_dirname: str = "reliability"

    cu_paradigms: list[str] | None = None
    cu_samples_filename: str = "cu_coding_by_sample_long.xlsx"
    cu_utts_filename: str = "cu_coding_by_utterance.xlsx"

    word_count_filename: str = "word_counting.xlsx"
    word_count_column: str = "word_count"
    wc_samples_filename: str = "word_counting_by_sample.xlsx"

    speaking_time_filename: str = "speaking_times.xlsx"
    speaking_time_column: str = "speaking_time"

    powers_coding_filename: str = "powers_coding.xlsx"
    powers_reliability_filename: str = "powers_reliability_coding.xlsx"
    spacy_model_name: str = "en_core_web_sm"

    target_vocabulary_resource_path: str = ""

    auto_blind: bool = False
    blind_columns: list[str] | None = None
    metadata_source: str = "transcript_tables.xlsx"
    codebook_filename: str = ""

    def __post_init__(self) -> None:
        sample_id_column = str(self.sample_id_column).strip()
        utterance_id_column = str(self.utterance_id_column).strip()
        if not sample_id_column:
            raise ValueError("sample_id_column must be a non-empty string.")
        if not utterance_id_column:
            raise ValueError("utterance_id_column must be a non-empty string.")
        spacy_model_name = str(self.spacy_model_name).strip()
        if not spacy_model_name:
            raise ValueError("spacy_model_name must be a non-empty string.")
        object.__setattr__(self, "sample_id_column", sample_id_column)
        object.__setattr__(self, "utterance_id_column", utterance_id_column)
        object.__setattr__(self, "spacy_model_name", spacy_model_name)
        object.__setattr__(self, "cu_paradigms", list(self.cu_paradigms or []))
        object.__setattr__(
            self,
            "blind_columns",
            list(self.blind_columns or ["sample_id"]),
        )

    @property
    def blinded_suffix(self) -> str:
        """Suffix used for blinded output columns."""
        return "_blinded"

    def get_blind_cols(self, mode: str) -> list[str]:
        """Return columns configured for a blinding mode."""
        if mode in {"coding", "analysis"}:
            return self.blind_columns
        raise ValueError(f"Unknown blinding mode: {mode}")

    def should_blind(self, mode: str) -> bool:
        """Return True when a blinding mode has configured columns."""
        return bool(self.auto_blind and self.get_blind_cols(mode))

    @property
    def sample_id_field(self) -> str:
        return self.sample_id_column

    @property
    def utterance_id_field(self) -> str:
        return self.utterance_id_column

    @property
    def blind_cols(self) -> list[str]:
        return self.blind_columns

    @property
    def transcript_table_file(self) -> str:
        return self.transcript_table_filename

    @property
    def cu_samples_file(self) -> str:
        return self.cu_samples_filename

    @property
    def cu_utts_file(self) -> str:
        return self.cu_utts_filename

    @property
    def word_count_file(self) -> str:
        return self.word_count_filename

    @property
    def word_count_field(self) -> str:
        return self.word_count_column

    @property
    def wc_samples_file(self) -> str:
        return self.wc_samples_filename

    @property
    def speaking_time_file(self) -> str:
        return self.speaking_time_filename

    @property
    def speaking_time_field(self) -> str:
        return self.speaking_time_column

    @property
    def powers_coding_file(self) -> str:
        return self.powers_coding_filename

    @property
    def powers_reliability_file(self) -> str:
        return self.powers_reliability_filename

    @property
    def spacy_model(self) -> str:
        return self.spacy_model_name

    @property
    def coding_blind_cols(self) -> list[str]:
        return self.blind_columns

    @property
    def analysis_blind_cols(self) -> list[str]:
        return self.blind_columns


# ------------------------------------------------------------------
# Config manager
# ------------------------------------------------------------------

class ConfigManager:
    """
    Read, validate, normalize, and expose DIAAD configuration.

    Configuration may be provided as a split directory containing
    project.yaml/advanced.yaml, a nested YAML file with project:/advanced:
    sections, or omitted to use the packaged defaults.
    """

    def __init__(
        self,
        config_dir: str | Path | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.config_dir = (
            Path(config_dir).expanduser().resolve()
            if config_dir is not None
            else None
        )
        self.config_overrides = dict(config_overrides or {})

        default_config = self._load_default_config()
        user_config, user_source = self._load_user_config(self.config_dir)
        merged_config = merge_defaults(default_config, user_config)
        self.config_source = build_config_source_metadata(
            user_source,
            default_path=self.default_config_path(),
            missing_sections=[
                section
                for section in CONFIG_SECTIONS
                if section not in user_config or not user_config[section]
            ],
            defaults_applied=True,
        )

        self._raw_project = merged_config["project"]
        self._raw_advanced = merged_config["advanced"]

        base_project = self._parse_project(self._raw_project)
        base_advanced = self._parse_advanced(self._raw_advanced)
        self._base_dict = self._to_dict(base_project, base_advanced)

        effective_project, effective_advanced = apply_config_overrides(
            self._raw_project,
            self._raw_advanced,
            self.config_overrides,
        )
        self.project = self._parse_project(effective_project)
        self.advanced = self._parse_advanced(effective_advanced)
        self.override_diff = diff_config_values(self._base_dict, self.to_dict())

        logger.info(
            "Loaded DIAAD configuration from %s (%s; defaults applied from %s)",
            self.config_source["path"],
            self.config_source["kind"],
            self.config_source["default_path"],
        )

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
    def exclude_speakers(self) -> list[str]:
        return self.project.exclude_speakers

    @property
    def num_coders(self) -> int:
        return self.project.num_coders

    @property
    def num_bins(self) -> int:
        return self.project.num_bins

    @property
    def stimulus_field(self) -> str:
        return self.project.stimulus_column

    @property
    def stimulus_column(self) -> str:
        return self.project.stimulus_column

    @property
    def automate_powers(self) -> bool:
        return self.project.automate_powers

    @property
    def auto_tabularize(self) -> bool:
        return self.project.auto_tabularize

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
    def transcript_table_file(self) -> str:
        return self.advanced.transcript_table_filename

    @property
    def transcript_table_filename(self) -> str:
        return self.advanced.transcript_table_filename

    @property
    def cu_paradigms(self) -> list[str]:
        return self.advanced.cu_paradigms

    @property
    def cu_utts_file(self) -> str:
        return self.advanced.cu_utts_filename

    @property
    def cu_utts_filename(self) -> str:
        return self.advanced.cu_utts_filename

    @property
    def cu_samples_file(self) -> str:
        return self.advanced.cu_samples_filename

    @property
    def cu_samples_filename(self) -> str:
        return self.advanced.cu_samples_filename

    @property
    def word_count_file(self) -> str:
        return self.advanced.word_count_filename

    @property
    def word_count_filename(self) -> str:
        return self.advanced.word_count_filename

    @property
    def word_count_field(self) -> str:
        return self.advanced.word_count_column

    @property
    def word_count_column(self) -> str:
        return self.advanced.word_count_column

    @property
    def wc_samples_file(self) -> str:
        return self.advanced.wc_samples_filename

    @property
    def wc_samples_filename(self) -> str:
        return self.advanced.wc_samples_filename

    @property
    def speaking_time_file(self) -> str:
        return self.advanced.speaking_time_filename

    @property
    def speaking_time_filename(self) -> str:
        return self.advanced.speaking_time_filename

    @property
    def speaking_time_field(self) -> str:
        return self.advanced.speaking_time_column

    @property
    def speaking_time_column(self) -> str:
        return self.advanced.speaking_time_column

    @property
    def sample_id_field(self) -> str:
        return self.advanced.sample_id_column

    @property
    def sample_id_column(self) -> str:
        return self.advanced.sample_id_column

    @property
    def utterance_id_field(self) -> str:
        return self.advanced.utterance_id_column

    @property
    def utterance_id_column(self) -> str:
        return self.advanced.utterance_id_column

    @property
    def powers_coding_file(self) -> str:
        return self.advanced.powers_coding_filename

    @property
    def powers_coding_filename(self) -> str:
        return self.advanced.powers_coding_filename

    @property
    def powers_reliability_file(self) -> str:
        return self.advanced.powers_reliability_filename

    @property
    def powers_reliability_filename(self) -> str:
        return self.advanced.powers_reliability_filename

    @property
    def spacy_model_name(self) -> str:
        return self.advanced.spacy_model_name

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
    def auto_blind(self) -> bool:
        return self.advanced.auto_blind

    @property
    def blind_cols(self) -> list[str]:
        return self.advanced.blind_columns

    @property
    def blind_columns(self) -> list[str]:
        return self.advanced.blind_columns

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
        return self._to_dict(self.project, self.advanced)

    @staticmethod
    def _to_dict(project: ProjectConfig, advanced: AdvancedConfig) -> dict[str, Any]:
        """Return normalized configuration dataclasses as a nested dictionary."""
        return {
            "project": {
                "input_dir": project.input_dir,
                "output_dir": project.output_dir,
                "random_seed": project.random_seed,
                "reliability_fraction": project.reliability_fraction,
                "shuffle_samples": project.shuffle_samples,
                "strip_clan": project.strip_clan,
                "prefer_correction": project.prefer_correction,
                "lowercase": project.lowercase,
                "exclude_speakers": project.exclude_speakers,
                "auto_tabularize": project.auto_tabularize,
                "num_bins": project.num_bins,
                "num_coders": project.num_coders,
                "stimulus_column": project.stimulus_column,
                "automate_powers": project.automate_powers,
                "metadata_fields": project.metadata_fields,
            },
            "advanced": {
                "transcript_table_filename": advanced.transcript_table_filename,
                "sample_id_column": advanced.sample_id_column,
                "utterance_id_column": advanced.utterance_id_column,
                "reliability_tag": advanced.reliability_tag,
                "reliability_dirname": advanced.reliability_dirname,
                "cu_paradigms": advanced.cu_paradigms,
                "cu_samples_filename": advanced.cu_samples_filename,
                "cu_utts_filename": advanced.cu_utts_filename,
                "word_count_filename": advanced.word_count_filename,
                "word_count_column": advanced.word_count_column,
                "wc_samples_filename": advanced.wc_samples_filename,
                "speaking_time_filename": advanced.speaking_time_filename,
                "speaking_time_column": advanced.speaking_time_column,
                "powers_coding_filename": advanced.powers_coding_filename,
                "powers_reliability_filename": advanced.powers_reliability_filename,
                "spacy_model_name": advanced.spacy_model_name,
                "target_vocabulary_resource_path": (
                    advanced.target_vocabulary_resource_path
                ),
                "auto_blind": advanced.auto_blind,
                "blind_columns": advanced.blind_columns,
                "metadata_source": advanced.metadata_source,
                "codebook_filename": advanced.codebook_filename,
            },
        }

    # ------------------------------------------------------------------
    # Internal file loading
    # ------------------------------------------------------------------

    @staticmethod
    def default_config_path() -> Path:
        """Return the packaged default config path when available on disk."""
        return Path(resource_files(DEFAULT_CONFIG_PACKAGE) / DEFAULT_CONFIG_FILENAME)

    def _load_default_config(self) -> dict[str, Any]:
        default_path = self.default_config_path()
        data = load_yaml_mapping(default_path)
        return {
            section: data.get(section, {})
            for section in CONFIG_SECTIONS
        }

    def _load_user_config(
        self,
        path: Path | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if path is None:
            logger.info("No DIAAD config source provided; using packaged defaults.")
            return {}, {
                "kind": "packaged_default",
                "path": None,
                "files": {},
                "missing_sections": list(CONFIG_SECTIONS),
            }

        loaded = load_sectioned_config(
            path,
            CONFIG_SECTIONS,
            allow_missing_sections=True,
        )

        return loaded.sections, {
            "kind": loaded.source.kind,
            "path": str(loaded.source.path),
            "files": {
                section: str(section_path)
                for section, section_path in loaded.source.files.items()
            },
            "missing_sections": list(loaded.source.missing_sections),
        }

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
            exclude_speakers=self._as_str_list(
                data.get("exclude_speakers"),
                default=[],
            ),
            auto_tabularize=self._as_bool(data.get("auto_tabularize"), default=False),
            num_bins=self._as_int(data.get("num_bins"), default=4),
            num_coders=self._as_int(data.get("num_coders"), default=0),
            stimulus_column=self._as_str(data.get("stimulus_column"), default=""),
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
            transcript_table_filename=self._as_str(
                data.get("transcript_table_filename"),
                default="transcript_tables.xlsx",
            ),
            sample_id_column=self._as_str(
                data.get("sample_id_column"),
                default="sample_id",
            ),
            utterance_id_column=self._as_str(
                data.get("utterance_id_column"),
                default="utterance_id",
            ),
            reliability_tag=self._as_str(
                data.get("reliability_tag"),
                default="_reliability",
            ),
            reliability_dirname=self._as_str(
                data.get("reliability_dirname"),
                default="reliability",
            ),
            cu_paradigms=self._as_str_list(data.get("cu_paradigms"), default=[]),
            cu_samples_filename=self._as_str(
                data.get("cu_samples_filename"),
                default="cu_coding_by_sample_long.xlsx",
            ),
            cu_utts_filename=self._as_str(
                data.get("cu_utts_filename"),
                default="cu_coding_by_utterance.xlsx",
            ),
            word_count_filename=self._as_str(
                data.get("word_count_filename"),
                default="word_counting.xlsx",
            ),
            word_count_column=self._as_str(
                data.get("word_count_column"),
                default="word_count",
            ),
            wc_samples_filename=self._as_str(
                data.get("wc_samples_filename"),
                default="word_counting_by_sample.xlsx",
            ),
            speaking_time_filename=self._as_str(
                data.get("speaking_time_filename"),
                default="speaking_times.xlsx",
            ),
            speaking_time_column=self._as_str(
                data.get("speaking_time_column"),
                default="speaking_time",
            ),
            powers_coding_filename=self._as_str(
                data.get("powers_coding_filename"),
                default="powers_coding.xlsx",
            ),
            powers_reliability_filename=self._as_str(
                data.get("powers_reliability_filename"),
                default="powers_reliability_coding.xlsx",
            ),
            spacy_model_name=self._as_str(
                data.get("spacy_model_name"),
                default="en_core_web_sm",
            ),
            target_vocabulary_resource_path=self._as_str(
                data.get("target_vocabulary_resource_path"),
                default="",
            ),
            metadata_source=self._as_str(
                data.get("metadata_source"),
                default="transcript_tables.xlsx",
            ),
            auto_blind=self._as_bool(data.get("auto_blind"), default=False),
            blind_columns=self._as_optional_str_list(data.get("blind_columns")),
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
