from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from diaad.core.logger import logger, _rel
from diaad.transcripts.transcript_tables import extract_transcript_data
from diaad.metadata.utils import validate_columns
from diaad.metadata.blinding import blind_file_identifiers, write_blind_codebook


DEFAULT_NUM_BINS = 4
DEFAULT_STIMULUS_COLUMN = "narrative"
TEMPLATE_SUBDIR = "coding_templates"


@dataclass(frozen=True)
class CodingTemplateConfig:
    """
    Normalized settings for generic manual coding template generation.
    """
    num_bins: int = DEFAULT_NUM_BINS
    stimulus_column: str = DEFAULT_STIMULUS_COLUMN

    @property
    def use_stimulus(self) -> bool:
        return bool(str(self.stimulus_column or "").strip())


def _normalize_coder_ids(
    coder_ids: Optional[list[str] | tuple[str, ...] | str],
) -> list[str]:
    """
    Normalize coder IDs to a non-empty list.

    Blank / None becomes a single blank coder_id column.
    """
    if coder_ids is None:
        return [""]

    if isinstance(coder_ids, str):
        coder_ids = [coder_ids]

    normalized = [str(x).strip() for x in coder_ids]
    normalized = [x for x in normalized if x != ""]

    return normalized or [""]


def _require_columns(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
    """
    Validate required columns are present.
    """
    validate_columns(df, cols, df_name=df_name)


def _coerce_bin_labels(num_bins: int) -> list[int]:
    """
    Return canonical bin labels as 1..num_bins.
    """
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}.")
    return list(range(1, num_bins + 1))


def _prepare_stimulus_lookup(
    sample_df: pd.DataFrame,
    stimulus_column: str,
) -> pd.DataFrame:
    """
    Return sample_id + stimulus lookup table.

    The output stimulus column is renamed to 'stimulus'.
    """
    _require_columns(sample_df, ["sample_id", stimulus_column], "sample_df")

    stim_df = sample_df.loc[:, ["sample_id", stimulus_column]].copy()
    stim_df = stim_df.rename(columns={stimulus_column: "stimulus"})
    stim_df = stim_df.drop_duplicates(subset=["sample_id"])

    return stim_df


def _prepare_utterance_template(
    utt_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    config: CodingTemplateConfig,
) -> pd.DataFrame:
    """
    Build the base utterance-level coding template.

    Columns:
        sample_id, utterance_id, stimulus?, utterance
    """
    _require_columns(utt_df, ["sample_id", "utterance_id", "utterance"], "utt_df")

    out = utt_df.loc[:, ["sample_id", "utterance_id", "utterance"]].copy()

    if config.use_stimulus:
        stim_df = _prepare_stimulus_lookup(sample_df, config.stimulus_column)
        out = out.merge(stim_df, on="sample_id", how="left", validate="m:1")
        out = out.loc[:, ["sample_id", "utterance_id", "stimulus", "utterance"]]

    return out


def _prepare_sample_template(
    sample_df: pd.DataFrame,
    config: CodingTemplateConfig,
) -> pd.DataFrame:
    """
    Build the base sample-level coding template.

    Columns:
        sample_id, stimulus?, bin
    """
    _require_columns(sample_df, ["sample_id"], "sample_df")

    cols = ["sample_id"]
    if config.use_stimulus:
        _require_columns(sample_df, [config.stimulus_column], "sample_df")
        cols.append(config.stimulus_column)

    out = sample_df.loc[:, cols].copy().drop_duplicates(subset=["sample_id"])

    if config.use_stimulus:
        out = out.rename(columns={config.stimulus_column: "stimulus"})
        out["bin"] = pd.NA
        out = out.loc[:, ["sample_id", "stimulus", "bin"]]
    else:
        out["bin"] = pd.NA
        out = out.loc[:, ["sample_id", "bin"]]

    return out


def _expand_by_coder(
    df: pd.DataFrame,
    coder_ids: list[str],
    insert_after: str,
) -> pd.DataFrame:
    """
    Replicate rows once per coder and insert coder_id after insert_after.
    """
    frames = []

    for coder_id in coder_ids:
        part = df.copy()
        part["coder_id"] = coder_id
        frames.append(part)

    out = pd.concat(frames, ignore_index=True)

    cols = list(out.columns)
    cols.remove("coder_id")
    insert_idx = cols.index(insert_after) + 1
    cols = cols[:insert_idx] + ["coder_id"] + cols[insert_idx:]

    return out.loc[:, cols]


def _apply_optional_identifier_blinding(
    df: pd.DataFrame,
    *,
    blind: bool,
    blinding_config,
    existing_codebook: pd.DataFrame | None = None,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optionally blind identifiers for manual coding files.

    Returns
    -------
    blinded_or_raw_df, codebook_df
    """
    if not blind:
        return df.copy(), pd.DataFrame()

    blinded_df, codebook_df = blind_file_identifiers(
        df=df,
        config=blinding_config,
        existing_codebook=existing_codebook,
        seed=seed,
    )
    return blinded_df, codebook_df


def build_utterance_coding_template(
    transcript_table_path: str | Path,
    *,
    coder_ids: Optional[list[str] | tuple[str, ...] | str] = None,
    template_config: CodingTemplateConfig | None = None,
    blind: bool = False,
    blinding_config=None,
    existing_codebook: pd.DataFrame | None = None,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build an utterance-level blank coding template from transcript tables.

    Output columns:
        sample_id, utterance_id, coder_id, stimulus?, utterance
    """
    template_config = template_config or CodingTemplateConfig()
    coder_ids = _normalize_coder_ids(coder_ids)

    sample_df = extract_transcript_data(transcript_table_path, kind="sample")
    utt_df = extract_transcript_data(transcript_table_path, kind="utterance")

    out = _prepare_utterance_template(
        utt_df=utt_df,
        sample_df=sample_df,
        config=template_config,
    )
    out = _expand_by_coder(out, coder_ids, insert_after="utterance_id")

    if blind:
        if blinding_config is None:
            raise ValueError("blinding_config is required when blind=True.")
        out, codebook_df = _apply_optional_identifier_blinding(
            out,
            blind=True,
            blinding_config=blinding_config,
            existing_codebook=existing_codebook,
            seed=seed,
        )
    else:
        codebook_df = pd.DataFrame()

    logger.info(
        f"Prepared utterance-level coding template with {len(out)} row(s)."
    )
    return out, codebook_df


def build_sample_coding_template(
    transcript_table_path: str | Path,
    *,
    coder_ids: Optional[list[str] | tuple[str, ...] | str] = None,
    template_config: CodingTemplateConfig | None = None,
    blind: bool = False,
    blinding_config=None,
    existing_codebook: pd.DataFrame | None = None,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a sample-level blank coding template from transcript tables.

    Output columns:
        sample_id, coder_id, stimulus?, bin
    """
    template_config = template_config or CodingTemplateConfig()
    coder_ids = _normalize_coder_ids(coder_ids)

    sample_df = extract_transcript_data(transcript_table_path, kind="sample")
    out = _prepare_sample_template(sample_df=sample_df, config=template_config)
    out = _expand_by_coder(out, coder_ids, insert_after="sample_id")

    if blind:
        if blinding_config is None:
            raise ValueError("blinding_config is required when blind=True.")
        out, codebook_df = _apply_optional_identifier_blinding(
            out,
            blind=True,
            blinding_config=blinding_config,
            existing_codebook=existing_codebook,
            seed=seed,
        )
    else:
        codebook_df = pd.DataFrame()

    logger.info(
        f"Prepared sample-level coding template with {len(out)} row(s)."
    )
    return out, codebook_df


def add_balanced_bins(
    df: pd.DataFrame,
    *,
    num_bins: int = DEFAULT_NUM_BINS,
    sample_id_col: str = "sample_id",
    bin_col: str = "bin",
) -> pd.DataFrame:
    """
    Fill bin assignments across unique samples as evenly as possible.

    Useful if the user wants an initial blank sample-level template with
    pre-assigned coding bins.
    """
    if sample_id_col not in df.columns:
        raise ValueError(f"'{sample_id_col}' not found in dataframe.")

    bin_labels = _coerce_bin_labels(num_bins)

    out = df.copy()
    sample_order = (
        out.loc[:, [sample_id_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    sample_order[bin_col] = [
        bin_labels[i % num_bins] for i in range(len(sample_order))
    ]

    out = out.drop(columns=[bin_col], errors="ignore")
    out = out.merge(sample_order, on=sample_id_col, how="left", validate="m:1")

    return out


def write_coding_template(
    df: pd.DataFrame,
    path: str | Path,
    *,
    codebook_df: pd.DataFrame | None = None,
    codebook_path: str | Path | None = None,
) -> Path:
    """
    Write a coding template to disk.

    Supports .xlsx and .csv. Optionally writes a paired codebook.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="coding_template")
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Template path must end in .xlsx or .csv")

    logger.info(f"Wrote coding template: {_rel(path)}")

    if codebook_df is not None and not codebook_df.empty:
        if codebook_path is None:
            stem = path.with_suffix("")
            codebook_path = f"{stem}_codebook.xlsx"
        write_blind_codebook(codebook_df, codebook_path)

    return path


def make_utterance_coding_template(
    transcript_table_path: str | Path,
    output_path: str | Path,
    *,
    coder_ids: Optional[list[str] | tuple[str, ...] | str] = None,
    num_bins: int = DEFAULT_NUM_BINS,
    stimulus_column: str = DEFAULT_STIMULUS_COLUMN,
    blind: bool = False,
    blinding_config=None,
    existing_codebook: pd.DataFrame | None = None,
    codebook_path: str | Path | None = None,
    seed: int = 99,
) -> Path:
    """
    Convenience wrapper: build and write an utterance-level coding template.
    """
    config = CodingTemplateConfig(
        num_bins=num_bins,
        stimulus_column=stimulus_column,
    )
    df, codebook_df = build_utterance_coding_template(
        transcript_table_path,
        coder_ids=coder_ids,
        template_config=config,
        blind=blind,
        blinding_config=blinding_config,
        existing_codebook=existing_codebook,
        seed=seed,
    )
    return write_coding_template(
        df,
        output_path,
        codebook_df=codebook_df,
        codebook_path=codebook_path,
    )


def make_sample_coding_template(
    transcript_table_path: str | Path,
    output_path: str | Path,
    *,
    coder_ids: Optional[list[str] | tuple[str, ...] | str] = None,
    num_bins: int = DEFAULT_NUM_BINS,
    stimulus_column: str = DEFAULT_STIMULUS_COLUMN,
    prefill_bins: bool = False,
    blind: bool = False,
    blinding_config=None,
    existing_codebook: pd.DataFrame | None = None,
    codebook_path: str | Path | None = None,
    seed: int = 99,
) -> Path:
    """
    Convenience wrapper: build and write a sample-level coding template.
    """
    config = CodingTemplateConfig(
        num_bins=num_bins,
        stimulus_column=stimulus_column,
    )
    df, codebook_df = build_sample_coding_template(
        transcript_table_path,
        coder_ids=coder_ids,
        template_config=config,
        blind=blind,
        blinding_config=blinding_config,
        existing_codebook=existing_codebook,
        seed=seed,
    )

    if prefill_bins:
        df = add_balanced_bins(df, num_bins=config.num_bins)

    return write_coding_template(
        df,
        output_path,
        codebook_df=codebook_df,
        codebook_path=codebook_path,
    )
