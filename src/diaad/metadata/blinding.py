from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import random
import pandas as pd
from pathlib import Path

from psair.core.logger import logger, get_rel_path
from diaad.metadata.utils import (
    present_cols,
    validate_columns,
    load_metadata_from_transcript_tables
)
from diaad.metadata.unblinding import _load_blind_codebook

if TYPE_CHECKING:
    from diaad.core.config import AdvancedConfig


def _has_codebook_discovery_context(
    *,
    match_metadata_fields=None,
    directories=None,
    codebook_filename: str = "",
) -> bool:
    return bool(match_metadata_fields or directories or str(codebook_filename or "").strip())


def _choose_join_keys(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    sample_id_field: str = "sample_id",
    utterance_id_field: str = "utterance_id",
) -> list[str]:
    """
    Choose the most specific usable join keys shared by df and metadata_df.

    Preference:
    1. configured sample + utterance identifiers if both are shared
    2. configured sample identifier if shared
    3. configured utterance identifier if shared
    4. otherwise error
    """
    sample_id_field = str(sample_id_field).strip()
    utterance_id_field = str(utterance_id_field).strip()
    requested = list(dict.fromkeys([sample_id_field, utterance_id_field]))
    shared = [c for c in requested if c in df.columns and c in metadata_df.columns]

    has_sample = sample_id_field in shared
    has_utterance = utterance_id_field in shared
    if has_sample and has_utterance and sample_id_field != utterance_id_field:
        return [sample_id_field, utterance_id_field]
    if has_sample:
        return [sample_id_field]
    if has_utterance:
        return [utterance_id_field]

    raise ValueError(
        "Could not determine join keys shared by input dataframe and metadata dataframe. "
        f"Requested identifier fields={requested}"
    )


def _deduplicate_metadata_for_join(
    metadata_df: pd.DataFrame,
    join_keys: list[str],
    needed_cols: list[str],
) -> pd.DataFrame:
    """
    Reduce metadata_df to join keys + needed columns and deduplicate on join keys.

    For sample-level joins, joined transcript metadata may contain repeated sample rows,
    so this step collapses duplicates safely.
    """
    keep_cols = [c for c in join_keys + needed_cols if c in metadata_df.columns]
    reduced = metadata_df.loc[:, keep_cols].copy()

    before = len(reduced)
    reduced = reduced.drop_duplicates(subset=join_keys)
    after = len(reduced)

    if after < before:
        logger.info(
            f"Deduplicated metadata for join on {join_keys}: {before} -> {after} rows"
        )

    return reduced


# ---------------------------------------------------------------------
# Metadata loading / recovery
# ---------------------------------------------------------------------

def _resolve_analysis_source_columns(
    df: pd.DataFrame,
    blind_cols: list[str],
    *,
    metadata_df: pd.DataFrame | None = None,
    sample_id_field: str = "sample_id",
    utterance_id_field: str = "utterance_id",
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    Ensure requested analysis blind columns are present in a working dataframe.

    Returns
    -------
    working_df
        Input df plus any recovered metadata columns.
    resolved_cols
        Blind columns successfully present in working_df.
    recovered_cols
        Blind columns recovered from metadata_df.
    join_keys
        Join keys actually used if metadata recovery occurred, else present
        configured identifier fields from df.
    """
    blind_cols = list(dict.fromkeys(blind_cols))
    present = present_cols(df, blind_cols)
    missing = [c for c in blind_cols if c not in present]

    id_fields = list(dict.fromkeys([sample_id_field, utterance_id_field]))
    present_identifier_cols = [c for c in id_fields if c in df.columns]

    if not missing:
        return df.copy(), present, [], present_identifier_cols

    if metadata_df is None:
        logger.warning(
            "Requested blind columns not found in df and no metadata_df provided; "
            "skipping columns: %s",
            missing,
        )
        return df.copy(), present, [], present_identifier_cols

    join_keys = _choose_join_keys(
        df,
        metadata_df,
        sample_id_field=sample_id_field,
        utterance_id_field=utterance_id_field,
    )

    recoverable = [c for c in missing if c in metadata_df.columns]
    still_missing = [c for c in missing if c not in metadata_df.columns]

    if still_missing:
        logger.warning(
            "Requested blind columns not found in df or metadata_df; skipping "
            "columns: %s",
            still_missing,
        )

    if not recoverable:
        return df.copy(), present, [], join_keys

    joinable_metadata = _deduplicate_metadata_for_join(
        metadata_df=metadata_df,
        join_keys=join_keys,
        needed_cols=recoverable,
    )

    working_df = df.merge(joinable_metadata, on=join_keys, how="left", validate="m:1")
    unresolved_after_join = [c for c in recoverable if c not in working_df.columns]

    if unresolved_after_join:
        raise ValueError(
            f"Failed to recover metadata columns after join: {unresolved_after_join}"
        )

    resolved_cols = present_cols(working_df, blind_cols)
    logger.info(
        f"Recovered blind columns from metadata via join on {join_keys}: {recoverable}"
    )
    return working_df, resolved_cols, recoverable, join_keys


# ---------------------------------------------------------------------
# Codebook generation / application
# ---------------------------------------------------------------------

def generate_integer_blind_codebook(
    df: pd.DataFrame,
    blind_cols: list[str],
    *,
    seed: int = 99,
) -> pd.DataFrame:
    """
    Generate integer blind codes for selected columns.

    Missing values are excluded from the codebook and remain missing when applied.

    Returns
    -------
    pd.DataFrame
        Long codebook with columns: column, raw_value, blind_code
    """
    blind_cols = list(dict.fromkeys(blind_cols))
    validate_columns(df, blind_cols, df_name="df")

    rng = random.Random(seed)
    rows = []

    for col in blind_cols:
        values = pd.Series(df[col].dropna().unique()).tolist()
        values = sorted(values, key=lambda x: str(x))
        rng.shuffle(values)

        for idx, raw_value in enumerate(values, start=1):
            rows.append(
                {
                    "column": col,
                    "raw_value": raw_value,
                    "blind_code": idx,
                }
            )

    codebook_df = pd.DataFrame(rows)

    if codebook_df.empty:
        logger.warning("Generated empty blind codebook.")
    else:
        logger.info(
            f"Generated integer blind codebook for {len(blind_cols)} column(s): {blind_cols}"
        )

    return codebook_df


def validate_blind_codebook_compatibility(
    df: pd.DataFrame,
    codebook_df: pd.DataFrame,
    blind_cols: list[str],
    *,
    require_value_coverage: bool = True,
    allow_extra_codebook_columns: bool = True,
) -> None:
    """
    Validate that a blind codebook is compatible with a dataframe and target columns.
    """
    blind_cols = list(dict.fromkeys(blind_cols))

    if not blind_cols:
        raise ValueError("blind_cols must contain at least one column name.")

    required_codebook_cols = ["column", "raw_value", "blind_code"]
    missing_codebook_cols = [c for c in required_codebook_cols if c not in codebook_df.columns]
    if missing_codebook_cols:
        raise ValueError(
            f"Codebook is missing required columns: {missing_codebook_cols}"
        )

    missing_df_cols = [c for c in blind_cols if c not in df.columns]
    if missing_df_cols:
        raise ValueError(
            f"Dataframe is missing columns required for compatibility check: {missing_df_cols}"
        )

    codebook_targets = set(codebook_df["column"].dropna().astype(str).unique())
    requested_targets = set(blind_cols)

    missing_target_cols = [c for c in blind_cols if c not in codebook_targets]
    if missing_target_cols:
        raise ValueError(
            f"Codebook does not contain required target column(s): {missing_target_cols}"
        )

    if not allow_extra_codebook_columns:
        extra_target_cols = sorted(codebook_targets - requested_targets)
        if extra_target_cols:
            raise ValueError(
                f"Codebook contains unexpected target column(s): {extra_target_cols}"
            )

    dupes = codebook_df.duplicated(subset=["column", "raw_value"], keep=False)
    if dupes.any():
        bad = (
            codebook_df.loc[dupes, ["column", "raw_value"]]
            .drop_duplicates()
            .sort_values(["column", "raw_value"], key=lambda s: s.astype(str))
        )
        preview = bad.to_dict(orient="records")
        raise ValueError(
            f"Codebook contains duplicate mappings for (column, raw_value): {preview}"
        )

    if not require_value_coverage:
        logger.info(
            f"Blind codebook compatibility check passed for columns {blind_cols} "
            "(structure only; value coverage not required)."
        )
        return

    uncovered: dict[str, list] = {}

    for col in blind_cols:
        codebook_values = set(
            v for v in codebook_df.loc[codebook_df["column"] == col, "raw_value"].tolist()
            if not pd.isna(v)
        )
        observed = pd.Series(df[col].dropna().unique()).tolist()
        missing_values = [value for value in observed if value not in codebook_values]

        if missing_values:
            uncovered[col] = sorted(missing_values, key=lambda x: str(x))

    if uncovered:
        raise ValueError(
            f"Codebook does not cover all observed dataframe values for requested "
            f"blind columns: {uncovered}"
        )

    logger.info(
        f"Blind codebook compatibility check passed for columns {blind_cols} "
        "(including observed non-missing value coverage)."
    )


def _apply_blind_codebook_as_new_columns(
    df: pd.DataFrame,
    codebook_df: pd.DataFrame,
    blind_cols: list[str],
    *,
    suffix: str,
) -> pd.DataFrame:
    """
    Append blinded columns named '{column}_blinded'.

    Missing values remain missing.
    """
    required = ["column", "raw_value", "blind_code"]
    validate_columns(codebook_df, required, df_name="codebook_df")

    out = df.copy()

    for col in blind_cols:
        if col not in out.columns:
            raise ValueError(f"Column '{col}' not present in dataframe.")

        sub = codebook_df.loc[codebook_df["column"] == col, ["raw_value", "blind_code"]]
        mapping = dict(zip(sub["raw_value"], sub["blind_code"]))

        blinded_col = f"{col}{suffix}"
        out[blinded_col] = out[col].map(mapping)

        logger.info(f"Applied blind codes to column '{col}' -> '{blinded_col}'")

    return out


def _replace_columns_with_blinded_versions(
    df: pd.DataFrame,
    blind_cols: list[str],
    *,
    suffix: str,
) -> pd.DataFrame:
    """
    Replace raw blind columns with integer-coded values in the same column names.

    Used for manual coding files.
    """
    out = df.copy()

    for col in blind_cols:
        blinded_col = f"{col}{suffix}"
        if blinded_col not in out.columns:
            raise ValueError(
                f"Expected blinded column '{blinded_col}' not found for replacement."
            )

        out[col] = out[blinded_col]
        out = out.drop(columns=[blinded_col])

    return out


# ---------------------------------------------------------------------
# Analysis blinding
# ---------------------------------------------------------------------

def blind_analysis_dataframe(
    df: pd.DataFrame,
    config: AdvancedConfig,
    *,
    metadata_df: pd.DataFrame | str | Path | list[str | Path] | None = None,
    match_metadata_fields=None,
    directories=None,
    existing_codebook: pd.DataFrame | None = None,
    discover_existing_codebook: bool = True,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Blind configured analysis columns and return both clean and diagnostic outputs.

    Returns
    -------
    blinded_df
        Analysis dataframe with original blind_cols removed and
        '{col}_blinded' columns retained.
    diagnostics_df
        Diagnostic dataframe containing join keys, original blind columns,
        and corresponding blinded columns. Missing values are preserved.
    codebook_df
        Long codebook with columns: column, raw_value, blind_code
    """
    blind_cols = list(dict.fromkeys(config.get_blind_cols("analysis") or []))
    if not blind_cols:
        logger.warning("No blind_cols configured; returning dataframe unchanged.")
        return df.copy(), pd.DataFrame(), pd.DataFrame()

    working_metadata = metadata_df
    missing_from_df = [c for c in blind_cols if c not in df.columns]

    if missing_from_df:
        if isinstance(working_metadata, pd.DataFrame):
            pass
        elif working_metadata is None:
            if config.metadata_source != "transcript_tables":
                raise ValueError(
                    f"Unsupported metadata_source: {config.metadata_source}"
                )
            working_metadata = load_metadata_from_transcript_tables(
                transcript_tables=None,
                match_metadata_fields=match_metadata_fields,
                directories=directories,
                combine=True
            )
        else:
            if config.metadata_source != "transcript_tables":
                raise ValueError(
                    f"Unsupported metadata_source: {config.metadata_source}"
                )
            working_metadata = load_metadata_from_transcript_tables(
                transcript_tables=working_metadata,
                match_metadata_fields=match_metadata_fields,
                directories=directories,
                combine=True
            )

    working_df, resolved_cols, recovered_cols, join_keys = _resolve_analysis_source_columns(
        df=df,
        blind_cols=blind_cols,
        metadata_df=working_metadata,
        sample_id_field=config.sample_id_field,
        utterance_id_field=config.utterance_id_field,
    )

    if not resolved_cols:
        raise ValueError("No requested analysis blind columns could be resolved.")

    if (
        existing_codebook is None
        and discover_existing_codebook
        and _has_codebook_discovery_context(
            match_metadata_fields=match_metadata_fields,
            directories=directories,
            codebook_filename=config.codebook_filename,
        )
    ):
        existing_codebook = _load_blind_codebook(
            match_metadata_fields=match_metadata_fields,
            directories=directories,
            codebook_filename=config.codebook_filename,
            required=False,
        )

    if existing_codebook is not None:
        validate_blind_codebook_compatibility(
            df=working_df,
            codebook_df=existing_codebook,
            blind_cols=resolved_cols,
            require_value_coverage=True,
            allow_extra_codebook_columns=True,
        )
        codebook_df = existing_codebook.copy()
        logger.info("Using existing blind codebook for analysis blinding.")
    else:
        codebook_df = generate_integer_blind_codebook(
            df=working_df,
            blind_cols=resolved_cols,
            seed=seed,
        )

    coded_df = _apply_blind_codebook_as_new_columns(
        df=working_df,
        codebook_df=codebook_df,
        blind_cols=resolved_cols,
        suffix=config.blinded_suffix
    )

    diagnostic_cols = []
    diagnostic_cols.extend([c for c in join_keys if c in coded_df.columns])
    diagnostic_cols.extend([c for c in resolved_cols if c in coded_df.columns])
    diagnostic_cols.extend(
        [f"{c}{config.blinded_suffix}" for c in resolved_cols if f"{c}{config.blinded_suffix}" in coded_df.columns]
    )

    diagnostics_df = coded_df.loc[:, list(dict.fromkeys(diagnostic_cols))].copy()

    blinded_df = coded_df.copy()
    cols_to_drop = [c for c in resolved_cols if c in blinded_df.columns]
    blinded_df = blinded_df.drop(columns=cols_to_drop)

    if recovered_cols:
        logger.info(
            f"Analysis dataframe blinded using metadata-recovered columns: {recovered_cols}"
        )
    else:
        logger.info("Analysis dataframe blinded using columns present in input df.")

    return blinded_df, diagnostics_df, codebook_df


# ---------------------------------------------------------------------
# Coding-file blinding
# ---------------------------------------------------------------------

def blind_file_identifiers(
    df: pd.DataFrame,
    config: AdvancedConfig,
    *,
    existing_codebook: Optional[pd.DataFrame] = None,
    match_metadata_fields=None,
    directories=None,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Blind identifier columns in a dataframe used for manual coding files.

    This replaces the configured identifier columns directly with integer blind codes.
    Missing values remain missing.
    """
    blind_cols = list(dict.fromkeys(config.get_blind_cols("coding") or []))

    if not blind_cols:
        logger.warning("No blind_cols configured; returning dataframe unchanged.")
        return df.copy(), pd.DataFrame()

    missing_from_df = [c for c in blind_cols if c not in df.columns]
    if missing_from_df:
        logger.warning(
            "Configured blind_cols not found in coding dataframe and will be "
            "skipped: %s",
            missing_from_df,
        )
    blind_cols = present_cols(df, blind_cols)
    if not blind_cols:
        logger.warning(
            "No configured blind_cols are present in coding dataframe; returning "
            "dataframe unchanged."
        )
        return df.copy(), pd.DataFrame()

    if existing_codebook is None and _has_codebook_discovery_context(
        match_metadata_fields=match_metadata_fields,
        directories=directories,
        codebook_filename=config.codebook_filename,
    ):
        existing_codebook = _load_blind_codebook(
            match_metadata_fields=match_metadata_fields,
            directories=directories,
            codebook_filename=config.codebook_filename,
            required=False,
        )

    if existing_codebook is not None:
        validate_blind_codebook_compatibility(
            df=df,
            codebook_df=existing_codebook,
            blind_cols=blind_cols,
            require_value_coverage=True,
            allow_extra_codebook_columns=True,
        )
        logger.info("Reusing existing blind codebook for file identifiers.")
        codebook_df = existing_codebook.copy()
    else:
        logger.info(
            f"Generating new blind codebook for identifier columns: {blind_cols}"
        )
        codebook_df = generate_integer_blind_codebook(
            df=df,
            blind_cols=blind_cols,
            seed=seed,
        )

    coded_df = _apply_blind_codebook_as_new_columns(
        df=df,
        codebook_df=codebook_df,
        blind_cols=blind_cols,
        suffix=config.blinded_suffix
    )

    blinded_df = _replace_columns_with_blinded_versions(
        df=coded_df,
        blind_cols=blind_cols,
        suffix=config.blinded_suffix
    )

    logger.info(f"Identifier columns blinded for coding files: {blind_cols}")

    return blinded_df, codebook_df


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

def write_blind_codebook(codebook_df: pd.DataFrame, path: str | Path) -> None:
    """
    Write a blind codebook to disk.

    Supports .xlsx and .csv.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".xlsx":
        codebook_df.to_excel(path, index=False)
    elif path.suffix.lower() == ".csv":
        codebook_df.to_csv(path, index=False)
    else:
        raise ValueError("Codebook path must end in .xlsx or .csv")

    logger.info(f"Blind codebook written to {get_rel_path(path)}")
