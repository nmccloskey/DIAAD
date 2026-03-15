from __future__ import annotations

from typing import Optional, Tuple

import random
import pandas as pd
from pathlib import Path

from diaad.core.logger import logger, _rel
from diaad.io.discovery import find_matching_files
from diaad.transcripts.transcript_tables import extract_transcript_data
from diaad.core.config import BlindingConfig

DEFAULT_ID_COLS = ("sample_id", "utterance_id")


def _normalize_to_list(x):
    """Return x as a list, preserving order for tuples/lists and wrapping scalars."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _present_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return requested columns that are present in df, preserving order."""
    return [c for c in cols if c in df.columns]


def _validate_columns(df: pd.DataFrame, required_cols: list[str], df_name: str = "DataFrame") -> None:
    """Raise ValueError if any required columns are absent."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _choose_join_keys(
    df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    id_cols: tuple[str, ...] | list[str] = DEFAULT_ID_COLS,
) -> list[str]:
    """
    Choose the most specific usable join keys shared by df and metadata_df.

    Preference:
    1. all requested keys that exist in both frames
    2. if that includes utterance_id, use both sample_id + utterance_id
    3. else if sample_id exists in both, use sample_id
    4. otherwise error
    """
    requested = list(id_cols)
    shared = [c for c in requested if c in df.columns and c in metadata_df.columns]

    if "utterance_id" in shared and "sample_id" in shared:
        return ["sample_id", "utterance_id"]
    if "sample_id" in shared:
        return ["sample_id"]

    if shared:
        logger.warning(
            f"Using nonstandard join key(s) for blinding metadata resolution: {shared}"
        )
        return shared

    raise ValueError(
        "Could not determine join keys shared by input dataframe and metadata dataframe. "
        f"Requested id_cols={requested}"
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


def _load_metadata_from_transcript_tables(
    transcript_tables=None,
    match_tiers=None,
    directories=None,
) -> pd.DataFrame:
    """
    Load and concatenate joined transcript-table metadata.

    Parameters
    ----------
    transcript_tables : Path | str | list[Path | str] | None
        Explicit transcript table path(s). If omitted, search via find_files().
    match_tiers : list[str] | None
        Tier labels used only when searching for transcript tables.
    directories : Path | str | list[Path | str] | None
        Directories searched when transcript_tables is omitted.

    Returns
    -------
    pd.DataFrame
        Concatenated joined transcript metadata.
    """
    if transcript_tables is None:
        transcript_tables = find_matching_files(
            match_tiers=match_tiers,
            directories=directories,
            search_base="transcript_tables",
            search_ext=".xlsx",
        )

    transcript_tables = [Path(p) for p in _normalize_to_list(transcript_tables)]

    if not transcript_tables:
        raise FileNotFoundError("No transcript tables found for metadata resolution.")

    metadata_dfs = []
    for path in transcript_tables:
        try:
            joined = extract_transcript_data(path, type="joined")
            joined["file"] = path.name
            metadata_dfs.append(joined)
        except Exception as e:
            logger.error(f"Failed loading transcript metadata from {_rel(path)}: {e}")
            raise

    metadata_df = pd.concat(metadata_dfs, ignore_index=True)
    logger.info(
        f"Loaded joined transcript metadata from {len(transcript_tables)} transcript table(s)"
    )
    return metadata_df


def _resolve_blind_source_columns(
    df: pd.DataFrame,
    blind_cols: list[str],
    metadata_df: pd.DataFrame | None = None,
    id_cols: tuple[str, ...] | list[str] = DEFAULT_ID_COLS,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Ensure requested blind columns are present in a working dataframe.

    If some blind columns are absent from df, recover them from metadata_df by joining
    on the most specific available id keys.

    Returns
    -------
    working_df : pd.DataFrame
        Original df plus any recovered blind columns.
    resolved_cols : list[str]
        Blind columns successfully resolved and present in working_df.
    recovered_cols : list[str]
        Blind columns recovered from metadata_df.
    """
    blind_cols = list(dict.fromkeys(blind_cols))
    present = _present_cols(df, blind_cols)
    missing = [c for c in blind_cols if c not in present]

    if not missing:
        return df.copy(), present, []

    if metadata_df is None:
        raise ValueError(
            f"Requested blind columns not found in df and no metadata_df provided: {missing}"
        )

    _validate_columns(metadata_df, [], df_name="metadata_df")
    join_keys = _choose_join_keys(df, metadata_df, id_cols=id_cols)

    recoverable = [c for c in missing if c in metadata_df.columns]
    still_missing = [c for c in missing if c not in metadata_df.columns]

    if still_missing:
        raise ValueError(
            f"Requested blind columns not found in df or metadata_df: {still_missing}"
        )

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

    resolved_cols = _present_cols(working_df, blind_cols)
    logger.info(
        f"Recovered blind columns from metadata via join on {join_keys}: {recoverable}"
    )
    return working_df, resolved_cols, recoverable


def _make_code(prefix: str, idx: int, width: int = 3) -> str:
    """Format a blind code like GRP001."""
    return f"{prefix}{idx:0{width}d}"


def generate_blind_codebook(
    df: pd.DataFrame,
    blind_cols: list[str],
    *,
    seed: int = 99,
    code_prefixes: dict[str, str] | None = None,
    width: int = 3,
    include_na: bool = False,
) -> pd.DataFrame:
    """
    Generate a tabular codebook for selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the columns to code.
    blind_cols : list[str]
        Columns to blind.
    seed : int, default 99
        Random seed for reproducible code assignment order.
    code_prefixes : dict[str, str] | None
        Optional per-column code prefixes, e.g. {"group": "GRP", "timepoint": "TP"}.
        Defaults to the first 3 uppercase letters of each column name.
    width : int, default 3
        Numeric zero-padding width.
    include_na : bool, default False
        Whether to assign explicit blind codes to NA values.

    Returns
    -------
    pd.DataFrame
        Codebook with columns: column, raw_value, blind_code
    """
    blind_cols = list(dict.fromkeys(blind_cols))
    _validate_columns(df, blind_cols, df_name="df")

    rng = random.Random(seed)
    code_prefixes = code_prefixes or {}
    rows = []

    for col in blind_cols:
        values = pd.Series(df[col].unique())

        if include_na:
            values = values.tolist()
        else:
            values = values.dropna().tolist()

        # Use string sort for determinism before shuffle, then shuffle reproducibly.
        values = sorted(values, key=lambda x: str(x))
        rng.shuffle(values)

        prefix = code_prefixes.get(col, "".join(ch for ch in col.upper() if ch.isalnum())[:3] or "BLD")

        for idx, raw_value in enumerate(values, start=1):
            rows.append(
                {
                    "column": col,
                    "raw_value": raw_value,
                    "blind_code": _make_code(prefix, idx, width=width),
                }
            )

    codebook = pd.DataFrame(rows)

    if codebook.empty:
        logger.warning("Generated empty blind codebook.")
    else:
        logger.info(
            f"Generated blind codebook for {len(blind_cols)} column(s): {blind_cols}"
        )

    return codebook


def apply_blind_codebook(
    df: pd.DataFrame,
    codebook_df: pd.DataFrame,
    *,
    append: bool = True,
    suffix: str = "_blind",
    inplace: bool = False,
    preserve_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Apply a tabular blind codebook to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    codebook_df : pd.DataFrame
        Tabular codebook with columns: column, raw_value, blind_code.
    append : bool, default True
        If True, append blinded columns with `suffix`.
        If False, replace original columns in place.
    suffix : str, default "_blind"
        Suffix for appended blinded columns.
    inplace : bool, default False
        If True, modify df in place.
    preserve_unmapped : bool, default True
        If True, values absent from the codebook remain unchanged.
        If False, unmapped values become NA.

    Returns
    -------
    pd.DataFrame
        Dataframe with blinded columns applied.
    """
    required = ["column", "raw_value", "blind_code"]
    _validate_columns(codebook_df, required, df_name="codebook_df")

    out = df if inplace else df.copy()

    for col in codebook_df["column"].drop_duplicates():
        if col not in out.columns:
            logger.warning(f"Codebook column '{col}' not present in dataframe; skipping.")
            continue

        sub = codebook_df.loc[codebook_df["column"] == col, ["raw_value", "blind_code"]]
        mapping = dict(zip(sub["raw_value"], sub["blind_code"]))

        blinded = out[col].map(mapping)

        if preserve_unmapped:
            blinded = blinded.where(blinded.notna(), out[col])

        target_col = f"{col}{suffix}" if append else col
        out[target_col] = blinded

        logger.info(
            f"Applied blind codes to column '{col}' -> '{target_col}'"
        )

    return out


def blind_dataframe(
    df: pd.DataFrame,
    blind_cols: list[str],
    *,
    metadata_df: pd.DataFrame | str | Path | list[str | Path] | None = None,
    match_tiers=None,
    directories=None,
    id_cols: tuple[str, ...] | list[str] = DEFAULT_ID_COLS,
    existing_codebook: pd.DataFrame | None = None,
    append: bool = True,
    suffix: str = "_blind",
    preserve_unmapped: bool = True,
    drop_recovered_source_cols: bool = False,
    seed: int = 99,
    code_prefixes: dict[str, str] | None = None,
    width: int = 3,
    include_na: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Blind selected columns in a dataframe, optionally recovering missing columns
    from transcript-table metadata.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to blind.
    blind_cols : list[str]
        Columns to blind.
    metadata_df : pd.DataFrame | None
        Optional metadata dataframe already loaded by caller.
    transcript_tables : Path | str | list[Path | str] | None
        Optional transcript table path(s) used only if metadata_df is not provided.
    match_tiers : list[str] | None
        Tier labels used when searching for transcript tables.
    directories : Path | str | list[Path | str] | None
        Directories searched when transcript_tables is omitted.
    id_cols : tuple[str, ...] | list[str], default ('sample_id', 'utterance_id')
        Preferred join keys for recovering metadata columns.
    existing_codebook : pd.DataFrame | None
        Optional precomputed codebook to reuse.
    append : bool, default True
        Append blinded columns rather than replacing originals.
    suffix : str, default '_blind'
        Suffix for appended blind columns.
    preserve_unmapped : bool, default True
        Preserve original values when a code is unavailable.
    seed : int, default 99
        Random seed for codebook generation.
    code_prefixes : dict[str, str] | None
        Optional prefixes for blind codes by column.
    width : int, default 3
        Numeric zero-padding width for generated blind codes.
    include_na : bool, default False
        Whether to generate explicit codes for NA values.

    Returns
    -------
    blinded_df : pd.DataFrame
        Dataframe with blinded columns applied.
    codebook_df : pd.DataFrame
        Tabular codebook with columns: column, raw_value, blind_code.
    """
    blind_cols = list(dict.fromkeys(blind_cols))
    if not blind_cols:
        raise ValueError("blind_cols must contain at least one column name.")

    working_metadata = metadata_df
    missing_from_df = [c for c in blind_cols if c not in df.columns]

    if missing_from_df:
        if isinstance(working_metadata, pd.DataFrame):
            pass
        elif working_metadata is None:
            working_metadata = _load_metadata_from_transcript_tables(
                transcript_tables=None,
                match_tiers=match_tiers,
                directories=directories,
            )
        else:
            working_metadata = _load_metadata_from_transcript_tables(
                transcript_tables=working_metadata,
                match_tiers=match_tiers,
                directories=directories,
            )

    working_df, resolved_cols, recovered_cols = _resolve_blind_source_columns(
        df=df,
        blind_cols=blind_cols,
        metadata_df=working_metadata,
        id_cols=id_cols,
    )

    if not resolved_cols:
        raise ValueError("No requested blind columns could be resolved.")

    if existing_codebook is not None:
        codebook_df = existing_codebook.copy()
        logger.info("Using existing blind codebook.")
    else:
        codebook_df = generate_blind_codebook(
            df=working_df,
            blind_cols=resolved_cols,
            seed=seed,
            code_prefixes=code_prefixes,
            width=width,
            include_na=include_na,
        )

    blinded_df = apply_blind_codebook(
        df=working_df,
        codebook_df=codebook_df,
        append=append,
        suffix=suffix,
        preserve_unmapped=preserve_unmapped,
        inplace=False,
    )

    if drop_recovered_source_cols and recovered_cols and append:
        cols_to_drop = [c for c in recovered_cols if c in blinded_df.columns]
        if append:
            blinded_df = blinded_df.drop(columns=cols_to_drop, errors="ignore")
            logger.info(
                f"Dropped recovered raw source columns after appending blind columns: {cols_to_drop}"
            )

    if recovered_cols:
        logger.info(
            f"Blind dataframe created using metadata-recovered columns: {recovered_cols}"
        )
    else:
        logger.info("Blind dataframe created using columns present in input df.")

    return blinded_df, codebook_df


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

    logger.info(f"Blind codebook written to {_rel(path)}")


def validate_blind_codebook_compatibility(
    df: pd.DataFrame,
    codebook_df: pd.DataFrame,
    blind_cols: list[str],
    *,
    require_value_coverage: bool = True,
    allow_extra_codebook_columns: bool = True,
    include_na: bool = False,
) -> None:
    """
    Validate that a blind codebook is compatible with a dataframe and target columns.

    Parameters
    ----------
    df
        DataFrame whose values may be blinded.
    codebook_df
        Tabular codebook with columns: 'column', 'raw_value', 'blind_code'.
    blind_cols
        Columns that must be supported by the codebook.
    require_value_coverage
        If True, require the codebook to contain mappings for all observed values
        in `df[blind_cols]` (excluding NA unless include_na=True).
    allow_extra_codebook_columns
        If False, raise an error when the codebook contains mappings for columns
        outside `blind_cols`.
    include_na
        If True, treat NA as a value that must be explicitly represented in the
        codebook when require_value_coverage=True.

    Raises
    ------
    ValueError
        If the codebook is incompatible.
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

    # Check for duplicate raw_value mappings within a column
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
        codebook_values = codebook_df.loc[codebook_df["column"] == col, "raw_value"]

        if include_na:
            observed = pd.Series(df[col].unique()).tolist()
            covered = codebook_values.tolist()

            missing_values = []
            for value in observed:
                if pd.isna(value):
                    if not codebook_values.isna().any():
                        missing_values.append(value)
                elif value not in set(v for v in covered if not pd.isna(v)):
                    missing_values.append(value)
        else:
            observed = pd.Series(df[col].dropna().unique()).tolist()
            covered = set(v for v in codebook_values.tolist() if not pd.isna(v))
            missing_values = [value for value in observed if value not in covered]

        if missing_values:
            uncovered[col] = sorted(missing_values, key=lambda x: str(x))

    if uncovered:
        raise ValueError(
            f"Codebook does not cover all observed dataframe values for requested "
            f"blind columns: {uncovered}"
        )

    logger.info(
        f"Blind codebook compatibility check passed for columns {blind_cols} "
        "(including observed value coverage)."
    )


def blind_file_identifiers(
    df: pd.DataFrame,
    config: BlindingConfig,
    *,
    existing_codebook: Optional[pd.DataFrame] = None,
    seed: int = 99,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Blind identifier columns in a dataframe used for manual coding files.

    This function replaces identifier columns (e.g., sample_id) with blinded
    codes so that human coders cannot infer metadata from the identifiers.

    Parameters
    ----------
    df
        DataFrame containing coding rows.
    config
        Blinding configuration object.
    existing_codebook
        Optional existing blind codebook. If provided and compatible,
        it will be reused.
    seed
        Random seed used when generating a new codebook.

    Returns
    -------
    blinded_df
        DataFrame with identifier columns replaced by blind codes.
    codebook_df
        Codebook used for blinding.
    """

    blind_cols = config.file_blind_cols

    if not blind_cols:
        logger.warning("No file_blind_cols configured; returning dataframe unchanged.")
        return df.copy(), pd.DataFrame()

    for col in blind_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' required for file blinding is not present in dataframe."
            )

    # ---------------------------------------------------------
    # Determine whether to reuse or generate a codebook
    # ---------------------------------------------------------

    if existing_codebook is not None:

        validate_blind_codebook_compatibility(
            df=df,
            codebook_df=existing_codebook,
            blind_cols=blind_cols,
            require_value_coverage=True,
            allow_extra_codebook_columns=True,
            include_na=False,
        )
        logger.info("Reusing existing blind codebook for file identifiers.")
        codebook_df = existing_codebook

    else:

        logger.info(
            f"Generating new blind codebook for identifier columns: {blind_cols}"
        )

        codebook_df = generate_blind_codebook(
            df=df,
            blind_cols=blind_cols,
            seed=seed,
            code_prefixes=config.code_prefixes,
        )

    # ---------------------------------------------------------
    # Apply the codebook (replace identifiers directly)
    # ---------------------------------------------------------

    blinded_df = apply_blind_codebook(
        df=df,
        codebook_df=codebook_df,
        append=False,
        inplace=False,
    )

    logger.info(
        f"Identifier columns blinded for coding files: {blind_cols}"
    )

    return blinded_df, codebook_df
