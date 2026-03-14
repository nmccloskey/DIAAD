from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from diaad.core.logger import logger, _rel
from diaad.io.discovery import find_matching_files
from diaad.transcripts.transcript_tables import extract_transcript_data

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
