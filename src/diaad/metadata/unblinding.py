from __future__ import annotations

from pathlib import Path
import pandas as pd

from diaad.core.logger import logger, _rel
from diaad.io.discovery import find_matching_files
from diaad.core.config import BlindingConfig
from diaad.metadata.utils import (
    validate_columns,
    normalize_to_list,
    load_metadata_from_transcript_tables
)


def _read_tabular_file(path: str | Path) -> pd.DataFrame:
    """
    Read a tabular file from disk.

    Supports .xlsx and .csv.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {_rel(path)}")

    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(
        f"Unsupported file type for {_rel(path)}. Expected .xlsx or .csv"
    )


def _choose_first_match(
    matches: list[str | Path] | None,
    *,
    resource_name: str,
    required: bool = False,
) -> Path | None:
    """
    Choose the first path from a list of matches, warning if multiple are present.
    """
    paths = [Path(p) for p in normalize_to_list(matches)]

    if not paths:
        if required:
            raise FileNotFoundError(f"No {resource_name} files found.")
        return None

    if len(paths) > 1:
        logger.warning(
            f"Multiple {resource_name} files detected; using first in list: "
            f"{[_rel(p) for p in paths]}"
        )

    return paths[0]


# ---------------------------------------------------------------------
# Resource finding / loading
# ---------------------------------------------------------------------

def _find_blind_codebook_path(
    *,
    blind_codebook=None,
    match_tiers=None,
    directories=None,
    search_base: str = "blind_codebook",
    search_ext: str = ".xlsx",
    required: bool = False,
) -> Path | None:
    """
    Resolve a blind codebook path.

    Priority
    --------
    1. Explicit blind_codebook path/list supplied by caller
    2. Discovery via find_matching_files(...)
    """
    if blind_codebook is not None and not isinstance(blind_codebook, pd.DataFrame):
        return _choose_first_match(
            matches=blind_codebook,
            resource_name="blind codebook",
            required=required,
        )

    matches = find_matching_files(
        match_tiers=match_tiers,
        directories=directories,
        search_base=search_base,
        search_ext=search_ext,
    )
    return _choose_first_match(
        matches=matches,
        resource_name="blind codebook",
        required=required,
    )


def _load_blind_codebook(
    *,
    blind_codebook=None,
    match_tiers=None,
    directories=None,
    search_base: str = "blind_codebook",
    search_ext: str = ".xlsx",
    required: bool = False,
) -> pd.DataFrame | None:
    """
    Load a blind codebook from a dataframe, explicit path, or discovered file.
    """
    if isinstance(blind_codebook, pd.DataFrame):
        logger.info("Using caller-supplied blind codebook dataframe.")
        return blind_codebook.copy()

    codebook_path = _find_blind_codebook_path(
        blind_codebook=blind_codebook,
        match_tiers=match_tiers,
        directories=directories,
        search_base=search_base,
        search_ext=search_ext,
        required=required,
    )

    if codebook_path is None:
        return None

    codebook_df = _read_tabular_file(codebook_path)
    logger.info(f"Loaded blind codebook from {_rel(codebook_path)}")
    return codebook_df


def _load_metadata_df(
    *,
    metadata_df=None,
    match_tiers=None,
    directories=None,
) -> pd.DataFrame | None:
    """
    Load metadata_df from a dataframe, explicit transcript-table path(s), or discovery.
    """
    if metadata_df is None:
        return load_metadata_from_transcript_tables(
            transcript_tables=None,
            match_tiers=match_tiers,
            directories=directories,
            combine=False,
        )

    if isinstance(metadata_df, pd.DataFrame):
        logger.info("Using caller-supplied metadata dataframe.")
        return metadata_df.copy()

    return load_metadata_from_transcript_tables(
        transcript_tables=metadata_df,
        match_tiers=match_tiers,
        directories=directories,
        combine=False,
    )


# ---------------------------------------------------------------------
# Decode-dictionary generation / validation
# ---------------------------------------------------------------------

def validate_decode_codebook(
    codebook_df: pd.DataFrame,
    *,
    target_cols: list[str] | None = None,
) -> None:
    """
    Validate that a codebook is structurally suitable for decoding.
    """
    required = ["column", "raw_value", "blind_code"]
    validate_columns(codebook_df, required, df_name="codebook_df")

    dupes_raw = codebook_df.duplicated(subset=["column", "raw_value"], keep=False)
    if dupes_raw.any():
        bad = (
            codebook_df.loc[dupes_raw, ["column", "raw_value"]]
            .drop_duplicates()
            .sort_values(["column", "raw_value"], key=lambda s: s.astype(str))
        )
        raise ValueError(
            "Codebook contains duplicate (column, raw_value) rows: "
            f"{bad.to_dict(orient='records')}"
        )

    dupes_blind = codebook_df.duplicated(subset=["column", "blind_code"], keep=False)
    if dupes_blind.any():
        bad = (
            codebook_df.loc[dupes_blind, ["column", "blind_code"]]
            .drop_duplicates()
            .sort_values(["column", "blind_code"], key=lambda s: s.astype(str))
        )
        raise ValueError(
            "Codebook contains duplicate (column, blind_code) rows: "
            f"{bad.to_dict(orient='records')}"
        )

    if target_cols is not None:
        target_cols = list(dict.fromkeys(target_cols))
        codebook_targets = set(codebook_df["column"].dropna().astype(str).unique())
        missing = [c for c in target_cols if c not in codebook_targets]
        if missing:
            raise ValueError(
                f"Codebook does not contain required target column(s): {missing}"
            )

    logger.info("Blind codebook passed decoding validation.")


def generate_blind_decode_dict(
    codebook_df: pd.DataFrame,
    *,
    target_cols: list[str] | None = None,
) -> dict[str, dict]:
    """
    Generate a nested decoding dictionary from a blind codebook.

    Returns
    -------
    dict[str, dict]
        Mapping of:
            {
                "column_name": {
                    blind_code: raw_value,
                    ...
                },
                ...
            }
    """
    validate_decode_codebook(codebook_df, target_cols=target_cols)

    if target_cols is None:
        target_cols = codebook_df["column"].dropna().astype(str).unique().tolist()
    else:
        target_cols = list(dict.fromkeys(target_cols))

    decode_dict = {}

    for col in target_cols:
        sub = codebook_df.loc[
            codebook_df["column"].astype(str) == str(col),
            ["blind_code", "raw_value"],
        ].copy()

        decode_dict[col] = dict(zip(sub["blind_code"], sub["raw_value"]))

    logger.info(
        f"Generated blind decode dictionary for {len(target_cols)} column(s): {target_cols}"
    )
    return decode_dict


# ---------------------------------------------------------------------
# Dataframe unblinding
# ---------------------------------------------------------------------

def unblind_dataframe(
    df: pd.DataFrame,
    codebook_df: pd.DataFrame,
    *,
    target_cols: list[str] | None = None,
    suffix: str = "_blinded",
    drop_blinded_cols: bool = True,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Unblind dataframe columns using a blind codebook.

    Supported patterns
    ------------------
    1. Analysis-style columns:
       - input contains '{col}_blinded'
       - output gets raw '{col}'

    2. In-place blinded columns (e.g., coding files):
       - input contains '{col}'
       - values are blind codes already stored in-place
       - output decodes '{col}' in-place

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to unblind.
    codebook_df : pd.DataFrame
        Blind codebook with columns: column, raw_value, blind_code
    target_cols : list[str] | None
        Target columns to decode. If None, infer from codebook.
    suffix : str, default "_blinded"
        Suffix used for blinded analysis columns.
    drop_blinded_cols : bool, default True
        If True, remove '{col}_blinded' after decoding.
    strict : bool, default False
        If True, raise an error when a requested target column is not present in df.
        If False, silently skip absent targets.
    """
    out = df.copy()

    if target_cols is None:
        target_cols = codebook_df["column"].dropna().astype(str).unique().tolist()
    else:
        target_cols = list(dict.fromkeys(target_cols))

    decode_dict = generate_blind_decode_dict(
        codebook_df=codebook_df,
        target_cols=target_cols,
    )

    decoded_any = []

    for col in target_cols:
        suffixed_col = f"{col}{suffix}"

        if suffixed_col in out.columns:
            out[col] = out[suffixed_col].map(decode_dict[col])
            if drop_blinded_cols:
                out = out.drop(columns=[suffixed_col])
            decoded_any.append(suffixed_col)
            logger.info(f"Unblinded analysis column '{suffixed_col}' -> '{col}'")
            continue

        if col in out.columns:
            out[col] = out[col].map(decode_dict[col])
            decoded_any.append(col)
            logger.info(f"Unblinded in-place coded column '{col}'")
            continue

        if strict:
            raise ValueError(
                f"Neither '{col}' nor '{suffixed_col}' found in dataframe for unblinding."
            )

    if not decoded_any:
        logger.warning("No dataframe columns were unblinded.")
    else:
        logger.info(f"Unblinded dataframe columns: {decoded_any}")

    return out


def maybe_unblind_dataframe(
    df: pd.DataFrame,
    config: BlindingConfig,
    *,
    blind_codebook=None,
    target_cols: list[str] | None = None,
    match_tiers=None,
    directories=None,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Unblind a dataframe if a blind codebook is available; otherwise return unchanged.

    This is useful for analysis modules that should dynamically respond to the
    presence or absence of blind codebooks without special-case logic.
    """
    codebook_df = _load_blind_codebook(
        blind_codebook=blind_codebook,
        match_tiers=match_tiers,
        directories=directories,
        required=False,
    )

    if codebook_df is None or codebook_df.empty:
        logger.info("No blind codebook found; returning dataframe unchanged.")
        return df.copy(), None

    unblinded_df = unblind_dataframe(
        df=df,
        codebook_df=codebook_df,
        target_cols=target_cols,
        suffix=config.blinded_suffix,
        strict=strict,
    )

    return unblinded_df, codebook_df


# ---------------------------------------------------------------------
# Optional convenience resolver
# ---------------------------------------------------------------------

def resolve_unblinding_resources(
    *,
    blind_codebook=None,
    metadata_df=None,
    match_tiers=None,
    directories=None,
    require_codebook: bool = False,
    require_metadata: bool = False,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Resolve both blind codebook and metadata_df for downstream analysis modules.

    Returns
    -------
    tuple[pd.DataFrame | None, pd.DataFrame | None]
        (codebook_df, metadata_df)
    """
    codebook_df = _load_blind_codebook(
        blind_codebook=blind_codebook,
        match_tiers=match_tiers,
        directories=directories,
        required=require_codebook,
    )

    resolved_metadata_df = None
    try:
        resolved_metadata_df = _load_metadata_df(
            metadata_df=metadata_df,
            match_tiers=match_tiers,
            directories=directories,
        )
    except FileNotFoundError:
        if require_metadata:
            raise
        logger.info("No metadata dataframe found for unblinding resource resolution.")

    return codebook_df, resolved_metadata_df
