from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from psair.core.logger import logger, get_rel_path
from psair.metadata.discovery import find_matching_files


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

DEFAULT_SPEAKING_TIME_FILE = "speaking_times.xlsx"
DEFAULT_SPEAKING_TIME_FIELD = "speaking_time"


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _normalize_to_list(value):
    """Return value as a list, preserving None -> []."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def validate_columns(df: pd.DataFrame, required_cols: list[str], df_name: str = "DataFrame") -> None:
    """Raise a helpful error if required columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{df_name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def present_cols(df: pd.DataFrame, preferred_order: list[str]) -> list[str]:
    """Return preferred columns that are actually present."""
    return [c for c in preferred_order if c in df.columns]


# ---------------------------------------------------------------------
# Reading / loading speaking times
# ---------------------------------------------------------------------

def read_speaking_time_table(
    speaking_time_file: str | Path | None = None,
    speaking_time_field: str = DEFAULT_SPEAKING_TIME_FIELD,
    directories=None,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Read and standardize a speaking-time spreadsheet.

    Expected structure
    ------------------
    Must contain:
        - sample_id (or the configured `sample_id_field`)
        - speaking_time  (or the configured `speaking_time_field`)

    Assumptions
    -----------
    - speaking_time is in seconds
    - one row per sample_id preferred
    - if duplicate sample_ids exist, values are summed with a warning

    Parameters
    ----------
    speaking_time_file : str | Path | None
        Exact file path, or filename/base pattern to search for.
        If None, defaults to DEFAULT_SPEAKING_TIME_FILE.
    speaking_time_field : str
        Name of the speaking time column in the file.
    directories : Path | str | list[Path | str] | None
        Directories to search if `speaking_time_file` is not an existing path.

    Returns
    -------
    pd.DataFrame
        Standardized table with columns:
            sample_id
            speaking_time
            speaking_minutes
    """
    speaking_time_file = speaking_time_file or DEFAULT_SPEAKING_TIME_FILE

    candidate_path = Path(speaking_time_file)
    if candidate_path.exists():
        matches = [candidate_path]
    else:
        search_base = Path(speaking_time_file).stem
        matches = find_matching_files(
            directories=directories,
            search_base=search_base,
            search_ext=".xlsx",
        )

    if not matches:
        raise FileNotFoundError(
            f"No speaking time file found matching: {speaking_time_file}"
        )

    if len(matches) > 1:
        logger.warning(
            "Multiple speaking time files detected; "
            f"using first in list: {get_rel_path(matches[0])}"
        )

    path = Path(matches[0])

    try:
        df = pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(f"Failed reading speaking time file {get_rel_path(path)}: {e}") from e

    validate_columns(
        df,
        required_cols=[sample_id_field, speaking_time_field],
        df_name="Speaking time table",
    )

    df = df[[sample_id_field, speaking_time_field]].copy()
    df = df.rename(columns={speaking_time_field: "speaking_time"})

    df["speaking_time"] = pd.to_numeric(df["speaking_time"], errors="coerce")

    bad_time = df["speaking_time"].isna().sum()
    if bad_time:
        logger.warning(
            f"Speaking time table contains {bad_time} rows with missing/non-numeric "
            "speaking_time; these rows will be retained as NaN."
        )

    dupes = df[sample_id_field].duplicated(keep=False)
    if dupes.any():
        n_dupe_rows = int(dupes.sum())
        logger.warning(
            f"Speaking time table contains {n_dupe_rows} rows with duplicate {sample_id_field} "
            f"values; collapsing by {sample_id_field} and summing speaking_time."
        )
        df = (
            df.groupby(sample_id_field, as_index=False, dropna=False)["speaking_time"]
            .sum(min_count=1)
        )

    df["speaking_minutes"] = df["speaking_time"] / 60.0

    logger.info(f"Loaded speaking times from {get_rel_path(path)}")
    return df


def speaking_time_dict_from_table(
    speaking_time_df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> dict:
    """
    Convert a standardized speaking time table to a sample_id -> seconds dict.
    """
    validate_columns(
        speaking_time_df,
        required_cols=[sample_id_field, "speaking_time"],
        df_name="Speaking time table",
    )

    return dict(zip(speaking_time_df[sample_id_field], speaking_time_df["speaking_time"]))


def load_speaking_time_dict(
    speaking_time_file: str | Path | None = None,
    speaking_time_field: str = DEFAULT_SPEAKING_TIME_FIELD,
    directories=None,
    sample_id_field: str = "sample_id",
) -> dict:
    """
    Convenience wrapper to read the speaking time file and return
    sample_id -> speaking_time_seconds.
    """
    df = read_speaking_time_table(
        speaking_time_file=speaking_time_file,
        speaking_time_field=speaking_time_field,
        directories=directories,
        sample_id_field=sample_id_field,
    )
    return speaking_time_dict_from_table(df, sample_id_field=sample_id_field)


# ---------------------------------------------------------------------
# Merge / compute helpers
# ---------------------------------------------------------------------

def merge_speaking_time(
    df: pd.DataFrame,
    speaking_time_df: pd.DataFrame,
    how: str = "left",
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Merge speaking-time information onto an analysis table by sample_id.

    Parameters
    ----------
    df : pd.DataFrame
        Analysis table with sample_id.
    speaking_time_df : pd.DataFrame
        Standardized speaking-time table from read_speaking_time_table().
    how : str
        Merge type; usually 'left'.

    Returns
    -------
    pd.DataFrame
        Original table plus:
            speaking_time
            speaking_minutes
    """
    validate_columns(df, [sample_id_field], df_name="Analysis table")
    validate_columns(
        speaking_time_df,
        [sample_id_field, "speaking_time", "speaking_minutes"],
        df_name="Speaking time table",
    )

    merged = pd.merge(
        df,
        speaking_time_df[[sample_id_field, "speaking_time", "speaking_minutes"]],
        on=sample_id_field,
        how=how,
    )

    missing = int(merged["speaking_time"].isna().sum())
    if missing:
        logger.warning(
            f"{missing} rows in analysis table have no matched speaking_time."
        )

    return merged


def compute_rate_per_minute(
    value_series: pd.Series,
    speaking_minutes_series: pd.Series,
) -> pd.Series:
    """
    Compute per-minute rate from a count-like series and speaking minutes.

    Returns NaN where numerator or denominator is missing, or where
    speaking_minutes <= 0.
    """
    value = pd.to_numeric(value_series, errors="coerce")
    minutes = pd.to_numeric(speaking_minutes_series, errors="coerce")

    rate = pd.Series(np.nan, index=value.index, dtype="float64")
    valid = value.notna() & minutes.notna() & (minutes > 0)
    rate.loc[valid] = value.loc[valid] / minutes.loc[valid]
    return rate


def add_rate_columns(
    df: pd.DataFrame,
    numerator_cols: list[str],
    speaking_minutes_col: str = "speaking_minutes",
    suffix: str = "_per_min",
    round_digits: int | None = 3,
) -> pd.DataFrame:
    """
    Add per-minute rate columns for one or more numerator columns.

    Example
    -------
    numerator_cols = ["cu", "p_sv", "p_rel"]
    ->
    cu_per_min, p_sv_per_min, p_rel_per_min
    """
    out = df.copy()

    validate_columns(out, [speaking_minutes_col], df_name="Rate input table")

    for col in numerator_cols:
        if col not in out.columns:
            logger.warning(f"Skipping missing numerator column: {col}")
            continue

        new_col = f"{col}{suffix}"
        out[new_col] = compute_rate_per_minute(
            out[col],
            out[speaking_minutes_col],
        )

        if round_digits is not None:
            out[new_col] = out[new_col].round(round_digits)

    return out
