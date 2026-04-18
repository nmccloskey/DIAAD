from __future__ import annotations

from pathlib import Path
import pandas as pd

from psair.core.logger import logger, get_rel_path
from psair.metadata.discovery import find_matching_files
from diaad.transcripts.transcript_tables import extract_transcript_data


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def normalize_to_list(x):
    """Return x as a list, preserving order for tuples/lists and wrapping scalars."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def present_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return requested columns that are present in df, preserving order."""
    return [c for c in cols if c in df.columns]


def validate_columns(
    df: pd.DataFrame,
    required_cols: list[str],
    df_name: str = "DataFrame",
) -> None:
    """Raise ValueError if any required columns are absent."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


# ---------------------------------------------------------------------
# Metadata loading / recovery
# ---------------------------------------------------------------------

def load_metadata_from_transcript_tables(
    transcript_tables=None,
    match_tiers=None,
    directories=None,
    *,
    combine: bool = True,
    include_source_file: bool = True,
) -> pd.DataFrame:
    """
    Load joined transcript-table metadata from one or more transcript tables.

    Parameters
    ----------
    transcript_tables : Path | str | list[Path | str] | None
        Explicit transcript table path(s). If None, matching files are discovered
        with ``find_matching_files(...)`` using ``search_base='transcript_tables'``.
    match_tiers : list[str] | None
        Tier labels passed to ``find_matching_files`` when ``transcript_tables`` is None.
    directories : Path | str | list[Path | str] | None
        Directories passed to ``find_matching_files`` when ``transcript_tables`` is None.
    combine : bool, default True
        If True, load and concatenate all resolved transcript tables.
        If False, warn when multiple are found and use only the first.
    include_source_file : bool, default True
        If True, append a ``file`` column containing the source filename.

    Returns
    -------
    pd.DataFrame
        Joined transcript metadata.

    Raises
    ------
    FileNotFoundError
        If no transcript tables are found.
    """
    if transcript_tables is None:
        transcript_tables = find_matching_files(
            match_tiers=match_tiers,
            directories=directories,
            search_base="transcript_tables",
            search_ext=".xlsx",
        )

    transcript_tables = [Path(p) for p in normalize_to_list(transcript_tables)]

    if not transcript_tables:
        raise FileNotFoundError("No transcript tables found for metadata resolution.")

    if not combine and len(transcript_tables) > 1:
        logger.warning(
            "Multiple transcript tables detected for metadata resolution; "
            f"using first in list: {get_rel_path(transcript_tables[0])}"
        )
        transcript_tables = [transcript_tables[0]]

    metadata_dfs = []

    for path in transcript_tables:
        try:
            joined = extract_transcript_data(path, type="joined")
            if include_source_file:
                joined["file"] = path.name
            metadata_dfs.append(joined)
        except Exception as e:
            logger.error(f"Failed loading transcript metadata from {get_rel_path(path)}: {e}")
            raise

    if len(metadata_dfs) == 1:
        metadata_df = metadata_dfs[0].copy()
    else:
        metadata_df = pd.concat(metadata_dfs, ignore_index=True)

    if combine:
        logger.info(
            f"Loaded joined transcript metadata from {len(transcript_tables)} "
            "transcript table(s)"
        )
    else:
        logger.info(f"Loaded transcript metadata from {get_rel_path(transcript_tables[0])}")

    return metadata_df
