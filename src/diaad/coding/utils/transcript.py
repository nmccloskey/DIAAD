from __future__ import annotations

from typing import TYPE_CHECKING

from psair.core.logger import logger

if TYPE_CHECKING:
    import pandas as pd


UNINTELLIGIBLE = {"xxx", "yyy", "www"}
DEFAULT_STIM_COLS = ["narrative", "scene", "story", "stimulus"]


def resolve_stim_cols(stimulus_field):
    """Use explicit stimulus_field when provided; otherwise fall back to legacy stimulus columns."""
    return [stimulus_field] if stimulus_field else DEFAULT_STIM_COLS


def drop_excluded_speaker_rows(
    df: pd.DataFrame,
    exclude_speakers=None,
    *,
    label: str = "analysis",
) -> pd.DataFrame:
    """Remove rows whose speaker label is excluded from analysis."""
    if not exclude_speakers or "speaker" not in df.columns:
        return df

    exclude_set = {str(s).strip().lower() for s in exclude_speakers if str(s).strip()}
    if not exclude_set:
        return df

    speaker_labels = df["speaker"].astype(str).str.strip().str.lower()
    keep_mask = ~speaker_labels.isin(exclude_set)
    n_excluded = int((~keep_mask).sum())

    if n_excluded:
        logger.info(
            f"Excluded {n_excluded} {label} row(s) from analysis based on speaker label."
        )

    return df.loc[keep_mask].copy()
