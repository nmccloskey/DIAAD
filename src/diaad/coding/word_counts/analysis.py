from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from psair.core.logger import logger, get_rel_path
from diaad.metadata.discovery import find_one_matching_file
from diaad.metadata.blinding import blind_analysis_dataframe, write_blind_codebook
from diaad.metadata.unblinding import maybe_unblind_dataframe


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_WORD_COUNT_FILE = "word_counting.xlsx"
DEFAULT_WORD_COUNT_FIELD = "word_count"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _drop_admin_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop coder/admin columns not needed in analysis outputs."""
    df = df.copy()
    df.drop(
        columns=[c for c in ["id", "comment", "wc_comment"] if c in df.columns],
        inplace=True,
        errors="ignore",
    )
    return df


def _coerce_word_count_series(series: pd.Series) -> pd.Series:
    """
    Coerce a word-count series to numeric.

    Assumptions
    -----------
    - Numeric values are valid counts.
    - Explicit neutral values like 'NA' become NaN.
    - Non-numeric junk becomes NaN.
    """
    return pd.to_numeric(series, errors="coerce")


def _count_nonmissing(x: pd.Series) -> float:
    """Count non-missing values; return NaN if the group is empty."""
    return int(x.notna().sum()) if len(x) > 0 else np.nan


def _count_missing(x: pd.Series) -> float:
    """Count missing values; return NaN if the group is empty."""
    return int(x.isna().sum()) if len(x) > 0 else np.nan


def _sum_words(x: pd.Series) -> float:
    """Sum non-missing word counts; NaN if no coded utterances exist."""
    return x.sum(min_count=1)


def _mean_words(x: pd.Series) -> float:
    """Mean words per coded utterance."""
    return x.mean()


def _sd_words(x: pd.Series) -> float:
    """Sample standard deviation of words per coded utterance."""
    return x.std(ddof=1)


def _min_words(x: pd.Series) -> float:
    """Minimum words among coded utterances."""
    return x.min()


def _max_words(x: pd.Series) -> float:
    """Maximum words among coded utterances."""
    return x.max()


def _summarize_word_counts(
    wc_df: pd.DataFrame,
    word_count_field: str,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Summarize utterance-level word counts by sample.

    Output metrics
    --------------
    - no_utt_coded: number of non-missing coded utterances
    - no_utt_missing: number of missing / neutral utterances
    - total_words: sum of utterance word counts
    - mean_words_per_utt
    - sd_words_per_utt
    - min_words_per_utt
    - max_words_per_utt
    """
    grouped = wc_df.groupby(sample_id_field)[word_count_field]

    summary = grouped.agg(
        no_utt_coded=_count_nonmissing,
        no_utt_missing=_count_missing,
        total_words=_sum_words,
        mean_words_per_utt=_mean_words,
        sd_words_per_utt=_sd_words,
        min_words_per_utt=_min_words,
        max_words_per_utt=_max_words,
    ).reset_index()

    numeric_cols = [
        "total_words",
        "mean_words_per_utt",
        "sd_words_per_utt",
        "min_words_per_utt",
        "max_words_per_utt",
    ]
    for col in numeric_cols:
        if col in summary.columns:
            summary[col] = summary[col].round(3)

    return summary


def _maybe_unblind_word_count_outputs(
    *,
    wc_utts: pd.DataFrame,
    wc_summary: pd.DataFrame | None,
    blinding_config=None,
    blind_codebook=None,
    input_dir=None,
    output_dir=None,
    sample_id_field: str = "sample_id",
):
    """
    Unblind sample identifiers in word-count analysis outputs if a coding-stage
    blind codebook is available.

    This function does not require transcript tables and does not reblind
    any outputs.
    """
    if blinding_config is None:
        return wc_utts, wc_summary, None

    target_cols = [sample_id_field]

    unblinded_wc_utts, codebook_df = maybe_unblind_dataframe(
        df=wc_utts,
        config=blinding_config,
        blind_codebook=blind_codebook,
        target_cols=target_cols,
        directories=[input_dir, output_dir],
        strict=False,
    )

    unblinded_wc_summary = None
    if wc_summary is not None:
        unblinded_wc_summary, _ = maybe_unblind_dataframe(
            df=wc_summary,
            config=blinding_config,
            blind_codebook=codebook_df if codebook_df is not None else blind_codebook,
            target_cols=target_cols,
            directories=[input_dir, output_dir],
            strict=False,
        )

    return unblinded_wc_utts, unblinded_wc_summary, codebook_df


def _codebook_covers_targets(codebook_df: pd.DataFrame | None, target_cols: list[str]) -> bool:
    if codebook_df is None or codebook_df.empty or "column" not in codebook_df.columns:
        return False
    available = set(codebook_df["column"].dropna().astype(str))
    return set(target_cols).issubset(available)


def _maybe_blind_word_count_outputs(
    *,
    wc_utts: pd.DataFrame,
    wc_summary: pd.DataFrame | None,
    blinding_config=None,
    codebook_df: pd.DataFrame | None = None,
    input_dir=None,
    output_dir=None,
    out_dir=None,
):
    if blinding_config is None or not blinding_config.should_blind("analysis"):
        return wc_utts, wc_summary

    target_cols = blinding_config.get_blind_cols("analysis")
    reusable_codebook = (
        codebook_df if _codebook_covers_targets(codebook_df, target_cols) else None
    )

    blinded_utts, diagnostics_df, analysis_codebook = blind_analysis_dataframe(
        wc_utts,
        blinding_config,
        existing_codebook=reusable_codebook,
        directories=[input_dir, output_dir],
    )

    blinded_summary = None
    if wc_summary is not None:
        blinded_summary, _, _ = blind_analysis_dataframe(
            wc_summary,
            blinding_config,
            existing_codebook=analysis_codebook,
            directories=[input_dir, output_dir],
        )

    if out_dir is not None and analysis_codebook is not None and not analysis_codebook.empty:
        write_blind_codebook(
            analysis_codebook,
            Path(out_dir) / "word_count_analysis_blind_codebook.xlsx",
        )
        if not diagnostics_df.empty:
            diagnostics_df.to_excel(
                Path(out_dir) / "word_count_analysis_blinding_diagnostics.xlsx",
                index=False,
            )

    return blinded_utts, blinded_summary


def _write_word_count_analysis_outputs(
    wc_utts: pd.DataFrame,
    wc_summary: pd.DataFrame | None,
    out_dir,
) -> None:
    """Write utterance- and sample-level word-count analysis files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    utterance_path = out_dir / "word_counting_by_utterance.xlsx"
    try:
        wc_utts.to_excel(utterance_path, index=False)
        logger.info(f"Saved utterance-level word-count analysis: {get_rel_path(utterance_path)}")
    except Exception as e:
        logger.error(f"Failed writing utterance-level file {get_rel_path(utterance_path)}: {e}")
        return

    if wc_summary is None or wc_summary.empty:
        logger.warning(f"No valid word-count summaries for {get_rel_path(out_dir)}")
        return

    summary_path = out_dir / "word_counting_by_sample.xlsx"
    try:
        wc_summary.to_excel(summary_path, index=False)
        logger.info(f"Saved word-count summary file: {get_rel_path(summary_path)}")
    except Exception as e:
        logger.error(f"Failed saving word-count summary to {get_rel_path(summary_path)}: {e}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def analyze_word_counts(
    input_dir,
    output_dir,
    word_count_file: str | Path | None = None,
    word_count_field: str = DEFAULT_WORD_COUNT_FIELD,
    blinding_config=None,
    blind_codebook=None,
    sample_id_field: str = "sample_id",
):
    """
    Summarize utterance-level word-count coding by sample.

    Expected input
    --------------
    A workbook such as `word_counting.xlsx` with columns like:
        sample_id, utterance_id, speaker, utterance, comment, id,
        word_count, wc_comment

    Behavior
    --------
    - Finds exactly one word-count coding workbook by exact filename.
    - Coerces the configured word-count field to numeric.
    - Treats non-numeric / neutral entries (e.g., 'NA') as missing.
    - Writes:
        * utterance-level file with cleaned numeric word-count column
        * sample-level summary with total, mean, SD, min, max

    Unblinding
    ----------
    If a coding-stage blind codebook is available, sample identifiers are
    unblinded in the analysis outputs. No transcript tables are required.
    This function does not reblind outputs.
    """
    word_count_file = word_count_file or DEFAULT_WORD_COUNT_FILE

    wc_analysis_dir = Path(output_dir) / "word_count_analysis"
    wc_analysis_dir.mkdir(parents=True, exist_ok=True)

    cod = find_one_matching_file(
        directories=[input_dir, output_dir],
        filename=word_count_file,
        label="word-count coding file",
    )

    try:
        wc_df = pd.read_excel(cod)
        logger.info(f"Processing word-count coding file: {get_rel_path(cod)}")
    except Exception as e:
        logger.error(f"Failed reading {get_rel_path(cod)}: {e}")
        return

    required_cols = [sample_id_field, word_count_field]
    missing = [c for c in required_cols if c not in wc_df.columns]
    if missing:
        logger.error(
            f"Word-count coding file is missing required columns: {missing}. "
            f"Available columns: {list(wc_df.columns)}"
        )
        return

    wc_df = _drop_admin_cols(wc_df)
    wc_df[word_count_field] = _coerce_word_count_series(wc_df[word_count_field])

    n_missing = int(wc_df[word_count_field].isna().sum())
    if n_missing:
        logger.info(
            f"{n_missing} utterance rows have missing/non-numeric `{word_count_field}` "
            "after coercion."
        )

    try:
        wc_summary = _summarize_word_counts(
            wc_df=wc_df,
            word_count_field=word_count_field,
            sample_id_field=sample_id_field,
        )
    except Exception as e:
        logger.error(f"Word-count aggregation failed for {get_rel_path(cod)}: {e}")
        return

    wc_df, wc_summary, codebook_df = _maybe_unblind_word_count_outputs(
        wc_utts=wc_df,
        wc_summary=wc_summary,
        blinding_config=blinding_config,
        blind_codebook=blind_codebook,
        input_dir=input_dir,
        output_dir=output_dir,
        sample_id_field=sample_id_field,
    )

    wc_df, wc_summary = _maybe_blind_word_count_outputs(
        wc_utts=wc_df,
        wc_summary=wc_summary,
        blinding_config=blinding_config,
        codebook_df=codebook_df,
        input_dir=input_dir,
        output_dir=output_dir,
        out_dir=wc_analysis_dir,
    )

    _write_word_count_analysis_outputs(
        wc_utts=wc_df,
        wc_summary=wc_summary,
        out_dir=wc_analysis_dir,
    )

    return wc_summary
