from __future__ import annotations

from pathlib import Path
import pandas as pd

from diaad.core.logger import logger, _rel
from diaad.io.discovery import find_matching_files
from diaad.coding.rates import (
    read_speaking_time_table,
    merge_speaking_time,
    add_rate_columns,
    validate_columns,
    present_cols,
)


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_WC_SAMPLES_FILE = "word_counting_by_sample.xlsx"
DEFAULT_WC_RATES_FILE = "word_counting_rates.xlsx"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_word_count_sample_summary(
    wc_samples_file: str | Path | None = None,
    directories=None,
) -> pd.DataFrame:
    """
    Read the canonical sample-level word-count summary table.

    Expected columns
    ----------------
    Required:
        - sample_id
        - total_words

    Preferred additional columns:
        - no_utt_coded
        - no_utt_missing
        - mean_words_per_utt
        - sd_words_per_utt
        - min_words_per_utt
        - max_words_per_utt

    Parameters
    ----------
    wc_samples_file : str | Path | None
        Exact file path, or filename/base pattern to search for.
        If None, defaults to DEFAULT_WC_SAMPLES_FILE.
    directories : Path | str | list[Path | str] | None
        Directories to search.

    Returns
    -------
    pd.DataFrame
        Sample-level word-count summary table.
    """
    wc_samples_file = wc_samples_file or DEFAULT_WC_SAMPLES_FILE

    candidate_path = Path(wc_samples_file)
    if candidate_path.exists():
        matches = [candidate_path]
    else:
        search_base = Path(wc_samples_file).stem
        matches = find_matching_files(
            directories=directories,
            search_base=search_base,
            search_ext=".xlsx",
        )

    if not matches:
        raise FileNotFoundError(
            f"No word-count sample summary file found matching: {wc_samples_file}"
        )

    if len(matches) > 1:
        logger.warning(
            "Multiple word-count sample summary files detected; "
            f"using first in list: {_rel(matches[0])}"
        )

    path = Path(matches[0])

    try:
        df = pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(
            f"Failed reading word-count sample summary {_rel(path)}: {e}"
        ) from e

    validate_columns(
        df,
        required_cols=["sample_id", "total_words"],
        df_name="Word-count sample summary",
    )

    logger.info(f"Loaded word-count sample summary from {_rel(path)}")
    return df


def finalize_word_count_rates_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the canonical word-count rates output columns in a clean order.
    """
    preferred = [
        "sample_id",
        "no_utt_coded",
        "no_utt_missing",
        "total_words",
        "mean_words_per_utt",
        "sd_words_per_utt",
        "min_words_per_utt",
        "max_words_per_utt",
        "speaking_time",
        "speaking_minutes",
        "total_words_per_min",
    ]
    return df[present_cols(df, preferred)].copy()


def write_word_count_rates_output(
    df: pd.DataFrame,
    output_dir,
    output_filename: str = DEFAULT_WC_RATES_FILE,
) -> Path:
    """
    Write the word-count rates table to disk.
    """
    out_dir = Path(output_dir) / "word_count_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / output_filename

    try:
        df.to_excel(out_path, index=False)
        logger.info(f"Saved word-count rates file: {_rel(out_path)}")
    except Exception as e:
        raise RuntimeError(
            f"Failed writing word-count rates file {_rel(out_path)}: {e}"
        ) from e

    return out_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def calculate_word_count_rates(
    input_dir,
    output_dir,
    wc_samples_file: str | Path | None = None,
    speaking_time_file: str | Path | None = None,
    speaking_time_field: str = "speaking_time",
):
    """
    Compute word-count rate per minute from the canonical sample-level
    word-count summary.

    Inputs
    ------
    - Word-count sample summary file, typically:
        word_counting_by_sample.xlsx
    - Speaking time spreadsheet with:
        sample_id
        speaking_time   (in seconds)

    Behavior
    --------
    - Reads the word-count sample summary.
    - Reads and standardizes the speaking time table.
    - Merges speaking times onto sample rows by sample_id.
    - Computes:
        * total_words_per_min
    - Writes a compact word-count rates table.

    Output columns
    --------------
    sample_id,
    no_utt_coded, no_utt_missing,
    total_words, mean_words_per_utt, sd_words_per_utt,
    min_words_per_utt, max_words_per_utt,
    speaking_time, speaking_minutes,
    total_words_per_min
    """
    wc_df = read_word_count_sample_summary(
        wc_samples_file=wc_samples_file,
        directories=[input_dir, output_dir],
    )

    speaking_time_df = read_speaking_time_table(
        speaking_time_file=speaking_time_file,
        speaking_time_field=speaking_time_field,
        directories=[input_dir, output_dir],
    )

    merged = merge_speaking_time(
        df=wc_df,
        speaking_time_df=speaking_time_df,
        how="left",
    )

    rated = add_rate_columns(
        merged,
        numerator_cols=["total_words"],
        speaking_minutes_col="speaking_minutes",
        suffix="_per_min",
        round_digits=3,
    )

    rated = finalize_word_count_rates_columns(rated)

    write_word_count_rates_output(
        rated,
        output_dir=output_dir,
        output_filename=DEFAULT_WC_RATES_FILE,
    )

    return rated
