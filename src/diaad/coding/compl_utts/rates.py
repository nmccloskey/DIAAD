from __future__ import annotations

from pathlib import Path
import pandas as pd

from psair.core.logger import logger, get_rel_path
from diaad.metadata.discovery import find_one_matching_file
from diaad.coding.utils.rates import (
    read_speaking_time_table,
    merge_speaking_time,
    add_rate_columns,
    validate_columns,
    present_cols,
)


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_CU_SAMPLES_FILE = "cu_coding_by_sample_long.xlsx"
DEFAULT_CU_RATES_FILE = "cu_coding_rates.xlsx"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_cu_sample_summary(
    cu_samples_file: str | Path | None = None,
    directories=None,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Read the canonical long-format CU sample summary table.

    Expected columns
    ----------------
    Required:
        - sample_id (or configured sample_id_field)
        - coder
        - paradigm
        - sv_col
        - rel_col
        - cu_col
        - p_sv
        - p_rel
        - cu

    Parameters
    ----------
    cu_samples_file : str | Path | None
        Exact file path or filename to search for.
        If None, defaults to DEFAULT_CU_SAMPLES_FILE.
    directories : Path | str | list[Path | str] | None
        Directories to search.

    Returns
    -------
    pd.DataFrame
        Canonical long-format CU sample summary.
    """
    cu_samples_file = cu_samples_file or DEFAULT_CU_SAMPLES_FILE

    path = find_one_matching_file(
        directories=directories,
        filename=cu_samples_file,
        label="CU sample summary file",
    )

    try:
        df = pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(f"Failed reading CU sample summary {get_rel_path(path)}: {e}") from e

    required_cols = [
        sample_id_field,
        "coder",
        "paradigm",
        "sv_col",
        "rel_col",
        "cu_col",
        "p_sv",
        "p_rel",
        "cu",
    ]
    validate_columns(df, required_cols, df_name="CU sample summary")

    logger.info(f"Loaded CU sample summary from {get_rel_path(path)}")
    return df


def finalize_cu_rates_columns(
    df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Keep the canonical CU rates output columns in a clean order.
    """
    preferred = [
        sample_id_field,
        "coder",
        "paradigm",
        "sv_col",
        "rel_col",
        "cu_col",
        "speaking_time",
        "speaking_minutes",
        "cu_per_min",
        "p_sv_per_min",
        "p_rel_per_min",
    ]
    return df[present_cols(df, preferred)].copy()


def write_cu_rates_output(
    df: pd.DataFrame,
    output_dir,
    output_filename: str = DEFAULT_CU_RATES_FILE,
) -> Path:
    """
    Write the CU rates table to disk.
    """
    out_dir = Path(output_dir) / "cu_coding_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / output_filename

    try:
        df.to_excel(out_path, index=False)
        logger.info(f"Saved CU rates file: {get_rel_path(out_path)}")
    except Exception as e:
        raise RuntimeError(f"Failed writing CU rates file {get_rel_path(out_path)}: {e}") from e

    return out_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def calculate_cu_rates(
    input_dir,
    output_dir,
    cu_samples_file: str | Path | None = None,
    speaking_time_file: str | Path | None = None,
    speaking_time_field: str = "speaking_time",
    sample_id_field: str = "sample_id",
):
    """
    Compute CU, SV, and REL rates per minute from the canonical long-format
    CU sample summary.

    Inputs
    ------
    - CU sample summary file (long format), typically:
        cu_coding_by_sample_long.xlsx
    - Speaking time spreadsheet with:
        sample_id
        speaking_time   (in seconds)

    Behavior
    --------
    - Reads the long CU sample summary.
    - Reads and standardizes the speaking time table.
    - Merges speaking times onto CU sample rows by sample_id.
    - Computes:
        * cu_per_min
        * p_sv_per_min
        * p_rel_per_min
    - Writes a compact CU rates table.

    Output columns
    --------------
    sample_id, coder, paradigm, sv_col, rel_col, cu_col,
    speaking_time, speaking_minutes,
    cu_per_min, p_sv_per_min, p_rel_per_min
    """
    cu_df = read_cu_sample_summary(
        cu_samples_file=cu_samples_file,
        directories=[input_dir, output_dir],
        sample_id_field=sample_id_field,
    )

    speaking_time_df = read_speaking_time_table(
        speaking_time_file=speaking_time_file,
        speaking_time_field=speaking_time_field,
        directories=[input_dir, output_dir],
        sample_id_field=sample_id_field,
    )

    merged = merge_speaking_time(
        df=cu_df,
        speaking_time_df=speaking_time_df,
        how="left",
        sample_id_field=sample_id_field,
    )

    rated = add_rate_columns(
        merged,
        numerator_cols=["cu", "p_sv", "p_rel"],
        speaking_minutes_col="speaking_minutes",
        suffix="_per_min",
        round_digits=3,
    )

    rated = finalize_cu_rates_columns(rated, sample_id_field=sample_id_field)

    write_cu_rates_output(
        rated,
        output_dir=output_dir,
        output_filename=DEFAULT_CU_RATES_FILE,
    )

    return rated
