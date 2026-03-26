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

DEFAULT_CU_SAMPLES_FILE = "cu_coding_by_sample_long.xlsx"
DEFAULT_CU_RATES_FILE = "cu_coding_rates.xlsx"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_cu_sample_summary(
    cu_samples_file: str | Path | None = None,
    directories=None,
) -> pd.DataFrame:
    """
    Read the canonical long-format CU sample summary table.

    Expected columns
    ----------------
    Required:
        - sample_id
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
        Exact file path, or filename/base pattern to search for.
        If None, defaults to DEFAULT_CU_SAMPLES_FILE.
    directories : Path | str | list[Path | str] | None
        Directories to search.

    Returns
    -------
    pd.DataFrame
        Canonical long-format CU sample summary.
    """
    cu_samples_file = cu_samples_file or DEFAULT_CU_SAMPLES_FILE

    candidate_path = Path(cu_samples_file)
    if candidate_path.exists():
        matches = [candidate_path]
    else:
        search_base = Path(cu_samples_file).stem
        matches = find_matching_files(
            directories=directories,
            search_base=search_base,
            search_ext=".xlsx",
        )

    if not matches:
        raise FileNotFoundError(
            f"No CU sample summary file found matching: {cu_samples_file}"
        )

    if len(matches) > 1:
        logger.warning(
            "Multiple CU sample summary files detected; "
            f"using first in list: {_rel(matches[0])}"
        )

    path = Path(matches[0])

    try:
        df = pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(f"Failed reading CU sample summary {_rel(path)}: {e}") from e

    required_cols = [
        "sample_id",
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

    logger.info(f"Loaded CU sample summary from {_rel(path)}")
    return df


def finalize_cu_rates_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the canonical CU rates output columns in a clean order.
    """
    preferred = [
        "sample_id",
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
        logger.info(f"Saved CU rates file: {_rel(out_path)}")
    except Exception as e:
        raise RuntimeError(f"Failed writing CU rates file {_rel(out_path)}: {e}") from e

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
    )

    speaking_time_df = read_speaking_time_table(
        speaking_time_file=speaking_time_file,
        speaking_time_field=speaking_time_field,
        directories=[input_dir, output_dir],
    )

    merged = merge_speaking_time(
        df=cu_df,
        speaking_time_df=speaking_time_df,
        how="left",
    )

    rated = add_rate_columns(
        merged,
        numerator_cols=["cu", "p_sv", "p_rel"],
        speaking_minutes_col="speaking_minutes",
        suffix="_per_min",
        round_digits=3,
    )

    rated = finalize_cu_rates_columns(rated)

    write_cu_rates_output(
        rated,
        output_dir=output_dir,
        output_filename=DEFAULT_CU_RATES_FILE,
    )

    return rated
