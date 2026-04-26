from __future__ import annotations

from pathlib import Path

import pandas as pd

from psair.core.logger import logger, get_rel_path
from diaad.coding.utils.rates import (
    add_rate_columns,
    merge_speaking_time,
    present_cols,
    read_speaking_time_table,
    validate_columns,
)


DEFAULT_POWERS_RATES_FILE = "powers_coding_rates.xlsx"
POWERS_ANALYSIS_SUBDIR = "powers_coding_analysis"


def find_powers_analysis_files(
    directories,
) -> list[Path]:
    """
    Locate POWERS analysis workbooks.
    """
    matches: list[Path] = []
    seen: set[Path] = set()

    for directory in directories or []:
        if directory is None:
            continue
        root = Path(directory)
        if not root.exists():
            continue

        for path in root.rglob("*powers*analysis*.xlsx"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                matches.append(path)

    return sorted(matches)


def read_powers_dialog_summaries(
    directories,
    sheet_name: str = "Dialogs",
) -> pd.DataFrame:
    """
    Read and combine sample/dialog-level POWERS analysis sheets.
    """
    matches = find_powers_analysis_files(directories)
    if not matches:
        raise FileNotFoundError("No POWERS analysis workbook found matching '*powers*analysis*.xlsx'.")

    frames: list[pd.DataFrame] = []

    for path in matches:
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed reading sheet {sheet_name!r} from {get_rel_path(path)}: {e}"
            ) from e

        validate_columns(df, ["sample_id"], df_name=f"POWERS {sheet_name} summary")

        df = df.copy()
        df["source_file"] = path.name
        frames.append(df)
        logger.info(f"Loaded POWERS {sheet_name} summary from {get_rel_path(path)}")

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if combined["sample_id"].duplicated().any():
        logger.warning(
            "Combined POWERS dialog summaries contain duplicate sample_id values; "
            "retaining all rows in the rates output."
        )

    return combined


def infer_powers_rate_numerators(df: pd.DataFrame) -> list[str]:
    """
    Infer count-like POWERS columns that should be expressed per minute.
    """
    excluded = {"speaking_time", "speaking_minutes"}
    numerator_cols: list[str] = []

    for col in df.columns:
        if col in {"sample_id", "source_file"} or col in excluded:
            continue
        if col.startswith("prop_") or col.startswith("ratio_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        numerator_cols.append(col)

    return numerator_cols


def finalize_powers_rates_columns(
    df: pd.DataFrame,
    numerator_cols: list[str],
) -> pd.DataFrame:
    """
    Keep the canonical POWERS rates output columns in a clean order.
    """
    rate_cols = [f"{col}_per_min" for col in numerator_cols if f"{col}_per_min" in df.columns]
    preferred = [
        "sample_id",
        "source_file",
        *numerator_cols,
        "speaking_time",
        "speaking_minutes",
        *rate_cols,
    ]
    return df[present_cols(df, preferred)].copy()


def write_powers_rates_output(
    df: pd.DataFrame,
    output_dir,
    output_filename: str = DEFAULT_POWERS_RATES_FILE,
) -> Path:
    """
    Write the POWERS rates table to disk.
    """
    out_dir = Path(output_dir) / POWERS_ANALYSIS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / output_filename

    try:
        df.to_excel(out_path, index=False)
        logger.info(f"Saved POWERS rates file: {get_rel_path(out_path)}")
    except Exception as e:
        raise RuntimeError(
            f"Failed writing POWERS rates file {get_rel_path(out_path)}: {e}"
        ) from e

    return out_path


def calculate_powers_rates(
    input_dir,
    output_dir,
    speaking_time_file: str | Path | None = None,
    speaking_time_field: str = "speaking_time",
):
    """
    Compute POWERS sample/dialog-level measures per minute.
    """
    powers_df = read_powers_dialog_summaries(directories=[input_dir, output_dir])

    speaking_time_df = read_speaking_time_table(
        speaking_time_file=speaking_time_file,
        speaking_time_field=speaking_time_field,
        directories=[input_dir, output_dir],
    )

    merged = merge_speaking_time(
        df=powers_df,
        speaking_time_df=speaking_time_df,
        how="left",
    )

    numerator_cols = infer_powers_rate_numerators(merged)
    rated = add_rate_columns(
        merged,
        numerator_cols=numerator_cols,
        speaking_minutes_col="speaking_minutes",
        suffix="_per_min",
        round_digits=3,
    )

    rated = finalize_powers_rates_columns(rated, numerator_cols=numerator_cols)

    write_powers_rates_output(
        rated,
        output_dir=output_dir,
        output_filename=DEFAULT_POWERS_RATES_FILE,
    )

    return rated
