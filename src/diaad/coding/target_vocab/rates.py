from __future__ import annotations

from pathlib import Path

import pandas as pd

from psair.core.logger import get_rel_path, logger

from diaad.coding.utils.rates import add_rate_columns, present_cols, validate_columns


DEFAULT_TARGET_VOCAB_RATES_FILE = "target_vocab_rates.xlsx"
TARGET_VOCAB_SUBDIR = "target_vocab"
TARGET_VOCAB_SUMMARY_SHEET = "summary"


def find_target_vocab_analysis_files(
    directories,
) -> list[Path]:
    """
    Locate target vocabulary analysis workbooks.
    """
    matches: list[Path] = []
    seen: set[Path] = set()

    for directory in directories or []:
        if directory is None:
            continue
        root = Path(directory)
        if not root.exists():
            continue

        for path in root.rglob("target_vocab_data_*.xlsx"):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                matches.append(path)

    return sorted(matches)


def read_target_vocab_summary(
    directories,
    sheet_name: str = TARGET_VOCAB_SUMMARY_SHEET,
) -> pd.DataFrame:
    """
    Read and combine target vocabulary summary sheets from analysis workbooks.
    """
    matches = find_target_vocab_analysis_files(directories)
    if not matches:
        raise FileNotFoundError("No target vocabulary analysis workbook found matching 'target_vocab_data_*.xlsx'.")

    frames: list[pd.DataFrame] = []
    for path in matches:
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed reading sheet {sheet_name!r} from {get_rel_path(path)}: {e}"
            ) from e

        validate_columns(
            df,
            required_cols=["sample_id", "speaking_time"],
            df_name="Target vocabulary summary",
        )
        df = df.copy()
        df["source_file"] = path.name
        frames.append(df)
        logger.info(f"Loaded target vocabulary summary from {get_rel_path(path)}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _ensure_speaking_minutes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["speaking_time"] = pd.to_numeric(out["speaking_time"], errors="coerce")
    out["speaking_minutes"] = out["speaking_time"] / 60.0
    return out


def infer_target_vocab_rate_numerators(df: pd.DataFrame) -> list[str]:
    """
    Infer count-like target vocabulary summary columns that should be rated per minute.
    """
    excluded = {
        "sample_id",
        "narrative",
        "speaking_time",
        "speaking_minutes",
        "source_file",
        "lexicon_coverage",
        "accuracy_pwa_percentile",
        "accuracy_control_percentile",
        "efficiency_pwa_percentile",
        "efficiency_control_percentile",
        "core_tokens_per_min",
    }
    numerator_cols: list[str] = []

    for col in df.columns:
        if col in excluded or col.endswith("_per_min"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        numerator_cols.append(col)

    return numerator_cols


def finalize_target_vocab_rates_columns(
    df: pd.DataFrame,
    numerator_cols: list[str],
) -> pd.DataFrame:
    """
    Keep the canonical target vocabulary rates output columns in a clean order.
    """
    preferred = [
        "sample_id",
        "narrative",
        "source_file",
        *numerator_cols,
        "speaking_time",
        "speaking_minutes",
        "core_tokens_per_min",
        *[f"{col}_per_min" for col in numerator_cols if f"{col}_per_min" in df.columns],
        "accuracy_pwa_percentile",
        "accuracy_control_percentile",
        "efficiency_pwa_percentile",
        "efficiency_control_percentile",
    ]
    return df[present_cols(df, preferred)].copy()


def write_target_vocab_rates_output(
    df: pd.DataFrame,
    output_dir,
    output_filename: str = DEFAULT_TARGET_VOCAB_RATES_FILE,
) -> Path:
    """
    Write the target vocabulary rates table to disk.
    """
    out_dir = Path(output_dir) / TARGET_VOCAB_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_filename

    try:
        df.to_excel(out_path, index=False)
        logger.info(f"Saved target vocabulary rates file: {get_rel_path(out_path)}")
    except Exception as e:
        raise RuntimeError(
            f"Failed writing target vocabulary rates file {get_rel_path(out_path)}: {e}"
        ) from e

    return out_path


def calculate_target_vocab_rates(
    input_dir,
    output_dir,
):
    """
    Compute target vocabulary per-minute rates from the analysis summary workbook.
    """
    summary_df = read_target_vocab_summary(directories=[input_dir, output_dir])
    summary_df = _ensure_speaking_minutes(summary_df)

    numerator_cols = infer_target_vocab_rate_numerators(summary_df)
    rated = add_rate_columns(
        summary_df,
        numerator_cols=numerator_cols,
        speaking_minutes_col="speaking_minutes",
        suffix="_per_min",
        round_digits=3,
    )
    rated = finalize_target_vocab_rates_columns(rated, numerator_cols=numerator_cols)

    write_target_vocab_rates_output(
        rated,
        output_dir=output_dir,
        output_filename=DEFAULT_TARGET_VOCAB_RATES_FILE,
    )

    return rated
