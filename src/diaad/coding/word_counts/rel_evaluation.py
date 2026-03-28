import numpy as np
import pandas as pd
from pathlib import Path

from diaad.core.logger import logger, _rel
from diaad.io.discovery import find_matching_files
from src.diaad.coding.utils.rel_eval_utils import percent_difference, calculate_icc_from_pingouin


def agreement(row):
    """
    Agreement = 1 if:
      - absolute difference <= 1 word, or
      - percent similarity >= 85%
    """
    abs_diff = abs(row["word_count_org"] - row["word_count_rel"])
    if abs_diff <= 1:
        return 1

    perc_diff = percent_difference(row["word_count_org"], row["word_count_rel"])
    perc_sim = 100 - perc_diff
    return 1 if perc_sim >= 85 else 0


def _write_word_rel_outputs(wc_merged, out_dir, rel_name):
    """Write Excel results and plain-text report for word-count reliability."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if wc_merged.empty:
        logger.warning(f"No merged rows found for {_rel(out_dir)}; skipping output.")
        return

    # Excel results
    results_path = out_dir / "word_count_reliability_results.xlsx"
    try:
        wc_merged.to_excel(results_path, index=False)
        logger.info(f"Wrote word-count reliability results: {_rel(results_path)}")
    except Exception as e:
        logger.error(f"Failed writing reliability results {_rel(results_path)}: {e}")
        return

    # ICC
    icc_value = calculate_icc_from_pingouin(wc_merged)
    logger.info(f"Calculated ICC(2,1) for {rel_name}: {icc_value}")

    # Agreement summary
    total = len(wc_merged)
    n_agree = int(np.nansum(wc_merged["agmt"])) if "agmt" in wc_merged.columns else 0
    perc_agree = round((n_agree / total) * 100, 1) if total > 0 else np.nan

    mean_abs_diff = round(float(wc_merged["abs_diff"].abs().mean()), 3) if total > 0 else np.nan
    mean_perc_diff = round(float(wc_merged["perc_diff"].mean()), 3) if total > 0 else np.nan
    mean_perc_sim = round(float(wc_merged["perc_sim"].mean()), 3) if total > 0 else np.nan

    report_path = out_dir / "word_count_reliability_report.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Word Count Reliability Report\n\n")
            f.write(f"Source reliability file: {rel_name}\n\n")

            if total > 0:
                f.write(f"Paired utterances: {total}\n")
                f.write(f"Utterances in agreement: {n_agree}/{total} ({perc_agree}%)\n")
                f.write(f"Mean absolute difference: {mean_abs_diff}\n")
                f.write(f"Mean percent difference: {mean_perc_diff}%\n")
                f.write(f"Mean percent similarity: {mean_perc_sim}%\n")
            else:
                f.write("No valid utterances available for agreement calculation.\n")

            f.write(f"ICC(2,1): {icc_value}\n")

        logger.info(f"Successfully wrote reliability report to {_rel(report_path)}")
    except Exception as e:
        logger.error(f"Failed writing reliability report {_rel(report_path)}: {e}")


def evaluate_word_count_reliability(input_dir, output_dir):
    """
    Evaluate word-count reliability by comparing primary and reliability files.

    Behavior
    --------
    - Finds word_counting.xlsx and word_count_reliability.xlsx.
    - If multiple matches are found for either search base, warns and uses only
      the first returned file.
    - Merges on sample_id and utterance_id.
    - Computes:
        abs_diff, perc_diff, perc_sim, agmt
    - Calculates ICC(2,1) with pingouin.intraclass_corr.
    - Writes merged table and plain-text report under:
        <output_dir>/word_count_reliability/

    Notes
    -----
    Agreement = 1 if |diff| <= 1 or percent similarity >= 85%.
    """
    word_rel_dir = Path(output_dir) / "word_count_reliability"
    word_rel_dir.mkdir(parents=True, exist_ok=True)

    coding_files = find_matching_files(
        directories=[input_dir, output_dir],
        search_base="word_counting",
    )
    rel_files = find_matching_files(
        directories=[input_dir, output_dir],
        search_base="word_count_reliability",
    )

    if not coding_files:
        logger.error("No word_counting.xlsx file found.")
        return

    if not rel_files:
        logger.error("No word_count_reliability.xlsx file found.")
        return

    if len(coding_files) > 1:
        logger.warning(
            "Multiple word-count coding files detected. "
            f"Using only the first returned file: {_rel(coding_files[0])}"
        )

    if len(rel_files) > 1:
        logger.warning(
            "Multiple word-count reliability files detected. "
            f"Using only the first returned file: {_rel(rel_files[0])}"
        )

    cod = coding_files[0]
    rel = rel_files[0]

    try:
        wc_df = pd.read_excel(cod)
        wc_rel_df = pd.read_excel(rel)
        logger.info(f"Processing pair: {_rel(cod)} + {_rel(rel)}")
    except Exception as e:
        logger.error(f"Failed reading {_rel(cod)} or {_rel(rel)}: {e}")
        return

    try:
        # Keep only the columns needed from the reliability sheet.
        rel_keep = ["sample_id", "utterance_id", "word_count"]
        missing_rel = [col for col in rel_keep if col not in wc_rel_df.columns]
        if missing_rel:
            raise KeyError(f"Missing required reliability columns: {missing_rel}")

        wc_rel_df = wc_rel_df[rel_keep].copy()

        # Treat NA-like values as missing for reliability evaluation.
        wc_rel_df["word_count"] = pd.to_numeric(wc_rel_df["word_count"], errors="coerce")
        wc_df["word_count"] = pd.to_numeric(wc_df["word_count"], errors="coerce")

        wc_rel_df = wc_rel_df.dropna(subset=["word_count"])
        wc_df = wc_df.dropna(subset=["word_count"])

        wc_merged = pd.merge(
            wc_df,
            wc_rel_df,
            on=["sample_id", "utterance_id"],
            how="inner",
            suffixes=("_org", "_rel"),
        )

        if len(wc_rel_df) != len(wc_merged):
            logger.warning(f"Row mismatch after merge on {_rel(rel)}")

    except Exception as e:
        logger.error(f"Failed merging {_rel(cod)} and {_rel(rel)}: {e}")
        return

    if wc_merged.empty:
        logger.warning("Merged word-count reliability dataframe is empty.")
        return

    wc_merged["abs_diff"] = (wc_merged["word_count_org"] - wc_merged["word_count_rel"]).abs()
    wc_merged["perc_diff"] = wc_merged.apply(
        lambda r: percent_difference(r["word_count_org"], r["word_count_rel"]),
        axis=1,
    )
    wc_merged["perc_sim"] = 100 - wc_merged["perc_diff"]
    wc_merged["agmt"] = wc_merged.apply(agreement, axis=1)

    _write_word_rel_outputs(
        wc_merged=wc_merged,
        out_dir=word_rel_dir,
        rel_name=rel.name,
    )
