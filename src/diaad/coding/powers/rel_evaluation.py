from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from psair.core.logger import logger, get_rel_path
from psair.metadata.discovery import find_matching_files
from diaad.coding.utils.rel_eval_utils import (
    percent_difference,
    calculate_icc_from_pingouin,
)


CONTINUOUS_COLS = [
    "speech_units",
    "content_words",
    "num_nouns",
    "filled_pauses",
    "circumlocutions",
    "sem_paras",
    "phon_errs",
    "neologisms",
    "lg_pauses",
]

CATEGORICAL_COLS = [
    "turn_type",
    "collab_repair",
]

JOIN_COLS = [
    "sample_id",
    "utterance_id",
]


LOW_VARIANCE_POOLED_VAR_THRESHOLD = 0.05
LOW_VARIANCE_MAX_VALUE_PROP_THRESHOLD = 0.95
SPARSE_NONZERO_EITHER_PCT_THRESHOLD = 5.0
EXACT_AGREEMENT_TOLERANCE = 0.0
NEAR_AGREEMENT_TOLERANCE = 1.0


def _get_first_match(files: list[Path], label: str) -> Path | None:
    """Return the first discovered file, warning if multiple were found."""
    if not files:
        logger.error(f"No {label} file found.")
        return None

    if len(files) > 1:
        logger.warning(
            f"Multiple {label} files detected. "
            f"Using only the first returned file: {get_rel_path(files[0])}"
        )

    return files[0]


def _read_powers_pair(org_file: Path, rel_file: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Read original and reliability POWERS files."""
    try:
        org_df = pd.read_excel(org_file)
        rel_df = pd.read_excel(rel_file)
        logger.info(f"Processing pair: {get_rel_path(org_file)} + {get_rel_path(rel_file)}")
        return org_df, rel_df
    except Exception as e:
        logger.error(f"Failed reading {get_rel_path(org_file)} or {get_rel_path(rel_file)}: {e}")
        return None, None


def _required_rel_cols() -> list[str]:
    """Return the minimum set of columns needed from the reliability file."""
    return JOIN_COLS + CONTINUOUS_COLS + CATEGORICAL_COLS


def _prepare_rel_subset(rel_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only evaluation-relevant columns from the reliability file."""
    keep = [col for col in _required_rel_cols() if col in rel_df.columns]
    missing = [col for col in JOIN_COLS if col not in rel_df.columns]
    if missing:
        raise KeyError(f"Missing required reliability join columns: {missing}")
    return rel_df.loc[:, keep].copy()


def _coerce_continuous_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Coerce continuous POWERS fields to numeric where present."""
    for col in CONTINUOUS_COLS:
        full_col = f"{col}{suffix}"
        if full_col in df.columns:
            df[full_col] = pd.to_numeric(df[full_col], errors="coerce")
    return df


def _merge_powers_reliability(org_df: pd.DataFrame, rel_df: pd.DataFrame) -> pd.DataFrame:
    """Merge original and reliability POWERS data on sample and utterance IDs."""
    rel_sub = _prepare_rel_subset(rel_df)

    # Diagnose duplicate keys before merge because duplicates can silently inflate rows.
    org_dupes = org_df.duplicated(subset=JOIN_COLS, keep=False).sum()
    rel_dupes = rel_sub.duplicated(subset=JOIN_COLS, keep=False).sum()

    if org_dupes:
        logger.warning(
            f"Original POWERS file contains {org_dupes} rows with duplicated join keys: {JOIN_COLS}."
        )
    if rel_dupes:
        logger.warning(
            f"Reliability POWERS file contains {rel_dupes} rows with duplicated join keys: {JOIN_COLS}."
        )

    merged = pd.merge(
        org_df,
        rel_sub,
        on=JOIN_COLS,
        how="inner",
        suffixes=("_org", "_rel"),
    )

    if len(rel_sub) != len(merged):
        logger.warning(
            "Row mismatch after merge on POWERS reliability file. "
            f"Reliability rows={len(rel_sub)}, merged rows={len(merged)}."
        )

    merged = merged.reset_index(drop=True)

    # Unique target for ICC. Do not use utterance_id alone because utterance IDs may repeat across samples.
    merged.insert(0, "reliability_id", np.arange(1, len(merged) + 1))

    merged = _coerce_continuous_cols(merged, "_org")
    merged = _coerce_continuous_cols(merged, "_rel")
    return merged


def _safe_pct(numerator: int | float, denominator: int | float) -> float:
    """Return percentage, guarding against division by zero."""
    if denominator == 0:
        return np.nan
    return round(float((numerator / denominator) * 100), 3)


def _max_value_prop(series: pd.Series) -> float:
    """Return the proportion of the most common non-missing value."""
    nonmissing = series.dropna()
    if nonmissing.empty:
        return np.nan
    return float(nonmissing.value_counts(normalize=True).iloc[0])


def _continuous_distribution_diagnostics(
    merged: pd.DataFrame,
    org_col: str,
    rel_col: str,
    paired: pd.DataFrame,
) -> dict:
    """Compute distribution diagnostics for a continuous/count metric."""
    total_rows = len(merged)

    org_missing = int(merged[org_col].isna().sum())
    rel_missing = int(merged[rel_col].isna().sum())
    paired_n = len(paired)
    rows_dropped_missing = total_rows - paired_n

    org_var = float(paired[org_col].var()) if paired_n > 1 else np.nan
    rel_var = float(paired[rel_col].var()) if paired_n > 1 else np.nan
    org_sd = float(paired[org_col].std()) if paired_n > 1 else np.nan
    rel_sd = float(paired[rel_col].std()) if paired_n > 1 else np.nan

    if not np.isnan(org_var) and not np.isnan(rel_var):
        pooled_var = float((org_var + rel_var) / 2)
        pooled_sd = float(np.sqrt(pooled_var))
    else:
        pooled_var = np.nan
        pooled_sd = np.nan

    exact_agreement = paired[org_col].eq(paired[rel_col])
    abs_diff = (paired[org_col] - paired[rel_col]).abs()

    both_zero = paired[org_col].eq(0) & paired[rel_col].eq(0)
    nonzero_either = paired[org_col].ne(0) | paired[rel_col].ne(0)

    org_max_prop = _max_value_prop(paired[org_col])
    rel_max_prop = _max_value_prop(paired[rel_col])
    max_value_prop = np.nanmax([org_max_prop, rel_max_prop])

    diagnostics = {
        "total_merged_rows": total_rows,
        "paired_utterances": paired_n,
        "org_missing": org_missing,
        "rel_missing": rel_missing,
        "rows_dropped_missing": rows_dropped_missing,
        "rows_dropped_missing_pct": _safe_pct(rows_dropped_missing, total_rows),
        "org_mean": round(float(paired[org_col].mean()), 4) if paired_n else np.nan,
        "rel_mean": round(float(paired[rel_col].mean()), 4) if paired_n else np.nan,
        "org_sd": round(org_sd, 4) if not np.isnan(org_sd) else np.nan,
        "rel_sd": round(rel_sd, 4) if not np.isnan(rel_sd) else np.nan,
        "org_var": round(org_var, 4) if not np.isnan(org_var) else np.nan,
        "rel_var": round(rel_var, 4) if not np.isnan(rel_var) else np.nan,
        "pooled_sd": round(pooled_sd, 4) if not np.isnan(pooled_sd) else np.nan,
        "pooled_var": round(pooled_var, 4) if not np.isnan(pooled_var) else np.nan,
        "org_unique_values": int(paired[org_col].nunique(dropna=True)),
        "rel_unique_values": int(paired[rel_col].nunique(dropna=True)),
        "max_value_prop_pct": round(max_value_prop * 100, 3) if not np.isnan(max_value_prop) else np.nan,
        "exact_agreement_pct": round(float(exact_agreement.mean() * 100), 3) if paired_n else np.nan,
        "within_1_count_pct": round(float(abs_diff.le(NEAR_AGREEMENT_TOLERANCE).mean() * 100), 3) if paired_n else np.nan,
        "both_zero_pct": round(float(both_zero.mean() * 100), 3) if paired_n else np.nan,
        "nonzero_either_pct": round(float(nonzero_either.mean() * 100), 3) if paired_n else np.nan,
    }

    return diagnostics


def _make_low_variance_warning(diagnostics: dict) -> str:
    """
    Return a warning string when ICC is likely unstable or misleading.

    ICC can be poor despite high agreement when the metric has very little
    between-target variance, is highly zero-inflated, or is dominated by one value.
    """
    warnings = []

    pooled_var = diagnostics.get("pooled_var", np.nan)
    max_value_prop_pct = diagnostics.get("max_value_prop_pct", np.nan)
    nonzero_either_pct = diagnostics.get("nonzero_either_pct", np.nan)
    org_unique = diagnostics.get("org_unique_values", np.nan)
    rel_unique = diagnostics.get("rel_unique_values", np.nan)

    if not np.isnan(pooled_var) and pooled_var < LOW_VARIANCE_POOLED_VAR_THRESHOLD:
        warnings.append("low pooled variance")

    if not np.isnan(max_value_prop_pct) and max_value_prop_pct >= LOW_VARIANCE_MAX_VALUE_PROP_THRESHOLD * 100:
        warnings.append("one value dominates distribution")

    if not np.isnan(nonzero_either_pct) and nonzero_either_pct < SPARSE_NONZERO_EITHER_PCT_THRESHOLD:
        warnings.append("sparse/zero-inflated count")

    if org_unique <= 1 or rel_unique <= 1:
        warnings.append("one coder has no score variation")

    if warnings:
        return "; ".join(warnings)

    return ""


def _compute_continuous_diffs(merged: pd.DataFrame) -> pd.DataFrame:
    """Add row-level difference columns for each continuous POWERS metric."""
    for col in CONTINUOUS_COLS:
        org_col = f"{col}_org"
        rel_col = f"{col}_rel"

        if org_col not in merged.columns or rel_col not in merged.columns:
            continue

        abs_col = f"{col}_abs_diff"
        perc_diff_col = f"{col}_perc_diff"
        perc_sim_col = f"{col}_perc_sim"

        valid = merged[[org_col, rel_col]].notna().all(axis=1)

        merged[abs_col] = np.nan
        merged.loc[valid, abs_col] = (merged.loc[valid, org_col] - merged.loc[valid, rel_col]).abs()

        merged[perc_diff_col] = np.nan
        merged.loc[valid, perc_diff_col] = merged.loc[valid].apply(
            lambda r: percent_difference(r[org_col], r[rel_col]),
            axis=1,
        )

        merged[perc_sim_col] = np.nan
        merged.loc[valid, perc_sim_col] = 100 - merged.loc[valid, perc_diff_col]

    return merged


def _compute_continuous_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize continuous reliability with ICC, agreement, and distribution diagnostics.

    ICC is retained, but sparse/low-variance metrics are flagged because ICC can be
    misleadingly low when most targets have identical or zero scores.
    """
    rows = []

    for col in CONTINUOUS_COLS:
        org_col = f"{col}_org"
        rel_col = f"{col}_rel"
        abs_col = f"{col}_abs_diff"
        perc_diff_col = f"{col}_perc_diff"
        perc_sim_col = f"{col}_perc_sim"

        if org_col not in merged.columns or rel_col not in merged.columns:
            continue

        paired = merged.dropna(subset=[org_col, rel_col]).copy()
        if paired.empty:
            continue

        diagnostics = _continuous_distribution_diagnostics(
            merged=merged,
            org_col=org_col,
            rel_col=rel_col,
            paired=paired,
        )

        low_variance_warning = _make_low_variance_warning(diagnostics)

        icc_value = calculate_icc_from_pingouin(
            df=paired,
            target_col="reliability_id",
            col_org=org_col,
            col_rel=rel_col,
            rater_labels=("org", "rel"),
        )

        rows.append(
            {
                "metric": col,
                "paired_utterances": len(paired),

                # Difference / similarity
                "mean_abs_diff": round(float(paired[abs_col].mean()), 3) if abs_col in paired.columns else np.nan,
                "mean_perc_diff": round(float(paired[perc_diff_col].mean()), 3) if perc_diff_col in paired.columns else np.nan,
                "mean_perc_sim": round(float(paired[perc_sim_col].mean()), 3) if perc_sim_col in paired.columns else np.nan,

                # Agreement diagnostics
                "exact_agreement_pct": diagnostics["exact_agreement_pct"],
                "within_1_count_pct": diagnostics["within_1_count_pct"],

                # ICC
                "ICC2": icc_value,
                "ICC_warning": low_variance_warning,

                # Missingness diagnostics
                "total_merged_rows": diagnostics["total_merged_rows"],
                "org_missing": diagnostics["org_missing"],
                "rel_missing": diagnostics["rel_missing"],
                "rows_dropped_missing": diagnostics["rows_dropped_missing"],
                "rows_dropped_missing_pct": diagnostics["rows_dropped_missing_pct"],

                # Distribution diagnostics
                "org_mean": diagnostics["org_mean"],
                "rel_mean": diagnostics["rel_mean"],
                "org_sd": diagnostics["org_sd"],
                "rel_sd": diagnostics["rel_sd"],
                "org_var": diagnostics["org_var"],
                "rel_var": diagnostics["rel_var"],
                "pooled_sd": diagnostics["pooled_sd"],
                "pooled_var": diagnostics["pooled_var"],
                "org_unique_values": diagnostics["org_unique_values"],
                "rel_unique_values": diagnostics["rel_unique_values"],
                "max_value_prop_pct": diagnostics["max_value_prop_pct"],
                "both_zero_pct": diagnostics["both_zero_pct"],
                "nonzero_either_pct": diagnostics["nonzero_either_pct"],
            }
        )

    return pd.DataFrame(rows)


def _compute_categorical_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Summarize categorical reliability with percent agreement and kappa."""
    rows = []

    for col in CATEGORICAL_COLS:
        org_col = f"{col}_org"
        rel_col = f"{col}_rel"

        if org_col not in merged.columns or rel_col not in merged.columns:
            continue

        if col == "collab_repair":
            y1 = (~merged[org_col].replace(0, np.nan).isna()).astype(int)
            y2 = (~merged[rel_col].replace(0, np.nan).isna()).astype(int)
        else:
            y1 = merged[org_col].fillna("MISSING").astype(str)
            y2 = merged[rel_col].fillna("MISSING").astype(str)

        try:
            agreement = round(float((y1 == y2).mean() * 100), 1)
        except Exception:
            agreement = np.nan

        try:
            if len(np.unique(y1)) <= 1 and len(np.unique(y2)) <= 1:
                kappa = np.nan
            else:
                kappa = round(float(cohen_kappa_score(y1, y2)), 4)
        except Exception:
            kappa = np.nan

        rows.append(
            {
                "metric": col,
                "paired_utterances": len(merged),
                "percent_agreement": agreement,
                "kappa": kappa,
            }
        )

    return pd.DataFrame(rows)


def _write_powers_rel_outputs(
    merged: pd.DataFrame,
    cont_summary: pd.DataFrame,
    cat_summary: pd.DataFrame,
    out_dir: Path,
    rel_name: str,
) -> None:
    """Write merged results, summaries, and plain-text report."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if merged.empty:
        logger.warning(f"No merged rows found for {get_rel_path(out_dir)}; skipping output.")
        return

    results_path = out_dir / "powers_reliability_results.xlsx"
    try:
        with pd.ExcelWriter(results_path, engine="xlsxwriter") as writer:
            merged.to_excel(writer, sheet_name="merged", index=False)
            if not cont_summary.empty:
                cont_summary.to_excel(writer, sheet_name="continuous_summary", index=False)
            if not cat_summary.empty:
                cat_summary.to_excel(writer, sheet_name="categorical_summary", index=False)
        logger.info(f"Wrote POWERS reliability results: {get_rel_path(results_path)}")
    except Exception as e:
        logger.error(f"Failed writing reliability results {get_rel_path(results_path)}: {e}")
        return

    report_path = out_dir / "powers_reliability_report.txt"
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("POWERS Reliability Report\n\n")
            f.write(f"Source reliability file: {rel_name}\n\n")
            f.write(f"Paired utterances: {len(merged)}\n\n")

            f.write(
                "Note: ICC(2,1) is variance-sensitive. Sparse or low-variance count metrics "
                "may show low ICC values despite high exact agreement or small absolute differences. "
                "For flagged metrics, interpret ICC alongside agreement and distribution diagnostics.\n\n"
            )

            f.write("Continuous metrics\n")
            f.write("------------------\n")
            if cont_summary.empty:
                f.write("No continuous metrics available.\n\n")
            else:
                for _, row in cont_summary.iterrows():
                    warning = row.get("ICC_warning", "")
                    warning_text = f", ICC_warning={warning}" if isinstance(warning, str) and warning else ""

                    f.write(
                        f"{row['metric']}: "
                        f"n={row['paired_utterances']}, "
                        f"mean_abs_diff={row['mean_abs_diff']}, "
                        f"mean_perc_diff={row['mean_perc_diff']}%, "
                        f"mean_perc_sim={row['mean_perc_sim']}%, "
                        f"exact_agreement={row.get('exact_agreement_pct', np.nan)}%, "
                        f"within_1_count={row.get('within_1_count_pct', np.nan)}%, "
                        f"ICC(2,1)={row['ICC2']}"
                        f"{warning_text}\n"
                    )

                    if isinstance(warning, str) and warning:
                        f.write(
                            f"    Distribution diagnostics: "
                            f"pooled_var={row.get('pooled_var', np.nan)}, "
                            f"pooled_sd={row.get('pooled_sd', np.nan)}, "
                            f"max_value_prop={row.get('max_value_prop_pct', np.nan)}%, "
                            f"both_zero={row.get('both_zero_pct', np.nan)}%, "
                            f"nonzero_either={row.get('nonzero_either_pct', np.nan)}%, "
                            f"rows_dropped_missing={row.get('rows_dropped_missing', np.nan)} "
                            f"({row.get('rows_dropped_missing_pct', np.nan)}%)\n"
                        )
                f.write("\n")

            f.write("Categorical metrics\n")
            f.write("-------------------\n")
            if cat_summary.empty:
                f.write("No categorical metrics available.\n")
            else:
                for _, row in cat_summary.iterrows():
                    f.write(
                        f"{row['metric']}: "
                        f"n={row['paired_utterances']}, "
                        f"agreement={row['percent_agreement']}%, "
                        f"kappa={row['kappa']}\n"
                    )

        logger.info(f"Successfully wrote reliability report to {get_rel_path(report_path)}")
    except Exception as e:
        logger.error(f"Failed writing reliability report {get_rel_path(report_path)}: {e}")


def evaluate_powers_reliability(
    input_dir,
    output_dir,
    powers_coding_file="powers_coding.xlsx",
    powers_reliability_file="powers_reliability_coding.xlsx",
):
    """
    Evaluate POWERS reliability by comparing coding and reliability files.

    Uses the first discovered powers_coding and powers_reliability_coding
    files, merges on sample_id and utterance_id, computes row-level absolute
    and percent differences for continuous measures, computes ICC(2,1) where
    applicable, and computes agreement/kappa for categorical measures.
    """
    out_dir = Path(output_dir) / "powers_reliability"
    out_dir.mkdir(parents=True, exist_ok=True)

    coding_files = find_matching_files(
        directories=[input_dir, output_dir],
        search_base=Path(powers_coding_file).stem,
    )
    rel_files = find_matching_files(
        directories=[input_dir, output_dir],
        search_base=Path(powers_reliability_file).stem,
    )

    org_file = _get_first_match(coding_files, "POWERS coding")
    rel_file = _get_first_match(rel_files, "POWERS reliability")
    if org_file is None or rel_file is None:
        return

    org_df, rel_df = _read_powers_pair(org_file, rel_file)
    if org_df is None or rel_df is None:
        return

    try:
        merged = _merge_powers_reliability(org_df, rel_df)
    except Exception as e:
        logger.error(f"Failed merging {get_rel_path(org_file)} and {get_rel_path(rel_file)}: {e}")
        return

    if merged.empty:
        logger.warning("Merged POWERS reliability dataframe is empty.")
        return

    merged = _compute_continuous_diffs(merged)
    cont_summary = _compute_continuous_summary(merged)
    cat_summary = _compute_categorical_summary(merged)

    _write_powers_rel_outputs(
        merged=merged,
        cont_summary=cont_summary,
        cat_summary=cat_summary,
        out_dir=out_dir,
        rel_name=rel_file.name,
    )
