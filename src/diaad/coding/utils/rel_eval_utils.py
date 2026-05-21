from __future__ import annotations

import numpy as np
import pandas as pd
from pingouin import intraclass_corr

from psair.core.logger import logger


def percent_difference(value1, value2):
    """
    Calculate percentage difference between two values.

    Returns 0 when both values are 0 and 100 when one value is 0 and the
    other is nonzero.
    """
    if value1 == 0 and value2 == 0:
        return 0

    if value1 == 0 or value2 == 0:
        logger.warning("One of the values is zero, returning 100%.")
        return 100

    diff = abs(value1 - value2)
    avg = (value1 + value2) / 2
    return round((diff / avg) * 100, 2)


def calculate_icc_from_pingouin(
    df: pd.DataFrame,
    target_col: str,
    col_org: str,
    col_rel: str,
    rater_labels: tuple[str, str] = ("org", "rel"),
) -> float:
    """
    Calculate ICC(2,1) from two paired score columns.

    Returns np.nan when data are insufficient or ICC2 is unavailable.
    """
    if df is None or df.empty:
        logger.warning("ICC calculation skipped: dataframe is empty.")
        return np.nan

    needed = [target_col, col_org, col_rel]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        logger.warning(f"ICC calculation skipped: missing columns {missing}.")
        return np.nan

    icc_df = df[needed].dropna(subset=needed).copy()
    if len(icc_df) < 2:
        logger.warning("ICC calculation skipped: fewer than 2 paired targets.")
        return np.nan

    long_df = pd.concat(
        [
            icc_df[[target_col, col_org]].rename(
                columns={target_col: "targets", col_org: "scores"}
            ).assign(raters=rater_labels[0]),
            icc_df[[target_col, col_rel]].rename(
                columns={target_col: "targets", col_rel: "scores"}
            ).assign(raters=rater_labels[1]),
        ],
        ignore_index=True,
    )

    try:
        icc_table = intraclass_corr(
            data=long_df,
            targets="targets",
            raters="raters",
            ratings="scores",
        )
        icc_row = icc_table.loc[icc_table["Type"] == "ICC2"]
        if icc_row.empty:
            logger.warning("Pingouin ICC table did not contain ICC2.")
            return np.nan

        return round(float(icc_row["ICC"].iloc[0]), 4)

    except Exception as e:
        logger.error(f"ICC calculation failed: {e}")
        return np.nan


def _unique_count(df: pd.DataFrame, cols: list[str]) -> int:
    """Return unique non-null key count for columns that exist in df."""
    if df is None or df.empty:
        return 0
    missing = [col for col in cols if col not in df.columns]
    if missing:
        return 0
    return int(df[cols].dropna(how="any").drop_duplicates().shape[0])


def coverage_summary(
    primary_df: pd.DataFrame,
    represented_df: pd.DataFrame,
    *,
    sample_id_field: str = "sample_id",
    utterance_id_field: str | None = "utterance_id",
    unit_label: str = "utterances",
    unit_key_cols: list[str] | None = None,
) -> dict:
    """
    Summarize how much of the primary coding file is represented in reliability data.

    Coverage is computed as unique reliability/merged keys divided by unique keys in
    the primary coding file. For non-utterance reliability tables, pass
    unit_key_cols and a matching unit_label.
    """
    sample_cols = [sample_id_field]
    unit_cols = unit_key_cols or (
        [sample_id_field, utterance_id_field] if utterance_id_field else []
    )

    primary_samples = _unique_count(primary_df, sample_cols)
    represented_samples = _unique_count(represented_df, sample_cols)
    primary_units = _unique_count(primary_df, unit_cols) if unit_cols else 0
    represented_units = _unique_count(represented_df, unit_cols) if unit_cols else 0

    return {
        "sample_label": "samples",
        "represented_samples": represented_samples,
        "primary_samples": primary_samples,
        "sample_pct": round((represented_samples / primary_samples) * 100, 1)
        if primary_samples
        else np.nan,
        "unit_label": unit_label,
        "represented_units": represented_units,
        "primary_units": primary_units,
        "unit_pct": round((represented_units / primary_units) * 100, 1)
        if primary_units
        else np.nan,
    }


def variance_pair_stats(df: pd.DataFrame, col_org: str, col_rel: str) -> dict:
    """Return paired variance diagnostics for two numeric columns."""
    if df is None or df.empty or col_org not in df.columns or col_rel not in df.columns:
        return {"org_var": np.nan, "rel_var": np.nan, "pooled_var": np.nan}

    paired = df[[col_org, col_rel]].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()
    if len(paired) < 2:
        return {"org_var": np.nan, "rel_var": np.nan, "pooled_var": np.nan}

    org_var = float(paired[col_org].var())
    rel_var = float(paired[col_rel].var())
    return {
        "org_var": round(org_var, 4),
        "rel_var": round(rel_var, 4),
        "pooled_var": round(float((org_var + rel_var) / 2), 4),
    }


def categorical_variance_pair_stats(y_org: pd.Series, y_rel: pd.Series) -> dict:
    """
    Return variance diagnostics for categorical ratings using shared label codes.

    The numeric magnitude of encoded categorical variance is not intrinsically
    meaningful, but zero or near-zero values are useful collapse diagnostics next
    to kappa.
    """
    paired = pd.DataFrame({"org": y_org, "rel": y_rel}).dropna()
    if len(paired) < 2:
        return {"org_var": np.nan, "rel_var": np.nan, "pooled_var": np.nan}

    labels = sorted(
        pd.concat([paired["org"], paired["rel"]], ignore_index=True).astype(str).unique()
    )
    label_codes = {label: idx for idx, label in enumerate(labels)}
    encoded = paired.astype(str).replace(label_codes)
    return variance_pair_stats(encoded, "org", "rel")


def write_coverage_section(handle, coverage: dict | None) -> None:
    """Write a standard coverage section into a plain-text report."""
    if not coverage:
        return

    handle.write("Coverage in primary coding file\n")
    handle.write("--------------------------------\n")
    handle.write(
        f"Samples represented: {coverage.get('represented_samples', np.nan)}/"
        f"{coverage.get('primary_samples', np.nan)} ({coverage.get('sample_pct', np.nan)}%)\n"
    )
    handle.write(
        f"{coverage.get('unit_label', 'utterances').capitalize()} represented: "
        f"{coverage.get('represented_units', np.nan)}/"
        f"{coverage.get('primary_units', np.nan)} ({coverage.get('unit_pct', np.nan)}%)\n\n"
    )
