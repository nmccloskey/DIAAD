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
