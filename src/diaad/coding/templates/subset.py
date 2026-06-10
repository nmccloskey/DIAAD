from __future__ import annotations

from numbers import Integral, Real
from pathlib import Path
import random
from typing import Any

import pandas as pd

from psair.core.logger import get_rel_path, logger

from diaad.coding.templates.utils import TEMPLATE_SUBDIR, require_columns
from diaad.coding.utils.sampling import calc_subset_size
from diaad.metadata.discovery import find_one_file_by_extension


SAMPLE_SUBSET_FILENAME = "sample_subset.xlsx"
SAMPLES_SHEET = "samples"
SUBSET_SHEET = "subset"
INPUT_EXCLUDE_COL = "exclude"
OUTPUT_SELECTED_COL = "selected"
OUTPUT_EXCLUDED_COL = "excluded"


def _find_sample_subset_input(input_dir: str | Path) -> Path:
    """
    Find exactly one Excel workbook in the input directory.
    """
    return find_one_file_by_extension(
        directories=input_dir,
        search_ext=".xlsx",
        label="sample subset input workbook",
    )


def _read_samples_sheet(path: Path, *, sample_id_field: str) -> pd.DataFrame:
    """
    Load and validate the samples sheet from a sample subset input workbook.
    """
    xls = pd.ExcelFile(path)
    if SAMPLES_SHEET not in xls.sheet_names:
        raise ValueError(
            f"{get_rel_path(path)} must contain a '{SAMPLES_SHEET}' sheet."
        )

    df = pd.read_excel(path, sheet_name=SAMPLES_SHEET)
    require_columns(df, [sample_id_field], f"{path.name}:{SAMPLES_SHEET}")

    missing = df[sample_id_field].isna()
    if df[sample_id_field].dtype == object:
        missing = missing | df[sample_id_field].map(
            lambda value: isinstance(value, str) and not value.strip()
        )
    if missing.any():
        raise ValueError(
            f"{get_rel_path(path)} sheet '{SAMPLES_SHEET}' contains "
            f"{int(missing.sum())} blank '{sample_id_field}' value(s)."
        )

    return df


def _coerce_binary_exclude_value(value: Any) -> int:
    """
    Coerce a single exclude value to 0 or 1.
    """
    if pd.isna(value):
        raise ValueError("missing value")

    if isinstance(value, str):
        stripped = value.strip()
        if stripped in {"0", "1"}:
            return int(stripped)
        raise ValueError(f"{value!r}")

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, Integral) and value in {0, 1}:
        return int(value)

    if isinstance(value, Real) and value in {0.0, 1.0}:
        return int(value)

    raise ValueError(f"{value!r}")


def _coerce_binary_exclude(series: pd.Series, *, path: Path) -> pd.Series:
    """
    Validate and normalize a user-provided exclude column.
    """
    values: list[int] = []
    invalid: list[Any] = []

    for value in series:
        try:
            values.append(_coerce_binary_exclude_value(value))
        except ValueError:
            invalid.append(value)

    if invalid:
        preview = ", ".join(repr(value) for value in invalid[:5])
        raise ValueError(
            f"{get_rel_path(path)} column '{INPUT_EXCLUDE_COL}' must contain "
            f"only 0/1 values; invalid value(s): {preview}."
        )

    return pd.Series(values, index=series.index, dtype="int64")


def _build_sample_status(
    input_df: pd.DataFrame,
    *,
    sample_id_field: str,
    path: Path,
) -> tuple[pd.DataFrame, bool]:
    """
    Collapse the input samples sheet to one status row per sample identifier.
    """
    unique_samples = input_df.loc[:, [sample_id_field]].drop_duplicates()
    re_subset_mode = INPUT_EXCLUDE_COL in input_df.columns

    if not re_subset_mode:
        out = unique_samples.copy()
        out[OUTPUT_EXCLUDED_COL] = 0
        return out, False

    work = input_df.loc[:, [sample_id_field, INPUT_EXCLUDE_COL]].copy()
    work[OUTPUT_EXCLUDED_COL] = _coerce_binary_exclude(
        work[INPUT_EXCLUDE_COL],
        path=path,
    )

    grouped = work.groupby(sample_id_field, sort=False)[OUTPUT_EXCLUDED_COL]
    min_values = grouped.min()
    max_values = grouped.max()
    conflicting = min_values[min_values != max_values].index.tolist()
    if conflicting:
        logger.warning(
            "Found %s sample identifier(s) with mixed exclude values in %s; "
            "marking a sample excluded when any duplicate row is excluded.",
            len(conflicting),
            get_rel_path(path),
        )

    excluded = max_values.rename(OUTPUT_EXCLUDED_COL).reset_index()
    out = unique_samples.merge(
        excluded,
        on=sample_id_field,
        how="left",
        validate="1:1",
    )
    out[OUTPUT_EXCLUDED_COL] = out[OUTPUT_EXCLUDED_COL].astype("int64")
    return out, True


def _select_samples(
    samples_df: pd.DataFrame,
    *,
    frac: float,
    sample_id_field: str,
    seed: int | None,
) -> pd.DataFrame:
    """
    Add a randomized selected column to a sample status dataframe.
    """
    if samples_df.empty:
        raise ValueError("No samples are available for subset selection.")

    target_size = calc_subset_size(frac=frac, samples=samples_df)
    candidates = samples_df.loc[
        samples_df[OUTPUT_EXCLUDED_COL] == 0,
        sample_id_field,
    ].tolist()

    if not candidates:
        raise ValueError("No eligible samples are available for subset selection.")

    n_samples = min(target_size, len(candidates))
    if n_samples < target_size:
        logger.warning(
            "Only %s/%s eligible samples are available for sample subsetting.",
            n_samples,
            target_size,
        )

    rng = random.Random(seed)
    selected = set(rng.sample(candidates, k=n_samples))

    out = samples_df.copy()
    out[OUTPUT_SELECTED_COL] = out[sample_id_field].isin(selected).astype("int64")
    return out.loc[:, [sample_id_field, OUTPUT_SELECTED_COL, OUTPUT_EXCLUDED_COL]]


def _write_sample_subset_workbook(
    samples_df: pd.DataFrame,
    *,
    output_dir: str | Path,
) -> Path:
    """
    Write sample subset status and selected rows.
    """
    template_dir = Path(output_dir) / TEMPLATE_SUBDIR
    template_dir.mkdir(parents=True, exist_ok=True)
    path = template_dir / SAMPLE_SUBSET_FILENAME

    subset_df = samples_df.loc[samples_df[OUTPUT_SELECTED_COL] == 1].copy()

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        samples_df.to_excel(writer, sheet_name=SAMPLES_SHEET, index=False)
        subset_df.to_excel(writer, sheet_name=SUBSET_SHEET, index=False)

    logger.info("Wrote sample subset workbook: %s", get_rel_path(path))
    return path


def make_sample_subset_file(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    frac: float,
    sample_id_field: str = "sample_id",
    seed: int | None = None,
) -> Path:
    """
    Create a randomized sample subset workbook from one input Excel workbook.

    The input directory must contain exactly one ``.xlsx`` file. That workbook
    must contain a ``samples`` sheet with ``sample_id_field``. If the sheet also
    contains a binary ``exclude`` column, selection is made only from rows whose
    collapsed sample-level ``exclude`` value is 0 while the target subset size
    is still calculated from the full sample set.
    """
    input_path = _find_sample_subset_input(input_dir)
    logger.info("Using sample subset input workbook: %s", get_rel_path(input_path))

    input_df = _read_samples_sheet(input_path, sample_id_field=sample_id_field)
    sample_status, re_subset_mode = _build_sample_status(
        input_df,
        sample_id_field=sample_id_field,
        path=input_path,
    )
    sample_status = _select_samples(
        sample_status,
        frac=frac,
        sample_id_field=sample_id_field,
        seed=seed,
    )

    logger.info(
        "Prepared sample %s: selected %s of %s sample(s); excluded %s.",
        "re-subset" if re_subset_mode else "subset",
        int(sample_status[OUTPUT_SELECTED_COL].sum()),
        len(sample_status),
        int(sample_status[OUTPUT_EXCLUDED_COL].sum()),
    )

    return _write_sample_subset_workbook(sample_status, output_dir=output_dir)
