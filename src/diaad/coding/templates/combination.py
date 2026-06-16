from __future__ import annotations

from pathlib import Path

import pandas as pd

from psair.core.logger import get_rel_path, logger

from diaad.coding.templates.utils import TEMPLATE_SUBDIR
from diaad.metadata.discovery import find_files_by_extension


COMBINED_TEMPLATE_FILENAME = "combined.xlsx"
METADATA_SHEET = "metadata"
COMBINED_ID_COL = "combined_id"
SOURCE_FILE_COL = "source_file"
METADATA_COLUMNS = [SOURCE_FILE_COL, "order", "sheet", "num_rows"]
RESERVED_OUTPUT_COLUMNS = {COMBINED_ID_COL, SOURCE_FILE_COL}


def _find_combination_inputs(input_dir: str | Path) -> list[Path]:
    """
    Find Excel workbooks recursively inside the input directory.
    """
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(
            f"DIAAD did not find a template combination input directory: "
            f"{get_rel_path(input_path)}."
        )

    paths = find_files_by_extension(directories=input_path, search_ext=".xlsx")
    if not paths:
        raise FileNotFoundError(
            f"DIAAD did not find any .xlsx files under "
            f"{get_rel_path(input_path)}."
        )

    return paths


def _relative_source_file(path: Path, input_dir: Path) -> str:
    """
    Return the workbook path relative to the input directory.
    """
    return path.resolve().relative_to(input_dir.resolve()).as_posix()


def _format_list(values: list[object]) -> str:
    """
    Format values for concise validation errors.
    """
    return ", ".join(repr(value) for value in values) if values else "none"


def _read_workbook(path: Path) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """
    Read all sheets from a workbook while preserving workbook sheet order.
    """
    with pd.ExcelFile(path) as xls:
        sheet_names = list(xls.sheet_names)
        if METADATA_SHEET in sheet_names:
            raise ValueError(
                f"{get_rel_path(path)} contains a '{METADATA_SHEET}' sheet, which "
                f"is reserved for {COMBINED_TEMPLATE_FILENAME} output metadata."
            )

        sheets = {
            sheet_name: pd.read_excel(xls, sheet_name=sheet_name)
            for sheet_name in sheet_names
        }
    return sheet_names, sheets


def _validate_sheet_names(
    *,
    path: Path,
    sheet_names: list[str],
    expected_sheet_names: list[str],
) -> None:
    """
    Require each input workbook to contain the same sheets.
    """
    missing = [name for name in expected_sheet_names if name not in sheet_names]
    extra = [name for name in sheet_names if name not in expected_sheet_names]
    if not missing and not extra:
        return

    raise ValueError(
        f"{get_rel_path(path)} has sheet names that do not match the first "
        f"input workbook. Missing: {_format_list(missing)}. "
        f"Extra: {_format_list(extra)}."
    )


def _validate_columns(
    *,
    path: Path,
    sheet_name: str,
    columns: list[object],
    expected_columns: list[object],
) -> None:
    """
    Require a sheet to contain the same columns as the first input workbook.
    """
    reserved = [col for col in columns if col in RESERVED_OUTPUT_COLUMNS]
    if reserved:
        raise ValueError(
            f"{get_rel_path(path)} sheet '{sheet_name}' contains reserved "
            f"output column(s): {_format_list(reserved)}."
        )

    missing = [name for name in expected_columns if name not in columns]
    extra = [name for name in columns if name not in expected_columns]
    if not missing and not extra:
        return

    raise ValueError(
        f"{get_rel_path(path)} sheet '{sheet_name}' has columns that do not "
        f"match the first input workbook. Missing: {_format_list(missing)}. "
        f"Extra: {_format_list(extra)}."
    )


def _read_consistent_workbooks(
    paths: list[Path],
    *,
    input_dir: Path,
) -> tuple[list[str], dict[str, list[pd.DataFrame]], pd.DataFrame]:
    """
    Read all workbooks and validate sheet and column consistency.
    """
    expected_sheet_names: list[str] | None = None
    expected_columns_by_sheet: dict[str, list[object]] = {}
    frames_by_sheet: dict[str, list[pd.DataFrame]] = {}
    metadata_rows: list[dict[str, object]] = []

    for order, path in enumerate(paths, start=1):
        sheet_names, sheets = _read_workbook(path)
        source_file = _relative_source_file(path, input_dir)

        if expected_sheet_names is None:
            expected_sheet_names = sheet_names
            frames_by_sheet = {sheet_name: [] for sheet_name in expected_sheet_names}

        _validate_sheet_names(
            path=path,
            sheet_names=sheet_names,
            expected_sheet_names=expected_sheet_names,
        )

        for sheet_name in expected_sheet_names:
            df = sheets[sheet_name]
            columns = list(df.columns)

            if sheet_name not in expected_columns_by_sheet:
                expected_columns_by_sheet[sheet_name] = columns

            _validate_columns(
                path=path,
                sheet_name=sheet_name,
                columns=columns,
                expected_columns=expected_columns_by_sheet[sheet_name],
            )

            part = df.loc[:, expected_columns_by_sheet[sheet_name]].copy()
            part.insert(0, SOURCE_FILE_COL, source_file)
            frames_by_sheet[sheet_name].append(part)
            metadata_rows.append(
                {
                    SOURCE_FILE_COL: source_file,
                    "order": order,
                    "sheet": sheet_name,
                    "num_rows": len(df),
                }
            )

    if expected_sheet_names is None:
        raise FileNotFoundError("No input workbooks were found for combination.")

    metadata_df = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)
    return expected_sheet_names, frames_by_sheet, metadata_df


def _combine_sheet_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Stack one sheet's input dataframes and add the generated primary key.
    """
    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, COMBINED_ID_COL, range(1, len(combined) + 1))
    return combined


def _write_combined_workbook(
    *,
    sheet_names: list[str],
    frames_by_sheet: dict[str, list[pd.DataFrame]],
    metadata_df: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """
    Write combined sheets plus combination metadata.
    """
    template_dir = Path(output_dir) / TEMPLATE_SUBDIR
    template_dir.mkdir(parents=True, exist_ok=True)
    path = template_dir / COMBINED_TEMPLATE_FILENAME

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in sheet_names:
            combined_df = _combine_sheet_frames(frames_by_sheet[sheet_name])
            combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
        metadata_df.to_excel(writer, sheet_name=METADATA_SHEET, index=False)

    logger.info("Wrote combined template workbook: %s", get_rel_path(path))
    return path


def make_combined_template_file(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Combine same-schema Excel template workbooks into one workbook.

    The command scans the input directory recursively for ``.xlsx`` files.
    All input workbooks must have the same sheet names, and each matching sheet
    must have the same columns. Empty sheets with headers are allowed. The
    output mirrors the input sheet names, adds ``combined_id`` and
    ``source_file`` to each sheet, and writes a ``metadata`` sheet that records
    row counts by source file and sheet.
    """
    input_path = Path(input_dir)
    paths = _find_combination_inputs(input_path)

    logger.info(
        "Combining %s template workbook(s): %s",
        len(paths),
        ", ".join(get_rel_path(path) for path in paths),
    )

    sheet_names, frames_by_sheet, metadata_df = _read_consistent_workbooks(
        paths,
        input_dir=input_path,
    )
    return _write_combined_workbook(
        sheet_names=sheet_names,
        frames_by_sheet=frames_by_sheet,
        metadata_df=metadata_df,
        output_dir=output_dir,
    )
