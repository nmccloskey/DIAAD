from __future__ import annotations

import re
from numbers import Integral, Real
from pathlib import Path
from typing import Dict, List

import pandas as pd

from psair.core.logger import logger, get_rel_path
from psair.metadata.discovery import find_one_matching_file


CHAT_SUBDIR = "chat_files"
TRANSCRIPT_SUBDIR = "transcript_tables"
TRANSCRIPT_FILENAME = "transcript_tables.xlsx"
TEMPLATE_HEADER_PATTERN = "*template_header.cha"
DERIVED_FILE_COL = "derived_file"

FILENAME_EXCLUDE_COLS = {
    "input_order",
    "shuffled_order",
    DERIVED_FILE_COL,
}

DEFAULT_TEMPLATE_HEADER = """@Begin
@Languages:\teng
@Participants:\tPAR0 Participant, INV Investigator
@ID:\teng|corpus_name|PAR0|||||Participant|||
@ID:\teng|corpus_name|INV|||||Investigator|||"""

CHAT_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u02bc": "'",
        "\uff07": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\uff02": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u060c": ",",
        "\u3001": ",",
        "\ufe10": ",",
        "\ufe11": ",",
        "\ufe50": ",",
        "\ufe51": ",",
        "\uff0c": ",",
        "\u00bf": "?",
        "\u037e": "?",
        "\u061f": "?",
        "\ufe56": "?",
        "\uff1f": "?",
        "\u00a1": "!",
        "\ufe57": "!",
        "\uff01": "!",
        "\u06d4": ".",
        "\u2024": ".",
        "\u3002": ".",
        "\ufe52": ".",
        "\uff0e": ".",
        "\uff61": ".",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2026": "...",
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
        "\u2028": " ",
        "\u2029": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u2060": "",
    }
)

MOJIBAKE_PUNCTUATION_REPLACEMENTS = {
    "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u201d": "-",
    "\u00e2\u20ac\u00a6": "...",
}


def _cell_to_text(value: object) -> str:
    """
    Return a plain string for table cells, treating missing values as blank.
    """
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, Integral) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, Real) and not isinstance(value, bool) and float(value).is_integer():
        return str(int(value))

    return str(value)


def _regularize_chat_text(value: object) -> str:
    """
    Normalize Unicode punctuation to conservative ASCII for CHAT export.
    """
    text = _cell_to_text(value)
    for source, replacement in MOJIBAKE_PUNCTUATION_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    text = text.translate(CHAT_PUNCTUATION_TRANSLATION)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    return text.strip()


def _safe_filename_part(value: object) -> str:
    """
    Convert a metadata value into a safe single filename component.
    """
    text = _cell_to_text(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r'[<>:"/\\|?*\r\n\t]+', "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("._ ")


def _filename_metadata_columns(
    samples_df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> list[str]:
    """
    Return columns eligible for metadata-derived filenames, in sheet order.
    """
    exclude_cols = set(FILENAME_EXCLUDE_COLS)
    exclude_cols.add(sample_id_field)
    return [col for col in samples_df.columns if col not in exclude_cols]


def _build_base_filename(
    row: pd.Series,
    metadata_cols: list[str],
    row_index: int,
    sample_id_field: str = "sample_id",
) -> str:
    """
    Build an underscore-connected CHAT filename from a sample row.
    """
    parts = [_safe_filename_part(row.get(col)) for col in metadata_cols]
    parts = [part for part in parts if part]

    if not parts:
        sample_id = _safe_filename_part(row.get(sample_id_field))
        parts = [sample_id or f"sample_{row_index + 1}"]

    return f"{'_'.join(parts)}.cha"


def _append_row_index(filename: str, row_index: int) -> str:
    """
    Append a 1-based sample row index before the filename suffix.
    """
    path = Path(filename)
    return f"{path.stem}_{row_index + 1}{path.suffix or '.cha'}"


def _assign_derived_files(
    samples_df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Add a derived_file column with generated CHAT filenames.
    """
    if sample_id_field not in samples_df.columns:
        raise ValueError(f"Samples sheet must contain a {sample_id_field!r} column.")

    samples_df = samples_df.copy()
    metadata_cols = _filename_metadata_columns(
        samples_df,
        sample_id_field=sample_id_field,
    )

    base_names = [
        _build_base_filename(
            row,
            metadata_cols,
            row_index,
            sample_id_field=sample_id_field,
        )
        for row_index, (_, row) in enumerate(samples_df.iterrows())
    ]

    name_counts = pd.Series(base_names).value_counts()
    duplicate_names = {name for name, count in name_counts.items() if count > 1}
    duplicate_sample_ids = samples_df[sample_id_field].map(_cell_to_text).duplicated(keep=False)

    if duplicate_names or duplicate_sample_ids.any():
        logger.warning(
            "Transcript table does not produce one unique CHAT filename per sample; "
            "appending sample row indexes to affected derived filenames."
        )

    derived_files: list[str] = []
    seen: set[str] = set()
    for row_index, base_name in enumerate(base_names):
        needs_index = base_name in duplicate_names or bool(duplicate_sample_ids.iloc[row_index])
        candidate = _append_row_index(base_name, row_index) if needs_index else base_name

        while candidate in seen:
            candidate = _append_row_index(candidate, row_index)

        seen.add(candidate)
        derived_files.append(candidate)

    samples_df[DERIVED_FILE_COL] = derived_files
    return samples_df


def _find_transcript_tables(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    """
    Find the one exact transcript table workbook for this detabularization run.
    """
    directories = [Path(input_dir).expanduser().resolve()]
    if output_dir is not None:
        output_path = Path(output_dir).expanduser().resolve()
        if output_path not in directories:
            directories.append(output_path)

    return find_one_matching_file(
        directories=directories,
        filename=TRANSCRIPT_FILENAME,
        match_mode="exact",
        deduplicate=False,
        label="transcript table",
    )


def _find_template_header(input_dir: str | Path) -> str:
    """
    Read an optional CHAT template header, or return the default header.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    templates = sorted(input_dir.rglob(TEMPLATE_HEADER_PATTERN))

    if not templates:
        logger.info("No template header found; using default CHAT header.")
        return DEFAULT_TEMPLATE_HEADER

    if len(templates) > 1:
        logger.warning(
            "Multiple template headers found; using first: %s",
            get_rel_path(templates[0]),
        )

    template_path = templates[0]
    logger.info("Using template header: %s", get_rel_path(template_path))
    return template_path.read_text(encoding="utf-8-sig")


def _normalize_template_header(header: str) -> list[str]:
    """
    Prepare template header lines and remove any accidental terminal @End.
    """
    lines = [line.rstrip("\r\n") for line in header.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip() == "@End":
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    return lines or DEFAULT_TEMPLATE_HEADER.splitlines()


def _load_transcript_table(path: Path) -> tuple[dict[str, pd.DataFrame], str, str]:
    """
    Load all sheets from a transcript table and identify samples/utterances.
    """
    with pd.ExcelFile(path, engine="openpyxl") as xls:
        sheets = {sheet: pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names}

    sheet_lookup = {sheet.lower(): sheet for sheet in sheets}
    if "samples" not in sheet_lookup:
        raise ValueError(f"Samples sheet not found in {path}.")
    if "utterances" not in sheet_lookup:
        raise ValueError(f"Utterances sheet not found in {path}.")

    return sheets, sheet_lookup["samples"], sheet_lookup["utterances"]


def _sort_utterances(utterance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort utterances by position fields when available, preserving row order otherwise.
    """
    sort_cols = [col for col in ("position", "position_sub") if col in utterance_df.columns]
    if not sort_cols:
        return utterance_df
    return utterance_df.sort_values(sort_cols, kind="mergesort")


def _format_speaker_tier(speaker: str) -> str:
    """
    Return a CHAT speaker tier label from a speaker cell.
    """
    speaker = speaker.strip()
    if not speaker:
        return ""
    if not speaker.startswith("*"):
        speaker = f"*{speaker}"
    if not speaker.endswith(":"):
        speaker = f"{speaker}:"
    return speaker


def _build_chat_text(header_lines: list[str], utterance_df: pd.DataFrame) -> str | None:
    """
    Build CHAT file text for one sample.
    """
    required = {"speaker", "utterance"}
    missing = sorted(required - set(utterance_df.columns))
    if missing:
        raise ValueError(f"Utterances sheet missing required columns: {missing}")

    lines = list(header_lines)
    utterance_count = 0

    for _, row in _sort_utterances(utterance_df).iterrows():
        speaker_tier = _format_speaker_tier(_cell_to_text(row.get("speaker")))
        utterance = _regularize_chat_text(row.get("utterance"))

        if not speaker_tier:
            logger.warning("Skipping utterance row with blank speaker.")
            continue

        lines.append(f"{speaker_tier}\t{utterance}")
        utterance_count += 1

        comment = _regularize_chat_text(row.get("comment"))
        if comment:
            lines.append(f"%com:\t{comment}")

    if utterance_count == 0:
        return None

    lines.append("@End")
    return "\n".join(lines) + "\n"


def _write_updated_transcript_table(
    sheets: dict[str, pd.DataFrame],
    samples_sheet_name: str,
    samples_df: pd.DataFrame,
    output_dir: Path,
) -> str | None:
    """
    Write a run-output copy of the workbook with derived_file in samples.
    """
    transcript_dir = output_dir / TRANSCRIPT_SUBDIR
    transcript_dir.mkdir(parents=True, exist_ok=True)

    output_path = transcript_dir / TRANSCRIPT_FILENAME

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, sheet_df in sheets.items():
                out_df = samples_df if sheet_name == samples_sheet_name else sheet_df
                out_df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info("Wrote transcript table with derived files: %s", get_rel_path(output_path))
        return str(output_path)
    except Exception as e:
        logger.error("Failed to write %s: %s", get_rel_path(output_path), e)
        return None


def detabularize_transcripts(
    input_dir: str | Path,
    output_dir: str | Path,
    sample_id_field: str = "sample_id",
) -> List[str]:
    """
    Convert transcript tables into CHAT (.cha) files.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing one ``transcript_tables.xlsx`` file
        and optionally a ``*template_header.cha`` file.
    output_dir : str or Path
        Run output directory. CHAT files are written to ``chat_files/``.

    Returns
    -------
    list[str]
        Paths to written CHAT files.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    chat_dir = output_dir / CHAT_SUBDIR
    chat_dir.mkdir(parents=True, exist_ok=True)

    transcript_table = _find_transcript_tables(input_dir, output_dir)

    header_lines = _normalize_template_header(_find_template_header(input_dir))
    written_chats: list[str] = []

    try:
        sheets, samples_sheet_name, utterances_sheet_name = _load_transcript_table(transcript_table)
        samples_df = _assign_derived_files(
            sheets[samples_sheet_name],
            sample_id_field=sample_id_field,
        )
        utterances_df = sheets[utterances_sheet_name].copy()

        if sample_id_field not in utterances_df.columns:
            raise ValueError(
                f"Utterances sheet must contain a {sample_id_field!r} column."
            )

        samples_df["_sample_key"] = samples_df[sample_id_field].map(_cell_to_text)
        utterances_df["_sample_key"] = utterances_df[sample_id_field].map(_cell_to_text)
        utterance_groups: Dict[str, pd.DataFrame] = dict(
            tuple(utterances_df.groupby("_sample_key", sort=False))
        )

        samples_for_output = samples_df.drop(columns=["_sample_key"])
        _write_updated_transcript_table(
            sheets,
            samples_sheet_name,
            samples_for_output,
            output_dir,
        )

        for _, sample_row in samples_df.iterrows():
            sample_key = sample_row["_sample_key"]
            derived_file = sample_row[DERIVED_FILE_COL]
            sample_utts = utterance_groups.get(sample_key)

            if sample_utts is None or sample_utts.empty:
                logger.warning(
                    "No utterances found for sample_id %r; no CHAT file written.",
                    sample_row.get(sample_id_field),
                )
                continue

            sample_utts = sample_utts.drop(columns=["_sample_key"], errors="ignore")
            chat_text = _build_chat_text(header_lines, sample_utts)
            if chat_text is None:
                logger.warning(
                    "No writable utterances found for sample_id %r; no CHAT file written.",
                    sample_row.get(sample_id_field),
                )
                continue

            chat_path = chat_dir / derived_file
            chat_path.write_text(chat_text, encoding="utf-8")
            written_chats.append(str(chat_path))

    except Exception as e:
        logger.error("Failed to detabularize %s: %s", get_rel_path(transcript_table), e)
        return []

    logger.info("Successfully wrote %s CHAT file(s) to %s.", len(written_chats), get_rel_path(chat_dir))
    return written_chats
