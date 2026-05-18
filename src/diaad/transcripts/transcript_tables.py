from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from psair.core.logger import logger, get_rel_path


SAMPLE_BASE_COLS = [
    "file",
    "file_ext",
    "file_dir",
    "input_order",
    "shuffled_order"
]

UTT_COLS = [
    "position",
    "position_sub",
    "speaker",
    "utterance",
    "comment",
]

TRANSCRIPT_SUBDIR = "transcript_tables"
TRANSCRIPT_FILENAME = "transcript_tables.xlsx"


def zero_pad(num: int, lower_bound: int = 3) -> int:
    """
    Determine adaptive zero-padding width for numeric identifiers.

    Parameters
    ----------
    num : int
        Maximum number expected in the sequence.
    lower_bound : int, default=3
        Minimum padding width.

    Returns
    -------
    int
        Padding width ensuring consistent formatting.
    """
    return max(lower_bound, len(str(max(num, 1))))


def _metadata_field_labels(
    metadata_fields: Dict[str, object],
    source_path: str | Path,
) -> list[object]:
    """
    Extract metadata field values from a transcript path.
    """
    path = Path(source_path)
    parts = [part for part in path.parts if part not in ("", ".")]

    labels = []
    for field in metadata_fields.values():
        if hasattr(field, "match_path_parts"):
            labels.append(field.match_path_parts(parts, source=str(path)))
        else:
            labels.append(field.match(str(path)))
    return labels


def _sample_file_components(source_path: str | Path) -> tuple[str, str, str]:
    """
    Return file stem, extension, and directory for sample-level output rows.
    """
    path = Path(source_path)
    file_dir = "" if str(path.parent) == "." else path.parent.as_posix()
    return path.stem, path.suffix, file_dir


def _count_total_utterances(chats: Dict[str, object], file_list: List[str]) -> int:
    """
    Count total utterances across all CHAT files in the run.

    Used to set a consistent utterance_id padding width.
    """
    total = 0
    for chat_file in file_list:
        try:
            chat_data = chats[chat_file]
            utterances = getattr(chat_data, "utterances", lambda: [])()
            total += len(utterances)
        except Exception as e:
            logger.warning(f"Could not count utterances in {get_rel_path(chat_file)}: {e}")
    return total


def _build_sample_id_map(
    file_list_sorted: List[str],
    *,
    shuffle: bool,
    rng: np.random.Generator | None,
    sample_pad: int,
) -> tuple[dict[str, str], dict[str, int]]:
    """
    Build mappings from file -> sample_id and file -> shuffled_order.
    """
    if shuffle:
        shuffled = file_list_sorted.copy()
        if rng is None:
            raise ValueError("Shuffle requested but RNG was not provided.")
        rng.shuffle(shuffled)
        file_to_shuffled_order = {f: i + 1 for i, f in enumerate(shuffled)}
        file_to_sample_id = {
            f: f"S{file_to_shuffled_order[f]:0{sample_pad}d}" for f in file_list_sorted
        }
    else:
        file_to_shuffled_order = {}
        file_to_sample_id = {
            f: f"S{i + 1:0{sample_pad}d}" for i, f in enumerate(file_list_sorted)
        }

    return file_to_sample_id, file_to_shuffled_order


def _write_transcript_tables(
    sample_df: pd.DataFrame,
    utt_df: pd.DataFrame,
    output_dir: Path,
) -> str | None:
    """
    Write transcript sample- and utterance-level tables to Excel.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Sample-level transcript table.
    utt_df : pd.DataFrame
        Utterance-level transcript table.
    output_dir : Path
        Project output directory.

    Returns
    -------
    str | None
        Path to the written file as a string, or None if writing failed.
    """
    transcript_dir = output_dir / TRANSCRIPT_SUBDIR
    transcript_dir.mkdir(parents=True, exist_ok=True)

    filename = transcript_dir / TRANSCRIPT_FILENAME

    try:
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            sample_df.to_excel(writer, sheet_name="samples", index=False)
            utt_df.to_excel(writer, sheet_name="utterances", index=False)
        logger.info(f"Wrote transcript table: {get_rel_path(filename)}")
        return str(filename)
    except Exception as e:
        logger.error(f"Failed to write {get_rel_path(filename)}: {e}")
        return None


def tabularize_transcripts(
    metadata_fields: Dict[str, object],
    chats: Dict[str, object],
    output_dir: Path,
    *,
    shuffle: bool = False,
    random_seed: int | None = 99,
    sample_id_field: str = "sample_id",
    utterance_id_field: str = "utterance_id",
) -> List[str]:
    """
    Create and write transcript tables (samples + utterances) to Excel.

    Parameters
    ----------
    metadata_fields : dict
        MetadataField objects used to extract transcript metadata.
    chats : dict
        CHAT file readers indexed by input-relative transcript path.
    output_dir : Path
        Directory to create a 'transcript_tables' subfolder within.
    shuffle : bool, default=False
        Disrupt automated file order when assigning sample identifiers.
    random_seed : int | None, default=99
        Seed for deterministic sample shuffling.

    Returns
    -------
    list[str]
        List containing the written transcript table path, if successful.

    Notes
    -----
    The output Excel file contains:
      • Sheet 'samples'     — sample-level metadata and file info
      • Sheet 'utterances'  — utterance-level text data
    """
    if not chats:
        logger.warning("No CHAT files provided; no transcript tables created.")
        return []

    metadata_field_names = list(metadata_fields.keys())
    sample_cols = [sample_id_field] + SAMPLE_BASE_COLS + metadata_field_names
    utt_cols = [sample_id_field, utterance_id_field] + UTT_COLS

    file_list_sorted = sorted(chats.keys())
    rng = np.random.default_rng(random_seed) if shuffle else None

    if shuffle:
        logger.info(f"Shuffling enabled for transcript tabularization (seed={random_seed}).")

    sample_pad = zero_pad(len(file_list_sorted), 3)
    total_utts = _count_total_utterances(chats, file_list_sorted)
    utt_pad = zero_pad(total_utts, 4)

    file_to_sample_id, file_to_shuffled_order = _build_sample_id_map(
        file_list_sorted,
        shuffle=shuffle,
        rng=rng,
        sample_pad=sample_pad,
    )

    sample_rows: List[list] = []
    utt_rows: List[list] = []

    for input_idx, chat_file in enumerate(
        tqdm(file_list_sorted, desc="Building transcript tables"),
        start=1,
    ):
        try:
            labels_all = _metadata_field_labels(metadata_fields, chat_file)
            sample_id = file_to_sample_id[chat_file]
            shuffled_order = file_to_shuffled_order.get(chat_file, np.nan)
            file_stem, file_ext, file_dir = _sample_file_components(chat_file)

            sample_rows.append(
                [
                    sample_id,
                    file_stem,
                    file_ext,
                    file_dir,
                    input_idx,
                    shuffled_order,
                ] + labels_all
            )

            chat_data = chats[chat_file]
            utterances = getattr(chat_data, "utterances", lambda: [])()

            for j, line in enumerate(utterances, start=1):
                speaker = getattr(line, "participant", None)
                utterance_annotations = getattr(line, "tiers", {}) or {}
                utterance_text = utterance_annotations.get(speaker, "")
                comment = utterance_annotations.get("%com", None)

                utt_id = f"U{j:0{utt_pad}d}"
                position = j
                position_sub = 0

                utt_rows.append(
                    [
                        sample_id,
                        utt_id,
                        position,
                        position_sub,
                        speaker,
                        utterance_text,
                        comment,
                    ]
                )
        except Exception as e:
            logger.error(f"Error processing {get_rel_path(chat_file)}: {e}")
            continue

    sample_df = pd.DataFrame(sample_rows, columns=sample_cols)
    utt_df = pd.DataFrame(utt_rows, columns=utt_cols)

    written_file = _write_transcript_tables(sample_df, utt_df, output_dir)

    written = [written_file] if written_file else []
    logger.info(f"Successfully wrote {len(written)} transcript table(s).")
    return written


def extract_transcript_data(
    transcript_table_path: str | Path,
    kind: str = "joined",
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Load data from a transcript table Excel file.

    Parameters
    ----------
    transcript_table_path : str or Path
        Path to an Excel file produced by `tabularize_transcripts`.
    kind : {'utterance', 'sample', 'joined'}, default='joined'
        Which dataset to return:
          - 'utterance': utterance-level data
          - 'sample': sample-level metadata
          - 'joined': merged table of both (inner join on `sample_id_field`)

    Returns
    -------
    pandas.DataFrame
        The requested DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the `kind` argument is invalid.
    """
    path = Path(transcript_table_path).expanduser().resolve()
    if not path.exists():
        logger.error(f"Transcript table not found: {get_rel_path(path)}")
        raise FileNotFoundError(f"Transcript table not found: {path}")

    if kind not in {"sample", "utterance", "joined"}:
        raise ValueError(
            f"Invalid kind '{kind}'. Must be 'sample', 'utterance', or 'joined'."
        )

    try:
        with pd.ExcelFile(path, engine="openpyxl") as xls:
            sheet_names = {s.lower() for s in xls.sheet_names}
            sample_df = (
                pd.read_excel(xls, sheet_name="samples")
                if "samples" in sheet_names else None
            )
            utt_df = (
                pd.read_excel(xls, sheet_name="utterances")
                if "utterances" in sheet_names else None
            )

        if kind == "sample":
            if sample_df is None:
                raise ValueError("Sample sheet not found in transcript table.")
            logger.info(f"Loaded sample data from {get_rel_path(path)}")
            return sample_df

        if kind == "utterance":
            if utt_df is None:
                raise ValueError("Utterance sheet not found in transcript table.")
            logger.info(f"Loaded utterance data from {get_rel_path(path)}")
            return utt_df

        if sample_df is None or utt_df is None:
            raise ValueError("Both sheets required for joined kind are missing.")

        missing_join = [
            sheet
            for sheet, df in (("samples", sample_df), ("utterances", utt_df))
            if sample_id_field not in df.columns
        ]
        if missing_join:
            raise ValueError(
                f"Missing join column {sample_id_field!r} in transcript table "
                f"sheet(s): {missing_join}."
            )

        joined = sample_df.merge(utt_df, on=sample_id_field, how="inner")
        logger.info(f"Loaded joined transcript data from {get_rel_path(path)}")
        return joined

    except Exception as e:
        logger.error(f"Failed to read {get_rel_path(path)}: {e}")
        raise
