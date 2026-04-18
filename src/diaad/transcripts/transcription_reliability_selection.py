from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from psair.core.logger import logger, get_rel_path
from psair.metadata.discovery import find_matching_files
from src.diaad.coding.utils.sampling import calc_subset_size
from diaad.transcripts.transcript_tables import extract_transcript_data


RELIABILITY_SELECTION_SUBDIR = "transcription_reliability_selection"
RELIABILITY_SELECTION_FILENAME = "transcription_reliability_samples.xlsx"
RELIABILITY_SELECTION_SHEET = "reliability_selection"
ALL_TRANSCRIPTS_SHEET = "all_transcripts"
SELECTED_COL = "selected_for_reliability"


def _build_samples_df_from_chats(tiers, chats) -> pd.DataFrame:
    """
    Build a sample-level dataframe from CHAT files.
    """
    columns = ["file"] + list(tiers.keys())
    rows: list[list[Any]] = []

    for cha_file in sorted(chats):
        try:
            labels = [t.match(str(cha_file)) for t in tiers.values()]
            rows.append([cha_file] + labels)
        except Exception as e:
            logger.error(f"Failed to parse tier labels for {get_rel_path(cha_file)}: {e}")

    return pd.DataFrame(rows, columns=columns)


def _load_samples_df_from_transcript_tables(
    input_dir: Path,
    output_dir: Path,
) -> pd.DataFrame | None:
    """
    Attempt to load sample-level data from a transcript table Excel file.

    Search order
    ------------
    Searches both input_dir and output_dir for files matching 'transcript_tables'.

    Behavior
    --------
    - If no transcript tables are found, returns None.
    - If multiple transcript tables are found, logs a warning and uses the first.
    """
    transcript_tables = find_matching_files(
        directories=[input_dir, output_dir],
        search_base="transcript_tables",
    )

    if not transcript_tables:
        logger.info("No transcript tables found.")
        return None

    if len(transcript_tables) > 1:
        logger.warning(
            "Multiple transcript tables found; using the first match: %s. Other matches: %s",
            get_rel_path(transcript_tables[0]),
            [get_rel_path(p) for p in transcript_tables[1:]],
        )

    transcript_table_path = transcript_tables[0]

    try:
        samples_df = extract_transcript_data(
            transcript_table_path=transcript_table_path,
            kind="sample",
        )
    except Exception as e:
        logger.error(
            f"Failed to load transcript table sample data from {get_rel_path(transcript_table_path)}: {e}"
        )
        return None

    if samples_df is None or samples_df.empty:
        logger.info(f"Transcript table {get_rel_path(transcript_table_path)} contained no sample rows.")
        return None

    logger.info(
        f"Loaded {len(samples_df)} samples from transcript table {get_rel_path(transcript_table_path)}."
    )
    return samples_df.copy()


def _write_blank_reliability_chat_files(
    subset_df: pd.DataFrame,
    chats,
    output_dir: Path,
) -> None:
    """
    Write blank reliability CHAT files containing only CHAT headers.
    """
    transc_rel_dir = output_dir / RELIABILITY_SELECTION_SUBDIR
    transc_rel_dir.mkdir(parents=True, exist_ok=True)

    for cha_file in tqdm(
        subset_df["file"].tolist(),
        desc="Writing blank reliability CHAT files",
    ):
        try:
            chat_data = chats[cha_file]
            strs = next(chat_data.to_strs())
            strs = ["@Begin"] + strs.split("\n") + ["@End"]

            filepath = transc_rel_dir / f"{Path(cha_file).stem}_reliability.cha"
            with filepath.open("w", encoding="utf-8") as f:
                for line in strs:
                    if line.startswith("@"):
                        f.write(line + "\n")

            logger.info(f"Wrote blank CHAT header: {get_rel_path(filepath)}")
        except Exception as e:
            logger.error(f"Failed to write blank CHAT for {get_rel_path(cha_file)}: {e}")


def _write_reliability_selection_excel(
    df_subset: pd.DataFrame,
    df_all: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    """
    Write transcription reliability selection workbook.
    """
    transc_rel_dir = output_dir / RELIABILITY_SELECTION_SUBDIR
    transc_rel_dir.mkdir(parents=True, exist_ok=True)

    filepath = transc_rel_dir / RELIABILITY_SELECTION_FILENAME

    try:
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            df_subset.to_excel(
                writer,
                sheet_name=RELIABILITY_SELECTION_SHEET,
                index=False,
            )
            df_all.to_excel(
                writer,
                sheet_name=ALL_TRANSCRIPTS_SHEET,
                index=False,
            )
        logger.info(f"Reliability Excel saved to: {get_rel_path(filepath)}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to write reliability Excel {get_rel_path(filepath)}: {e}")
        return None


def select_transcription_reliability_samples(
    tiers,
    chats,
    frac,
    output_dir,
    input_dir=None,
):
    """
    Select random transcripts for transcription reliability and save both sampled
    and full sample sets.

    Preferred input source is a transcript table Excel file. If no transcript
    table is found, the function falls back to CHAT files.

    Outputs
    -------
    Creates:
      - one Excel file with sheets:
          * 'reliability_selection'
          * 'all_transcripts'
      - blank '_reliability.cha' header files when CHAT objects are available
    """
    logger.info("Starting transcription reliability sample selection.")

    output_dir = Path(output_dir)
    input_dir = Path(input_dir) if input_dir is not None else output_dir

    samples_df = _load_samples_df_from_transcript_tables(
        input_dir=input_dir,
        output_dir=output_dir,
    )

    used_transcript_tables = samples_df is not None

    if samples_df is None:
        if not chats:
            logger.error(
                "No transcript tables found and no CHAT files provided; "
                "cannot select transcription reliability samples."
            )
            return None

        logger.info("Falling back to CHAT files for reliability sample selection.")
        samples_df = _build_samples_df_from_chats(tiers, chats)

    if samples_df.empty:
        logger.error("No samples available for transcription reliability selection.")
        return None

    subset_size = calc_subset_size(frac=frac, samples=samples_df)
    if subset_size <= 0:
        logger.error("Calculated subset size is 0; no reliability samples selected.")
        return None

    subset_indices = random.sample(list(samples_df.index), k=subset_size)

    df_subset = samples_df.loc[subset_indices].copy()
    df_all = samples_df.copy()

    df_all[SELECTED_COL] = 0
    df_all.loc[subset_indices, SELECTED_COL] = 1

    if "file" in df_subset.columns:
        df_subset = df_subset.sort_values("file").reset_index(drop=True)
    df_all = df_all.sort_values("file").reset_index(drop=True) if "file" in df_all.columns else df_all

    logger.info(
        f"Selected {len(df_subset)} of {len(df_all)} samples for transcription reliability."
    )

    _write_reliability_selection_excel(df_subset, df_all, output_dir)

    if chats:
        _write_blank_reliability_chat_files(df_subset, chats, output_dir)
    elif used_transcript_tables:
        logger.info(
            "Transcript-table selection completed, but CHAT objects were not available; "
            "skipping blank reliability CHAT file creation."
        )


def reselect_transcription_reliability_samples(input_dir, output_dir, frac):
    """
    Reselect new transcription reliability samples excluding prior ones.

    Steps:
      - Locate `*transcription_reliability_samples.xlsx` files in input_dir.
      - For each file, reload sheets.
      - Prefer `selected_for_reliability` in `all_transcripts` to identify
        prior selections; fall back to the `reliability_selection` sheet.
      - Draw n = target subset size from remaining candidates.
      - Save to `output_dir/reselected_transcription_reliability/`.

    Parameters
    ----------
    input_dir : Path
        Directory with existing reliability Excel files.
    output_dir : Path
        Destination for reselected outputs.
    frac : float
        Fraction of files to select (0 < frac ≤ 1).
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    reselect_dir = output_dir / "reselected_transcription_reliability"
    reselect_dir.mkdir(parents=True, exist_ok=True)

    transc_sel_files = list(input_dir.rglob("*transcription_reliability_samples.xlsx"))
    if not transc_sel_files:
        logger.warning(f"No reliability transcription files found in {get_rel_path(input_dir)}")
        return

    for filepath in transc_sel_files:
        try:
            xls = pd.ExcelFile(filepath)
            required_sheets = {"all_transcripts", "reliability_selection"}
            if not required_sheets <= set(xls.sheet_names):
                logger.warning(f"Skipping {get_rel_path(filepath)}: missing sheets.")
                continue

            df_all = pd.read_excel(filepath, sheet_name="all_transcripts")
            df_rel = pd.read_excel(filepath, sheet_name="reliability_selection")

            if "file" not in df_all.columns:
                logger.warning(f"Skipping {get_rel_path(filepath)}: 'file' column missing in all_transcripts.")
                continue

            if SELECTED_COL in df_all.columns:
                used_files = set(df_all.loc[df_all[SELECTED_COL] == 1, "file"])
            else:
                if "file" not in df_rel.columns:
                    logger.warning(
                        f"Skipping {get_rel_path(filepath)}: no '{SELECTED_COL}' column in all_transcripts "
                        "and 'file' column missing in reliability_selection."
                    )
                    continue
                used_files = set(df_rel["file"])

            candidates = df_all[~df_all["file"].isin(used_files)].copy()

            if candidates.empty:
                logger.info(f"No remaining candidates in {get_rel_path(filepath)}, skipping.")
                continue

            n_target = calc_subset_size(frac=frac, samples=df_all)
            n_samples = min(n_target, len(candidates))

            if n_samples < n_target:
                logger.warning(
                    f"Only {n_samples}/{n_target} candidates available for {get_rel_path(filepath)} "
                    f"(candidates exhausted; cannot meet frac={frac})."
                )

            sample_df = candidates.sample(n=n_samples).copy()
            sample_df[SELECTED_COL] = 1

            outpath = reselect_dir / f"reselected_{filepath.name}"

            with pd.ExcelWriter(outpath, engine="openpyxl") as writer:
                sample_df.to_excel(
                    writer,
                    index=False,
                    sheet_name="reselected_reliability",
                )

            logger.info(f"Reselected {n_samples} files → {get_rel_path(outpath)}")

        except Exception as e:
            logger.error(f"Failed to reselect samples for {get_rel_path(filepath)}: {e}")