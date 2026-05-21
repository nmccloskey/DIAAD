from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from psair.core.logger import logger, get_rel_path
from diaad.metadata.discovery import find_one_matching_file
from diaad.metadata.unblinding import maybe_unblind_dataframe


TURN_AGG_COLS = [
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

TURN_TYPES = ["T", "MT", "ST", "NV"]


def number_turns(turn_type_col: pd.Series) -> list[str]:
    """
    Assign sequential labels to POWERS turn types.

    Valid turn types receive labels such as T1, MT2, ST3, and NV1. Invalid or
    blank rows inherit the prior label when possible; otherwise they are marked X.
    """
    labels: list[str] = []
    counts = {t: 0 for t in TURN_TYPES}

    for value in turn_type_col:
        if value not in counts:
            if labels:
                labels.append(labels[-1])
            else:
                logger.warning("First turn_type value was blank/invalid; marking as X.")
                labels.append("X")
            continue

        counts[value] += 1
        labels.append(f"{value}{counts[value]}")

    return labels


def count_value(val):
    """Return a groupby aggregation function counting values equal to `val`."""
    def inner(series):
        return np.sum(series == val)
    return inner


def add_turn_labels(utt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert sequential `turn_label` values immediately after `turn_type`.
    """
    if "turn_type" not in utt_df.columns:
        raise KeyError("Missing required column: turn_type")

    df = utt_df.copy()
    insert_at = df.columns.get_loc("turn_type") + 1

    if "turn_label" in df.columns:
        df = df.drop(columns=["turn_label"])

    df.insert(insert_at, "turn_label", number_turns(df["turn_type"]))
    return df


def compute_level_summaries(
    utt_df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> dict[str, pd.DataFrame]:
    """
    Build utterance-, turn-, speaker-, and dialog-level POWERS summaries.
    """
    return {
        "Utterances": utt_df,
        "Turns": _compute_turn_summary(utt_df, sample_id_field=sample_id_field),
        "Speakers": _compute_speaker_summary(utt_df, sample_id_field=sample_id_field),
        "Dialogs": _compute_dialog_summary(utt_df, sample_id_field=sample_id_field),
    }


def _compute_turn_summary(
    utt_df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Aggregate utterances to the turn level.
    """
    group_cols = [
        col for col in [sample_id_field, "speaker", "turn_label"]
        if col in utt_df.columns
    ]
    if len(group_cols) < 3:
        raise KeyError(
            f"Turn summary requires {sample_id_field}, speaker, and turn_label."
        )

    agg_map = {
        f"{col}_sum": (col, "sum")
        for col in TURN_AGG_COLS
        if col in utt_df.columns
    }

    return (
        utt_df.groupby(group_cols)
        .agg(**agg_map)
        .reset_index()
    )


def _compute_speaker_summary(
    utt_df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Aggregate utterances to the speaker level and add derived metrics.
    """
    if not {sample_id_field, "speaker"}.issubset(utt_df.columns):
        raise KeyError(f"Speaker summary requires {sample_id_field} and speaker.")

    agg_map = {
        f"{col}_sum": (col, "sum")
        for col in TURN_AGG_COLS
        if col in utt_df.columns
    }

    if "turn_label" in utt_df.columns:
        agg_map["total_turns"] = ("turn_label", "nunique")

    if "turn_type" in utt_df.columns:
        for ttype in TURN_TYPES:
            agg_map[f"num_{ttype}"] = ("turn_type", count_value(ttype))

    speaker_df = (
        utt_df.groupby([sample_id_field, "speaker"])
        .agg(**agg_map)
        .reset_index()
    )

    _add_speaker_derived_metrics(speaker_df)
    return speaker_df


def _add_speaker_derived_metrics(speaker_df: pd.DataFrame) -> None:
    """
    Add mean turn length and ratio-based speaker metrics in place.
    """
    if {"speech_units_sum", "total_turns"}.issubset(speaker_df.columns):
        speaker_df["mean_turn_length"] = (
            speaker_df["speech_units_sum"]
            / speaker_df["total_turns"].replace(0, np.nan)
        )

    numerator_cols = {
        "num_nouns": "num_nouns_sum",
        "content_words": "content_words_sum",
    }
    denominator_cols = {
        "speech_units": "speech_units_sum",
        "total_turns": "total_turns",
        "num_ST": "num_ST",
    }

    for num_name, num_col in numerator_cols.items():
        if num_col not in speaker_df.columns:
            continue

        for denom_name, denom_col in denominator_cols.items():
            if denom_col not in speaker_df.columns:
                continue

            out_col = f"ratio_{num_name}_to_{denom_name}"
            speaker_df[out_col] = (
                speaker_df[num_col] / speaker_df[denom_col].replace(0, np.nan)
            )

    for ttype in ["ST", "MT"]:
        num_col = f"num_{ttype}"
        if {num_col, "total_turns"}.issubset(speaker_df.columns):
            speaker_df[f"ratio_{ttype}s_to_turns"] = (
                speaker_df[num_col] / speaker_df["total_turns"].replace(0, np.nan)
            )


def _compute_dialog_summary(
    utt_df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Aggregate utterances to the dialog/sample level.
    """
    if sample_id_field not in utt_df.columns:
        raise KeyError(f"Dialog summary requires {sample_id_field}.")

    agg_map = {
        f"{col}_sum": (col, "sum")
        for col in TURN_AGG_COLS
        if col in utt_df.columns
    }

    if "collab_repair" in utt_df.columns:
        agg_map["num_repairs"] = ("collab_repair", "nunique")
        agg_map["prop_repairs"] = ("collab_repair", lambda x: x.notna().mean())

    return utt_df.groupby(sample_id_field).agg(**agg_map).reset_index()


def write_analysis_workbook(out_path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    """
    Write multiple dataframes to one Excel workbook.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for sheet_name, df in sheets.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)


def analyze_powers_coding(
    input_dir,
    output_dir,
    powers_coding_file="powers_coding.xlsx",
    blinding_config=None,
    blind_codebook=None,
    sample_id_field="sample_id",
):
    """
    Analyze one unprefixed POWERS coding file and write summary workbooks.

    The exact configured powers_coding filename is read, assigned sequential
    turn labels, summarized to turn/speaker/dialog levels, and written to
    <output_dir>/powers_coding_analysis/.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    out_dir = output_dir / "powers_coding_analysis"

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {get_rel_path(out_dir)}")
    except Exception as e:
        logger.error(f"Failed to create POWERS analysis directory {get_rel_path(out_dir)}: {e}")
        return

    pc_file = find_one_matching_file(
        directories=[input_dir, output_dir],
        filename=powers_coding_file,
        label="POWERS coding file",
    )

    utt_df = _read_powers_file(pc_file)
    if utt_df is None:
        return

    if blinding_config is not None:
        utt_df, _ = maybe_unblind_dataframe(
            utt_df,
            blinding_config,
            blind_codebook=blind_codebook,
            target_cols=[sample_id_field],
            directories=[input_dir, output_dir],
            strict=False,
        )

    try:
        utt_df = add_turn_labels(utt_df)
        sheets = compute_level_summaries(
            utt_df,
            sample_id_field=sample_id_field,
        )
    except Exception as e:
        logger.error(f"Failed to analyze {get_rel_path(pc_file)}: {e}")
        return

    out_file = out_dir / f"{pc_file.stem.replace('coding', 'analysis')}.xlsx"

    try:
        write_analysis_workbook(out_file, sheets)
        logger.info(f"Wrote analysis workbook: {get_rel_path(out_file)}")
    except Exception as e:
        logger.error(f"Failed to write {get_rel_path(out_file)}: {e}")


def _read_powers_file(pc_file: Path) -> pd.DataFrame | None:
    """
    Read one POWERS coding workbook.
    """
    try:
        df = pd.read_excel(pc_file)
        logger.info(f"Processing file: {get_rel_path(pc_file)}")
        return df
    except Exception as e:
        logger.error(f"Failed to read file {get_rel_path(pc_file)}: {e}")
        return None
