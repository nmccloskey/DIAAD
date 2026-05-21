from __future__ import annotations

import random
import pandas as pd
import numpy as np
from pathlib import Path

from psair.core.logger import logger, get_rel_path
from diaad.coding.utils import segment
from diaad.coding.utils.sampling import calc_subset_size
from diaad.metadata.discovery import find_one_matching_file
from diaad.transcripts.transcript_tables import extract_transcript_data
from diaad.metadata.blinding import blind_file_identifiers, write_blind_codebook
from diaad.coding.powers.automation import run_automation


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SECTION_C_cols = [
    "content_words",
    "num_nouns",
    "circumlocutions",
    "sem_paras",
    "phon_errs",
    "neologisms",
    "comments",
    "lg_pauses",
    "filled_pauses",
]

POWERS_cols = [
    # Meta
    "POWERS_comment",
    # Section A
    "speech_units",
    # Section B
    "turn_type",
] + SECTION_C_cols + [
    # Section D
    "collab_repair",
]

COMM_cols = [
    "communication",
    "topic",
    "subject",
    "dialogue",
    "conversation",
]

TT_DROP_COLS = [
    "file",
    "file_ext",
    "file_dir",
    "input_order",
    "shuffled_order",
    "position",
    "position_sub",
]

SECTION_E_cols = [
    "type_of_day",
    "amount_of_enjoyment",
    "degree_of_difficulty",
    "other_notes",
]


# ---------------------------------------------------------------------
# POWERS file generation helpers
# ---------------------------------------------------------------------

def _get_transcript_table(input_dir, output_dir) -> Path | None:
    """
    Locate the transcript table used to generate POWERS files.
    """
    return find_one_matching_file(
        directories=[input_dir, output_dir],
        filename="transcript_tables.xlsx",
        label="transcript table file",
    )


def _load_utterance_table(
    transcript_table: Path,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame | None:
    """
    Load utterance-level transcript data from a transcript table file.
    """
    try:
        return extract_transcript_data(
            transcript_table,
            sample_id_field=sample_id_field,
        )
    except Exception as e:
        logger.error(f"Failed to read transcript table {get_rel_path(transcript_table)}: {e}")
        return None


def _match_metadata_field_values(metadata_fields, source_path: str) -> list[str]:
    """
    Extract metadata values from a path using metadata field matchers.
    """
    path = Path(source_path)
    parts = [part for part in path.parts if part not in ("", ".")]
    values = []
    for field in metadata_fields.values():
        if hasattr(field, "match_path_parts"):
            value = field.match_path_parts(parts, return_none=True, source=str(path))
        else:
            value = field.match(str(path), return_none=True)
        if value is not None:
            values.append(value)
    return values


def _validate_identifier_columns(
    df: pd.DataFrame,
    sample_id_field: str,
    utterance_id_field: str,
) -> None:
    """Raise a clear error when configured POWERS identifier columns are absent."""
    missing = [
        col
        for col in (sample_id_field, utterance_id_field)
        if col not in df.columns
    ]
    if missing:
        raise KeyError(f"Missing required POWERS identifier columns: {missing}")


def _prepare_powers_dataframe(
    uttdf: pd.DataFrame,
    metadata_fields,
    exclude_participants,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Shuffle by sample, drop non-POWERS transcript columns, and initialize fields.
    """
    shuffled = _shuffle_by_sample(uttdf, sample_id_field=sample_id_field)
    drop_cols = _get_powers_drop_cols(shuffled, metadata_fields)
    df = shuffled.drop(columns=drop_cols).copy()

    logger.info(f"Transcript columns: {list(shuffled.columns)}")
    logger.info(f"Final drop cols: {drop_cols}")

    df["coder_id"] = ""
    for col in POWERS_cols:
        if col in SECTION_C_cols:
            df[col] = np.where(df["speaker"].isin(exclude_participants), "NA", "")
        else:
            df[col] = ""

    return df


def _shuffle_by_sample(
    df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Shuffle transcript rows at the sample level.
    """
    if sample_id_field not in df.columns:
        raise KeyError(f"Missing required sample identifier column: {sample_id_field}")

    subdfs = [subdf for _, subdf in df.groupby(sample_id_field, sort=False)]
    random.shuffle(subdfs)
    if not subdfs:
        return df.copy()
    return pd.concat(subdfs, ignore_index=True)


def _get_powers_drop_cols(df: pd.DataFrame, metadata_fields) -> list[str]:
    """
    Identify transcript-table columns to drop before POWERS export.
    """
    non_comm_metadata_cols = [
        field.name for field in metadata_fields.values()
        if field.name.lower() not in COMM_cols
    ]
    return [
        col for col in TT_DROP_COLS + non_comm_metadata_cols
        if col in df.columns
    ]


def _resolve_powers_coder_ids(num_coders: int) -> list[str]:
    """
    Resolve POWERS coder IDs from num_coders.
    """
    if num_coders <= 0:
        return [""]
    return [str(i) for i in range(1, num_coders + 1)]


def _assign_primary_coders(
    df: pd.DataFrame,
    coder_ids: list[str],
    sample_id_field: str = "sample_id",
) -> tuple[pd.DataFrame, dict[str, str], list[list[str]]]:
    """
    Assign primary coders across samples and populate `coder_id`.
    """
    if sample_id_field not in df.columns:
        raise KeyError(f"Missing required sample identifier column: {sample_id_field}")

    sample_ids = list(df[sample_id_field].drop_duplicates())
    if not sample_ids:
        logger.warning("No sample_ids found in transcript table.")
        return df, {}, []

    segments = segment(sample_ids, n=len(coder_ids))
    primary_assignment: dict[str, str] = {}

    for seg, coder_id in zip(segments, coder_ids):
        for sample_id in seg:
            primary_assignment[sample_id] = coder_id

    df["coder_id"] = df[sample_id_field].map(primary_assignment).fillna("")
    return df, primary_assignment, segments


def _build_reliability_dataframe(
    pc_df: pd.DataFrame,
    frac: float,
    coder_ids: list[str],
    primary_assignment: dict[str, str],
    segments: list[list[str]],
    sample_id_field: str = "sample_id",
) -> pd.DataFrame | None:
    """
    Build a reliability subset dataframe when frac > 0.
    """
    if frac == 0:
        logger.info("frac=0 detected; no reliability subset will be generated.")
        return None

    rel_samples = _select_reliability_samples(segments, frac)
    if not rel_samples:
        logger.info("No reliability samples were selected.")
        return None

    if sample_id_field not in pc_df.columns:
        raise KeyError(f"Missing required sample identifier column: {sample_id_field}")

    rel_df = pc_df[pc_df[sample_id_field].isin(rel_samples)].copy()
    rel_assignment = _rotate_reliability_assignments(primary_assignment, coder_ids)
    rel_df["coder_id"] = rel_df[sample_id_field].map(rel_assignment).fillna("")

    logger.info(
        f"Selected {len(set(rel_df[sample_id_field]))} samples for reliability "
        f"from {len(set(pc_df[sample_id_field]))} total samples."
    )
    return rel_df


def _select_reliability_samples(segments: list[list[str]], frac: float) -> list[str]:
    """
    Sample reliability items within each primary coder segment.
    """
    rel_samples: list[str] = []
    for seg in segments:
        if not seg:
            continue
        n_rel = calc_subset_size(frac=frac, samples=seg)
        if n_rel == 0:
            continue
        rel_samples.extend(random.sample(seg, k=n_rel))
    return rel_samples


def _rotate_reliability_assignments(
    primary_map: dict[str, str],
    coder_ids: list[str],
) -> dict[str, str]:
    """
    Derive reliability coder IDs from primary assignment.
    """
    if not coder_ids or coder_ids == [""]:
        return {sample_id: "" for sample_id in primary_map}

    if len(coder_ids) == 1:
        only_id = coder_ids[0]
        return {sample_id: only_id for sample_id in primary_map}

    idx = {coder_id: i for i, coder_id in enumerate(coder_ids)}
    return {
        sample_id: coder_ids[(idx[primary_coder] + 1) % len(coder_ids)]
        for sample_id, primary_coder in primary_map.items()
    }


def _apply_export_blinding(
    pc_df: pd.DataFrame,
    rel_df: pd.DataFrame | None,
    transcript_table: Path,
    blinding_config=None,
    input_dir=None,
    output_dir=None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Blind POWERS exports at write time when configured.
    """
    codebook_df = None
    if blinding_config is None or not blinding_config.should_blind("coding"):
        return pc_df, rel_df, codebook_df

    try:
        pc_export, codebook_df = blind_file_identifiers(
            pc_df,
            config=blinding_config,
            directories=[input_dir, output_dir],
        )
        rel_export = None
        if rel_df is not None:
            rel_export, _ = blind_file_identifiers(
                rel_df,
                config=blinding_config,
                existing_codebook=codebook_df,
            )

        logger.info("Applied file blinding to POWERS exports.")
        return pc_export, rel_export, codebook_df

    except Exception as e:
        logger.error(f"Failed to apply file blinding to POWERS exports: {e}")
        raise


def _write_powers_exports(
    pc_export: pd.DataFrame,
    rel_export: pd.DataFrame | None,
    codebook_df: pd.DataFrame | None,
    powers_dir: Path,
    metadata_values: list[str],
    powers_coding_file: str = "powers_coding.xlsx",
    powers_reliability_file: str = "powers_reliability_coding.xlsx",
    sample_id_field: str = "sample_id",
) -> None:
    """
    Write POWERS coding and reliability workbooks to disk.
    """
    pc_filename = Path(powers_dir, *metadata_values, powers_coding_file)
    rel_filename = Path(powers_dir, *metadata_values, powers_reliability_file)

    _write_primary_powers_workbook(
        pc_export,
        pc_filename,
        sample_id_field=sample_id_field,
    )
    if rel_export is not None:
        _write_excel(rel_export, rel_filename, "POWERS reliability coding")
    if codebook_df is not None and not codebook_df.empty:
        write_blind_codebook(
            codebook_df,
            Path(powers_dir, *metadata_values, "powers_blind_codebook.xlsx"),
        )


def _build_section_e_dataframe(
    df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Build the sample-level Section E sheet for primary POWERS coding.
    """
    if sample_id_field not in df.columns:
        raise KeyError(f"Missing required sample identifier column: {sample_id_field}")

    section_e = df[[sample_id_field]].drop_duplicates().copy()
    for col in SECTION_E_cols:
        section_e[col] = ""
    return section_e


def _write_primary_powers_workbook(
    df: pd.DataFrame,
    filename: Path,
    sample_id_field: str = "sample_id",
) -> None:
    """
    Write primary POWERS coding workbook with utterance and Section E sheets.
    """
    try:
        filename.parent.mkdir(parents=True, exist_ok=True)
        section_e = _build_section_e_dataframe(df, sample_id_field=sample_id_field)
        with pd.ExcelWriter(filename, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="utterance_coding", index=False, na_rep="")
            section_e.to_excel(writer, sheet_name="section_e", index=False, na_rep="")
        logger.info(f"Successfully wrote POWERS coding file: {get_rel_path(filename)}")
    except Exception as e:
        logger.error(f"Failed to write POWERS coding file {get_rel_path(filename)}: {e}")


def _write_excel(df: pd.DataFrame, filename: Path, label: str) -> None:
    """
    Write a dataframe to Excel with consistent logging.
    """
    try:
        filename.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(filename, index=False, na_rep="")
        logger.info(f"Successfully wrote {label} file: {get_rel_path(filename)}")
    except Exception as e:
        logger.error(f"Failed to write {label} file {get_rel_path(filename)}: {e}")


def make_powers_coding_files(
    metadata_fields,
    frac,
    num_coders,
    input_dir,
    output_dir,
    exclude_participants,
    automate_powers=True,
    blinding_config=None,
    powers_coding_file="powers_coding.xlsx",
    powers_reliability_file="powers_reliability_coding.xlsx",
    sample_id_field="sample_id",
    utterance_id_field="utterance_id",
):
    """
    Build POWERS coding and reliability workbooks from an utterance table.

    One unprefixed coding layer is used per file. Coder assignment is stored
    only in `coder_id`; there are no c1_/c2_ column blocks. If `frac > 0`,
    a reliability subset is also written. For one coder, reliability keeps
    the same coder ID; for multiple coders, reliability rotates assignment
    to the next coder. Blinding, when enabled, is applied only at export.

    DIAAD requires exactly one transcript_tables.xlsx file for this workflow.
    """
    powers_dir = Path(output_dir) / "powers_coding"
    powers_dir.mkdir(parents=True, exist_ok=True)

    transcript_table = _get_transcript_table(input_dir, output_dir)
    if transcript_table is None:
        return

    uttdf = _load_utterance_table(
        transcript_table,
        sample_id_field=sample_id_field,
    )
    if uttdf is None:
        return

    metadata_values = _match_metadata_field_values(metadata_fields, str(transcript_table))
    _validate_identifier_columns(
        uttdf,
        sample_id_field=sample_id_field,
        utterance_id_field=utterance_id_field,
    )
    pc_df = _prepare_powers_dataframe(
        uttdf,
        metadata_fields,
        exclude_participants,
        sample_id_field=sample_id_field,
    )

    if automate_powers:
        pc_df = run_automation(pc_df)

    coder_ids = _resolve_powers_coder_ids(num_coders)
    pc_df, primary_assignment, segments = _assign_primary_coders(
        pc_df,
        coder_ids,
        sample_id_field=sample_id_field,
    )

    rel_df = _build_reliability_dataframe(
        pc_df=pc_df,
        frac=frac,
        coder_ids=coder_ids,
        primary_assignment=primary_assignment,
        segments=segments,
        sample_id_field=sample_id_field,
    )

    pc_export, rel_export, codebook_df = _apply_export_blinding(
        pc_df=pc_df,
        rel_df=rel_df,
        transcript_table=transcript_table,
        blinding_config=blinding_config,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    _write_powers_exports(
        pc_export=pc_export,
        rel_export=rel_export,
        codebook_df=codebook_df,
        powers_dir=powers_dir,
        metadata_values=metadata_values,
        powers_coding_file=powers_coding_file,
        powers_reliability_file=powers_reliability_file,
        sample_id_field=sample_id_field,
    )
