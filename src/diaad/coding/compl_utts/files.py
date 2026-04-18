import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from psair.core.logger import logger, get_rel_path
from src.diaad.coding.utils.sampling import calc_subset_size
from psair.metadata.discovery import find_matching_files
from diaad.transcripts.transcript_tables import extract_transcript_data
from diaad.coding.utils import segment, assign_coders, resolve_stim_cols
from diaad.metadata.blinding import blind_file_identifiers, write_blind_codebook


def _coder_ids(num_coders: int) -> list[int]:
    """Return canonical integer coder IDs: 1..num_coders."""
    return list(range(1, max(0, int(num_coders)) + 1))


def _resolve_coder_mode(num_coders: int) -> tuple[str, list[int]]:
    """
    Resolve workflow mode from num_coders.

    Modes
    -----
    zero:
        Blank primary coding file, optional blank reliability subset.
    single:
        One coder in primary file, optional blank reliability subset.
    two:
        Two coders split primary assignments; reliability goes to opposite coder.
    three:
        3+ coder workflow using c1/c2 primary + c3 reliability.
        If num_coders > 3, only coder IDs 1, 2, 3 are used.
    """
    coder_ids = _coder_ids(num_coders)

    if num_coders <= 0:
        return "zero", []
    if num_coders == 1:
        return "single", coder_ids
    if num_coders == 2:
        return "two", coder_ids

    if num_coders > 3:
        logger.warning(
            f"num_coders={num_coders} detected for CU coding. "
            "Using the 3-coder workflow with coder IDs [1, 2, 3]; additional coders will be ignored."
        )

    return "three", [1, 2, 3]


def _assign_single_coding_columns(
    df: pd.DataFrame,
    cu_paradigms,
    exclude_participants,
) -> pd.DataFrame:
    """Add one unprefixed coding layer."""
    base_cols = ["id", "sv", "rel", "comment"]
    for col in base_cols:
        df[col] = np.where(df["speaker"].isin(exclude_participants), "NA", "")

    if len(cu_paradigms) < 2:
        return df

    for tag in ["sv", "rel"]:
        df.drop(columns=[tag], inplace=True, errors="ignore")
        for paradigm in cu_paradigms:
            new_col = f"{tag}_{paradigm}"
            df[new_col] = np.where(df["speaker"].isin(exclude_participants), "NA", "")

    return df


def _assign_three_coding_columns(
    df: pd.DataFrame,
    cu_paradigms,
    exclude_participants,
) -> pd.DataFrame:
    """Add c1/c2 coding layers."""
    base_cols = [
        "c1_id", "c1_sv", "c1_rel", "c1_comment",
        "c2_id", "c2_sv", "c2_rel", "c2_comment",
    ]
    for col in base_cols:
        df[col] = np.where(df["speaker"].isin(exclude_participants), "NA", "")

    if len(cu_paradigms) < 2:
        return df

    for prefix in ["c1", "c2"]:
        for tag in ["sv", "rel"]:
            df.drop(columns=[f"{prefix}_{tag}"], inplace=True, errors="ignore")
            for paradigm in cu_paradigms:
                new_col = f"{prefix}_{tag}_{paradigm}"
                df[new_col] = np.where(
                    df["speaker"].isin(exclude_participants),
                    "NA",
                    "",
                )

    return df


def _prepare_blank_reliability_subset(
    cu_df: pd.DataFrame,
    sample_ids: list,
    frac: float,
) -> pd.DataFrame | None:
    """Build a blank reliability subset with no assigned coder."""
    n_rel_samples = calc_subset_size(frac=frac, samples=sample_ids)
    if n_rel_samples == 0:
        return None

    rel_samples = random.sample(sample_ids, k=n_rel_samples)
    return cu_df[cu_df["sample_id"].isin(rel_samples)].copy()


def _prepare_two_coder_reliability_subset(
    cu_df: pd.DataFrame,
    seg: list,
    rel_coder: int,
    frac: float,
) -> pd.DataFrame | None:
    """Build a 2-coder reliability subset from one coder's primary assignments."""
    n_rel_samples = calc_subset_size(frac=frac, samples=seg)
    if n_rel_samples == 0:
        return None

    rel_samples = random.sample(seg, k=n_rel_samples)
    relsegdf = cu_df[cu_df["sample_id"].isin(rel_samples)].copy()
    relsegdf["id"] = rel_coder
    return relsegdf


def _prepare_three_coder_reliability_subset(
    cu_df: pd.DataFrame,
    seg: list,
    assn: list[int],
    frac: float,
    cu_paradigms,
) -> pd.DataFrame | None:
    """Build the reliability subset for the 3-coder workflow."""
    n_rel_samples = calc_subset_size(frac=frac, samples=seg)
    if n_rel_samples == 0:
        return None

    rel_samples = random.sample(seg, k=n_rel_samples)
    relsegdf = cu_df[cu_df["sample_id"].isin(rel_samples)].copy()
    relsegdf.drop(columns=["c1_id", "c1_comment"], inplace=True, errors="ignore")

    if len(cu_paradigms) >= 2:
        for tag in ["sv", "rel"]:
            for paradigm in cu_paradigms:
                old = f"c2_{tag}_{paradigm}"
                new = f"c3_{tag}_{paradigm}"
                if old in relsegdf.columns:
                    relsegdf.rename(columns={old: new}, inplace=True)
                relsegdf.drop(
                    columns=[f"c1_{tag}_{paradigm}"],
                    inplace=True,
                    errors="ignore",
                )
        relsegdf.rename(columns={"c2_comment": "c3_comment"}, inplace=True)

    else:
        renames = {
            "c2_sv": "c3_sv",
            "c2_rel": "c3_rel",
            "c2_comment": "c3_comment",
        }
        relsegdf.rename(
            columns={k: v for k, v in renames.items() if k in relsegdf.columns},
            inplace=True,
        )
        relsegdf.drop(columns=["c1_sv", "c1_rel"], inplace=True, errors="ignore")

        for col in ["c3_sv", "c3_rel", "c3_comment"]:
            if col not in relsegdf.columns:
                relsegdf[col] = ""

    relsegdf.drop(columns=["c2_id"], inplace=True, errors="ignore")
    relsegdf.insert(relsegdf.columns.get_loc("c3_comment"), "c3_id", assn[2])

    return relsegdf


def _prepare_cu_base_dataframe(
    uttdf: pd.DataFrame,
    tiers,
    stimulus_field,
) -> pd.DataFrame:
    """
    Shuffle sample blocks and drop non-coding columns from an utterance dataframe.
    """
    stim_cols = resolve_stim_cols(stimulus_field)

    subdfs = [subdf for _, subdf in uttdf.groupby(by="sample_id")]
    random.shuffle(subdfs)
    shuffled_utt_df = pd.concat(subdfs, ignore_index=True)

    drop_cols = [
        col
        for col in ["file", "speaking_time"] + [t for t in tiers if t.lower() not in stim_cols]
        if col in shuffled_utt_df.columns
    ]

    return shuffled_utt_df.drop(columns=drop_cols).copy()


def _build_cu_assignments(
    cu_df: pd.DataFrame,
    *,
    mode: str,
    coder_ids: list[int],
    frac: float,
    cu_paradigms,
    exclude_participants,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Populate coder assignment columns and construct the reliability dataframe.
    """
    if mode == "three":
        cu_df = _assign_three_coding_columns(cu_df, cu_paradigms, exclude_participants)
    else:
        cu_df = _assign_single_coding_columns(cu_df, cu_paradigms, exclude_participants)

    unique_ids = list(cu_df["sample_id"].drop_duplicates())
    rel_subsets: list[pd.DataFrame] = []

    if mode in {"single", "zero"}:
        coder = coder_ids[0] if coder_ids else ""
        cu_df["id"] = np.where(
            cu_df["speaker"].isin(exclude_participants),
            "NA",
            coder,
        )

        if frac > 0:
            rel_subsets.append(_prepare_blank_reliability_subset(cu_df, unique_ids, frac))

    elif mode == "two":
        segments = segment(unique_ids, n=2)

        for seg, coder in zip(segments, coder_ids):
            cu_df.loc[cu_df["sample_id"].isin(seg), "id"] = coder

        if frac > 0:
            rel_subsets.extend(
                [
                    _prepare_two_coder_reliability_subset(cu_df, segments[0], coder_ids[1], frac),
                    _prepare_two_coder_reliability_subset(cu_df, segments[1], coder_ids[0], frac),
                ]
            )

    else:
        segments = segment(unique_ids, n=len(coder_ids))
        assignments = assign_coders(coder_ids)

        for seg, assn in zip(segments, assignments):
            cu_df.loc[cu_df["sample_id"].isin(seg), ["c1_id", "c2_id"]] = assn[:2]

            if frac > 0:
                rel_subsets.append(
                    _prepare_three_coder_reliability_subset(
                        cu_df,
                        seg,
                        assn,
                        frac,
                        cu_paradigms,
                    )
                )

    rel_subsets = [df for df in rel_subsets if df is not None]
    rel_df = pd.concat(rel_subsets, ignore_index=True) if rel_subsets else None

    return cu_df, rel_df


def _blind_coding_exports(
    cu_df: pd.DataFrame,
    rel_df: pd.DataFrame | None,
    *,
    blinding_config=None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Apply file-identifier blinding to CU and reliability exports.

    Returns
    -------
    export_cu_df
        CU coding dataframe, possibly blinded.
    export_rel_df
        Reliability dataframe, possibly blinded.
    codebook_df
        Codebook used for blinding, or None if blinding was not applied.
    """
    export_cu_df = cu_df.copy()
    export_rel_df = rel_df.copy() if rel_df is not None else None
    codebook_df = None

    if blinding_config is None or not blinding_config.should_blind("coding"):
        return export_cu_df, export_rel_df, codebook_df

    export_cu_df, codebook_df = blind_file_identifiers(
        export_cu_df,
        config=blinding_config,
    )

    if export_rel_df is not None:
        export_rel_df, _ = blind_file_identifiers(
            export_rel_df,
            config=blinding_config,
            existing_codebook=codebook_df,
        )

    return export_cu_df, export_rel_df, codebook_df


def _write_cu_outputs(
    *,
    export_cu_df: pd.DataFrame,
    export_rel_df: pd.DataFrame | None,
    codebook_df: pd.DataFrame | None,
    cu_coding_dir: Path,
    source_name: str,
    total_unique_ids: int,
) -> None:
    """
    Write CU coding outputs, reliability outputs, and optional blind codebook.
    """
    export_cu_df.to_excel(cu_coding_dir / "cu_coding.xlsx", index=False)

    if export_rel_df is not None:
        logger.info(
            f"{source_name}: reliability={export_rel_df['sample_id'].nunique()} / total={total_unique_ids}"
        )
        export_rel_df.to_excel(
            cu_coding_dir / "cu_reliability_coding.xlsx",
            index=False,
        )
    else:
        logger.info(f"{source_name}: no reliability subset generated.")

    if codebook_df is not None and not codebook_df.empty:
        write_blind_codebook(
            codebook_df,
            cu_coding_dir / "cu_blind_codebook.xlsx",
        )


def make_cu_coding_files(
    tiers,
    frac,
    num_coders,
    input_dir,
    output_dir,
    cu_paradigms,
    exclude_participants,
    stimulus_field,
    blinding_config=None,
):
    """
    Build CU coding and reliability workbooks from an utterance table.

    Modes
    -----
    0 coders:
        One unprefixed coding layer (id, sv, rel, comment) with blank coder IDs.
        If frac > 0, a blank reliability subset may also be generated.

    1 coder:
        One unprefixed coding layer with coder ID 1 in the id column.
        Reliability, if requested, is generated as a blank subset rather than as an
        independent coder assignment.

    2 coders:
        One unprefixed coding layer. Samples are split across coders 1 and 2;
        reliability subset is reassigned to the opposite coder.

    3+ coders:
        Current c1/c2 primary coding workflow plus c3 reliability workflow.
        If num_coders > 3, only coder IDs 1, 2, and 3 are used.

    Notes
    -----
    frac == 0 means no reliability subset is generated.
    stimulus_field overrides legacy stimulus-column fallback behavior.

    Transcript table selection
    --------------------------
    If multiple transcript tables are detected, only the first returned match is processed.

    Blinding
    --------
    Blinding is applied only at export time, after coder assignment and reliability
    subset selection, so all internal workflow logic continues to operate on the
    original identifiers.
    """
    cu_coding_dir = Path(output_dir) / "cu_coding"
    cu_coding_dir.mkdir(parents=True, exist_ok=True)

    mode, coder_ids = _resolve_coder_mode(num_coders)

    if frac == 0:
        logger.info("frac=0 detected; no reliability subset will be generated.")

    transcript_tables = find_matching_files(
        directories=[input_dir, output_dir],
        search_base="transcript_tables",
    )

    if not transcript_tables:
        logger.error("No transcript_tables file found.")
        return

    if len(transcript_tables) > 1:
        logger.warning(
            "Multiple transcript tables detected. "
            f"Processing only the first returned file: {get_rel_path(transcript_tables[0])}"
        )

    transcript_table = transcript_tables[0]

    try:
        uttdf = extract_transcript_data(transcript_table)

        cu_df = _prepare_cu_base_dataframe(
            uttdf=uttdf,
            tiers=tiers,
            stimulus_field=stimulus_field,
        )

        cu_df, rel_df = _build_cu_assignments(
            cu_df,
            mode=mode,
            coder_ids=coder_ids,
            frac=frac,
            cu_paradigms=cu_paradigms,
            exclude_participants=exclude_participants,
        )

        export_cu_df, export_rel_df, codebook_df = _blind_coding_exports(
            cu_df,
            rel_df,
            blinding_config=blinding_config,
        )

        _write_cu_outputs(
            export_cu_df=export_cu_df,
            export_rel_df=export_rel_df,
            codebook_df=codebook_df,
            cu_coding_dir=cu_coding_dir,
            source_name=transcript_table.name,
            total_unique_ids=cu_df["sample_id"].nunique(),
        )

    except Exception as e:
        logger.error(f"Failed processing {get_rel_path(transcript_table)}: {e}")
