import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from diaad.utils.logger import logger, _rel
from diaad.utils.sampling import calc_subset_size
from diaad.io.discovery import find_matching_files
from diaad.transcripts.tables import extract_transcript_data
from diaad.coding.utils import segment, assign_coders, normalize_coders, resolve_stim_cols


def _assign_single_coding_columns(df, cu_paradigms, exclude_participants):
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


def _assign_three_coding_columns(df, cu_paradigms, exclude_participants):
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
                df[new_col] = np.where(df["speaker"].isin(exclude_participants), "NA", "")
    return df


def _prepare_blank_reliability_subset(cu_df, sample_ids, frac):
    """Build a blank reliability subset with no assigned coder."""
    n_rel_samples = calc_subset_size(frac=frac, samples=sample_ids)
    if n_rel_samples == 0:
        return None

    rel_samples = random.sample(sample_ids, k=n_rel_samples)
    return cu_df[cu_df["sample_id"].isin(rel_samples)].copy()


def _prepare_two_coder_reliability_subset(cu_df, seg, rel_coder, frac):
    """Build a 2-coder reliability subset from one coder's primary assignments."""
    n_rel_samples = calc_subset_size(frac=frac, samples=seg)
    if n_rel_samples == 0:
        return None

    rel_samples = random.sample(seg, k=n_rel_samples)
    relsegdf = cu_df[cu_df["sample_id"].isin(rel_samples)].copy()
    relsegdf["id"] = rel_coder
    return relsegdf


def _prepare_three_coder_reliability_subset(cu_df, seg, assn, frac, cu_paradigms):
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
                old, new = f"c2_{tag}_{paradigm}", f"c3_{tag}_{paradigm}"
                if old in relsegdf:
                    relsegdf.rename(columns={old: new}, inplace=True)
                relsegdf.drop(columns=[f"c1_{tag}_{paradigm}"], inplace=True, errors="ignore")
        relsegdf.rename(columns={"c2_comment": "c3_comment"}, inplace=True)
    else:
        renames = {"c2_sv": "c3_sv", "c2_rel": "c3_rel", "c2_comment": "c3_comment"}
        relsegdf.rename(columns={k: v for k, v in renames.items() if k in relsegdf}, inplace=True)
        relsegdf.drop(columns=["c1_sv", "c1_rel"], inplace=True, errors="ignore")
        for col in ["c3_sv", "c3_rel", "c3_comment"]:
            if col not in relsegdf:
                relsegdf[col] = ""

    relsegdf.drop(columns=["c2_id"], inplace=True, errors="ignore")
    relsegdf.insert(relsegdf.columns.get_loc("c3_comment"), "c3_id", assn[2])
    return relsegdf


def make_cu_coding_files(
    tiers,
    frac,
    coders,
    input_dir,
    output_dir,
    cu_paradigms,
    exclude_participants,
    narrative_field,
):
    """
    Build CU coding and reliability workbooks from utterance tables.

    Modes
    -----
    0 coders:
        One unprefixed coding layer (id, sv, rel, comment) with blank coder IDs.
        If frac > 0, a blank reliability subset may also be generated.
    1 coder:
        One unprefixed coding layer with that coder in the id column.
        Reliability, if requested, is generated as a blank subset rather than as an independent coder assignment.
    2 coders:
        One unprefixed coding layer. Samples are split across coders; reliability
        subset is reassigned to the opposite coder.
    3+ coders:
        Current c1/c2 primary coding workflow plus c3 reliability workflow.
        If >3 coders are provided, only the first three are used.

    Notes
    -----
    frac == 0 means no reliability subset is generated.
    narrative_field overrides legacy stimulus-column fallback behavior.
    """
    cu_coding_dir = Path(output_dir) / "cu_coding"
    cu_coding_dir.mkdir(parents=True, exist_ok=True)

    stim_cols = resolve_stim_cols(narrative_field)
    mode, coders = normalize_coders(coders)
    if frac == 0:
        logger.info("frac=0 detected; no reliability subset will be generated.")

    transcript_tables = find_matching_files(
        directories=[input_dir, output_dir],
        search_base="transcript_tables",
    )
    utt_dfs = [extract_transcript_data(tt) for tt in transcript_tables]

    for file, uttdf in tqdm(zip(transcript_tables, utt_dfs), desc="Generating CU coding files"):
        try:
            labels = [t.match(file.name, return_none=True) for t in tiers.values()]
            labels = [l for l in labels if l]
            label_path = Path(cu_coding_dir, *labels)
            label_path.mkdir(parents=True, exist_ok=True)
            lab_str = "_".join(labels) + "_" if labels else ""

            subdfs = [subdf for _, subdf in uttdf.groupby(by="sample_id")]
            random.shuffle(subdfs)
            shuffled_utt_df = pd.concat(subdfs, ignore_index=True)

            drop_cols = [
                col for col in ["file", "speaking_time"] + [t for t in tiers if t.lower() not in stim_cols]
                if col in shuffled_utt_df.columns
            ]
            cu_df = shuffled_utt_df.drop(columns=drop_cols).copy()

            if mode == "three":
                cu_df = _assign_three_coding_columns(cu_df, cu_paradigms, exclude_participants)
            else:
                cu_df = _assign_single_coding_columns(cu_df, cu_paradigms, exclude_participants)

            unique_ids = list(cu_df["sample_id"].drop_duplicates())
            rel_subsets = []

            if mode in {"single", "zero"}:
                coder = coders[0] if coders else ""
                cu_df["id"] = np.where(
                    cu_df["speaker"].isin(exclude_participants),
                    "NA",
                    coder,
                )

                if frac > 0:
                    rel_subsets.append(
                        _prepare_blank_reliability_subset(cu_df, unique_ids, frac)
                    )

            elif mode == "two":
                segments = segment(unique_ids, n=2)
                for seg, coder in zip(segments, coders):
                    cu_df.loc[cu_df["sample_id"].isin(seg), "id"] = coder

                if frac > 0:
                    rel_subsets.extend([
                        _prepare_two_coder_reliability_subset(
                            cu_df, segments[0], coders[1], frac
                        ),
                        _prepare_two_coder_reliability_subset(
                            cu_df, segments[1], coders[0], frac
                        ),
                    ])

            else:
                segments = segment(unique_ids, n=len(coders))
                assignments = assign_coders(coders)
                for seg, assn in zip(segments, assignments):
                    cu_df.loc[cu_df["sample_id"].isin(seg), ["c1_id", "c2_id"]] = assn[:2]
                    if frac > 0:
                        rel_subsets.append(
                            _prepare_three_coder_reliability_subset(
                                cu_df, seg, assn, frac, cu_paradigms
                            )
                        )

            rel_subsets = [df for df in rel_subsets if df is not None]
            if rel_subsets:
                reldf = pd.concat(rel_subsets, ignore_index=True)
                logger.info(f"{file.name}: reliability={len(set(reldf['sample_id']))} / total={len(unique_ids)}")
                reldf.to_excel(label_path / f"{lab_str}cu_reliability_coding.xlsx", index=False)
            else:
                logger.info(f"{file.name}: no reliability subset generated.")

            cu_df.to_excel(label_path / f"{lab_str}cu_coding.xlsx", index=False)

        except Exception as e:
            logger.error(f"Failed processing {_rel(file)}: {e}")
