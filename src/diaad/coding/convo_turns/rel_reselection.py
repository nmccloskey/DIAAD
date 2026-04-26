from __future__ import annotations

import random
from pathlib import Path

from tqdm import tqdm

from psair.core.logger import logger
from diaad.coding.utils.reselection_utils import (
    discover_reliability_pairs,
    load_original_and_reliability,
    collect_used_ids,
    select_new_samples,
    write_reselected_reliability,
)


def _discover_turn_pairs(metadata_fields, input_dir: Path) -> dict[Path, list[Path]]:
    """
    Discover original/reliability turns files, falling back to filename pairing.
    """
    pairs = discover_reliability_pairs(
        metadata_fields=metadata_fields,
        input_dir=input_dir,
        coding_glob="*conversation_turns_template.xlsx",
        rel_glob="*conversation_turns_reliability*.xlsx",
        rel_label="TURNS",
    )

    if any(rel_mates for rel_mates in pairs.values()):
        return pairs

    coding_files = sorted(input_dir.rglob("*conversation_turns_template.xlsx"))
    rel_files = sorted(input_dir.rglob("*conversation_turns_reliability*.xlsx"))

    if not coding_files or not rel_files:
        return pairs

    if len(coding_files) > 1:
        logger.warning(
            "[TURNS] Multiple conversation turns coding files detected during fallback pairing. "
            "Using the first returned file only."
        )

    logger.info("[TURNS] Falling back to filename-based pairing for turns reselection.")
    return {coding_files[0]: rel_files}


def _build_turns_reliability_frame(df_org, re_ids):
    """
    Build a reselected digital conversation turns reliability workbook.

    Session and bin structure are preserved, while turns are cleared so the
    reliability file is ready for fresh coding.
    """
    sub = df_org[df_org["sample_id"].isin(re_ids)].copy()

    if "turns" in sub.columns:
        sub["turns"] = ""

    return sub


def reselect_digital_convo_turns_rel(
    metadata_fields,
    input_dir,
    output_dir,
    frac=0.2,
    random_seed=None,
):
    """
    Reselect digital conversation turn reliability samples, excluding any
    sample_id already present in prior turns reliability files.
    """
    rng = random.Random(random_seed) if random_seed is not None else random

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    out_dir = output_dir / "reselected_turns_reliability"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_turn_pairs(metadata_fields=metadata_fields, input_dir=input_dir)

    if not pairs:
        logger.warning("No digital conversation turns files found for reselection.")
        return

    for org_file, rel_mates in tqdm(pairs.items(), desc="Reselecting turns reliability"):
        df_org, rel_dfs = load_original_and_reliability(org_file, rel_mates, rel_label="TURNS")
        if df_org is None:
            continue

        used_ids = collect_used_ids(rel_dfs)
        new_ids = select_new_samples(df_org, used_ids, frac, rng=rng)
        if not new_ids:
            continue

        try:
            new_df = _build_turns_reliability_frame(df_org, new_ids)
        except Exception as e:
            logger.error(
                f"[TURNS] Failed building reselected reliability frame for {org_file.name}: {e}"
            )
            continue

        write_reselected_reliability(
            df=new_df,
            org_file=org_file,
            out_dir=out_dir,
            suffix="conversation_turns_reliability_template",
            stem_token="conversation_turns_template",
            rel_label="TURNS",
        )
