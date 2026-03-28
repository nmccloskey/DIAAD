import random
from pathlib import Path

from tqdm import tqdm

from diaad.core.logger import logger
from diaad.coding.powers.automation import run_automation
from src.diaad.coding.utils.reselection_utils import (
    cols_to_comment,
    post_comment_cols,
    ordered_union,
    discover_reliability_pairs,
    load_original_and_reliability,
    collect_used_ids,
    select_new_samples,
    write_reselected_reliability,
)


def _ensure_powers_reliability_columns(df):
    """Ensure likely POWERS reliability-side administrative columns exist."""
    for col in ["POWERS_comment", "coder_id", "id"]:
        if col not in df.columns:
            df[col] = ""
    return df


def _clear_manual_powers_fields(df):
    """
    Clear manual POWERS coding fields while preserving automated/helper columns.

    This keeps the reselected file ready for fresh reliability coding rather than
    carrying over prior coding decisions from the original file.
    """
    manual_cols = [
        "id",
        "POWERS_comment",
        "speech_units",
        "turn_type",
        "content_words",
        "num_nouns",
        "circumlocutions",
        "sem_paras",
        "phon_errs",
        "neologisms",
        "comments",
        "lg_pauses",
        "filled_pauses",
        "collab_repair",
    ]

    for col in manual_cols:
        if col in df.columns:
            df[col] = ""

    return df


def _build_powers_reliability_frame(df_org, rel_template, re_ids, automate_powers=True):
    """
    Build a reselected POWERS reliability workbook.
    """
    sub = df_org[df_org["sample_id"].isin(re_ids)].copy()

    head_cols = cols_to_comment(df_org)
    template_tail = post_comment_cols(rel_template) if rel_template is not None else []

    for col in template_tail:
        if col not in sub.columns:
            sub[col] = ""

    sub = _ensure_powers_reliability_columns(sub)
    sub = _clear_manual_powers_fields(sub)

    if automate_powers:
        try:
            sub = run_automation(sub)
        except Exception as e:
            logger.error(f"[POWERS] Failed applying automation during reselection: {e}")

    final_cols = ordered_union(head_cols, template_tail)
    final_cols = [col for col in final_cols if col in sub.columns]

    return sub.loc[:, final_cols]


def reselect_powers_rel(
    tiers,
    input_dir,
    output_dir,
    frac=0.2,
    random_seed=None,
    automate_powers=True,
):
    """
    Reselect POWERS reliability samples, excluding any `sample_id` already
    present in prior POWERS reliability files.
    """
    rng = random.Random(random_seed) if random_seed is not None else random

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    out_dir = output_dir / "reselected_powers_reliability"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_reliability_pairs(
        tiers=tiers,
        input_dir=input_dir,
        coding_glob="*powers_coding.xlsx",
        rel_glob="*powers_reliability_coding.xlsx",
        rel_label="POWERS",
    )

    if not pairs:
        logger.warning("No POWERS files found for reselection.")
        return

    for org_file, rel_mates in tqdm(pairs.items(), desc="Reselecting POWERS reliability"):
        df_org, rel_dfs = load_original_and_reliability(
            org_file,
            rel_mates,
            rel_label="POWERS",
        )
        if df_org is None:
            continue

        used_ids = collect_used_ids(rel_dfs)
        new_ids = select_new_samples(df_org, used_ids, frac, rng=rng)
        if not new_ids:
            continue

        rel_template = rel_dfs[0] if rel_dfs else None

        try:
            new_df = _build_powers_reliability_frame(
                df_org=df_org,
                rel_template=rel_template,
                re_ids=new_ids,
                automate_powers=automate_powers,
            )
        except Exception as e:
            logger.error(
                f"[POWERS] Failed building reselected reliability frame for "
                f"{org_file.name}: {e}"
            )
            continue

        write_reselected_reliability(
            df=new_df,
            org_file=org_file,
            out_dir=out_dir,
            suffix="powers_reliability_coding",
            stem_token="powers_coding",
            rel_label="POWERS",
        )
