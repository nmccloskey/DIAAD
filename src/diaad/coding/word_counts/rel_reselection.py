import random
from pathlib import Path

from tqdm import tqdm

from diaad.core.logger import logger
from diaad.coding.word_counts.files import count_words
from diaad.coding.reselection_utils import (
    cols_to_comment,
    post_comment_cols,
    ordered_union,
    discover_reliability_pairs,
    load_original_and_reliability,
    collect_used_ids,
    select_new_samples,
    write_reselected_reliability,
)


def _ensure_wc_reliability_columns(df):
    """Ensure likely word-count reliability-side administrative columns exist."""
    for col in ["wc_rel_com"]:
        if col not in df.columns:
            df[col] = ""
    return df


def _build_wc_reliability_frame(df_org, rel_template, re_ids):
    """
    Build a reselected word-count reliability workbook.
    """
    sub = df_org[df_org["sample_id"].isin(re_ids)].copy()

    head_cols = cols_to_comment(df_org)
    template_tail = post_comment_cols(rel_template) if rel_template is not None else []

    for col in template_tail:
        if col not in sub.columns:
            sub[col] = ""

    sub = _ensure_wc_reliability_columns(sub)

    def compute_wc(row):
        return count_words(row.get("utterance", ""))
    sub["word_count"] = sub.apply(compute_wc, axis=1)

    final_cols = ordered_union(head_cols, template_tail)
    final_cols = [col for col in final_cols if col in sub.columns]

    return sub.loc[:, final_cols]


def reselect_wc_rel(tiers, input_dir, output_dir, frac=0.2, random_seed=None):
    """
    Reselect word-count reliability samples, excluding any `sample_id` already
    present in prior word-count reliability files.
    """
    rng = random.Random(random_seed) if random_seed is not None else random

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    out_dir = output_dir / "reselected_word_count_reliability"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_reliability_pairs(
        tiers=tiers,
        input_dir=input_dir,
        coding_glob="*word_counting.xlsx",
        rel_glob="*word_count_reliability.xlsx",
        rel_label="WC",
    )

    if not pairs:
        logger.warning("No WC files found for reselection.")
        return

    for org_file, rel_mates in tqdm(pairs.items(), desc="Reselecting WC reliability"):
        df_org, rel_dfs = load_original_and_reliability(org_file, rel_mates, rel_label="WC")
        if df_org is None:
            continue

        used_ids = collect_used_ids(rel_dfs)
        new_ids = select_new_samples(df_org, used_ids, frac, rng=rng)
        if not new_ids:
            continue

        rel_template = rel_dfs[0] if rel_dfs else None

        try:
            new_df = _build_wc_reliability_frame(df_org, rel_template, new_ids)
        except Exception as e:
            logger.error(f"[WC] Failed building reselected reliability frame for {org_file.name}: {e}")
            continue

        write_reselected_reliability(
            df=new_df,
            org_file=org_file,
            out_dir=out_dir,
            suffix="word_count_reliability",
            stem_token="word_counting",
            rel_label="WC",
        )
