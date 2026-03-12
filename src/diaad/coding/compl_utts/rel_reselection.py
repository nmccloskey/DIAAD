import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from diaad.utils.logger import logger, _rel
from diaad.utils.auxiliary import calc_subset_size

# --- helpers ---
def _label_one(tier_obj, fname: str):
    try:
        if hasattr(tier_obj, "match"):
            return tier_obj.match(fname)
    except Exception:
        pass
    # Fallback: no label for this tier
    return None

def _labels_for(tiers, path: Path):
    if not tiers:
        # If tiers are not provided, fallback to just the stem so pairs can still match
        return (path.stem,)
    labels = []
    for t in tiers.values():
        try:
            labels.append(_label_one(t, path.name))
        except Exception:
            labels.append(None)
    return tuple(labels)

def _cols_to_comment(df):
    if "comment" in df.columns:
        idx = df.columns.get_loc("comment")
        return list(df.columns[: idx + 1])
    return list(df.columns)

def _discover_reliability_pairs(tiers, input_dir, rel_type):
    """Return dict of {original_file: [reliability_files]} matched by tier labels."""
    if rel_type == "CU":
        coding_glob, rel_glob = "*cu_coding.xlsx", "*cu_reliability_coding.xlsx"
    else:
        coding_glob, rel_glob = "*word_counting.xlsx", "*word_count_reliability.xlsx"

    coding_files = list(input_dir.rglob(coding_glob))
    rel_files = list(input_dir.rglob(rel_glob))
    rel_labels = {p: _labels_for(tiers, p) for p in rel_files}

    matches = {}
    for org in coding_files:
        org_labels = _labels_for(tiers, org)
        matched = [p for p, labs in rel_labels.items() if labs == org_labels]
        if not matched:
            logger.warning(f"[{rel_type}] No reliability files for {org.name}")
        matches[org] = matched
    return matches

def _load_original_and_reliability(org_file, rel_mates, rel_type):
    """Load original and reliability DataFrames; ensure sample_id present."""
    try:
        df_org = pd.read_excel(org_file)
    except Exception as e:
        logger.error(f"Failed reading {_rel(org_file)}: {e}")
        return None, None

    rel_dfs = []
    for rf in rel_mates:
        try:
            rel_dfs.append(pd.read_excel(rf))
        except Exception as e:
            logger.warning(f"Failed reading {_rel(rf)}: {e}")
    if "sample_id" not in df_org:
        logger.warning(f"[{rel_type}] Missing sample_id in {org_file.name}")
        return None, None
    rel_dfs = [r for r in rel_dfs if "sample_id" in r]
    return df_org, rel_dfs

def _select_new_samples(df_org, used_ids, frac):
    """Return list of reselected sample_ids not already used."""
    all_ids = set(df_org["sample_id"].dropna().astype(str))
    available = list(all_ids - used_ids)
    if not available:
        logger.warning("No unused samples available.")
        return []
    n = calc_subset_size(frac=frac, samples=all_ids)
    if len(available) < n:
        n = len(available)
    return random.sample(available, n)

def _build_reliability_frame(df_org, rel_template, re_ids, rel_type):
    """Create new reliability DataFrame aligned with template columns."""
    sub = df_org[df_org["sample_id"].astype(str).isin(re_ids)].copy()
    head_cols = _cols_to_comment(df_org)

    if "comment" in rel_template:
        start = rel_template.columns.get_loc("comment") + 1
        post_cols = rel_template.columns[start:]
    else:
        post_cols = rel_template.columns

    for col in post_cols:
        if col not in sub:
            sub[col] = ""

    if rel_type == "CU":
        for c in ["c3_id", "c3_comment"]:
            if c not in sub:
                sub[c] = ""
    else:  # WC
        if "wc_rel_com" not in sub:
            sub["wc_rel_com"] = ""

    cols = [c for c in head_cols if c in sub] + [c for c in post_cols if c in sub and c not in head_cols]
    return sub.loc[:, cols]

def _write_reselected_reliability(df, org_file, out_dir, rel_type):
    """Save reselected reliability DataFrame to Excel."""
    stem = org_file.stem
    suffix = "cu_reliability_coding" if rel_type == "CU" else "word_count_reliability"
    base = stem.replace("cu_coding", "").replace("word_counting", "").rstrip("_")
    out_path = out_dir / f"{base}_reselected_{suffix}.xlsx"
    try:
        df.to_excel(out_path, index=False)
        logger.info(f"[{rel_type}] Saved {_rel(out_path)}")
    except Exception as e:
        logger.error(f"[{rel_type}] Failed writing {_rel(out_path)}: {e}")

def reselect_cu_wc_reliability(
    tiers, input_dir, output_dir, frac=0.2, rel_type="CU"
):
    """
    Reselect reliability samples for CU or WC coding tables, excluding
    any `sample_id` already present in prior reliability files.

    Behavior:
      1. Match original coder files with reliability counterparts by tier labels.
      2. Exclude used sample_ids; randomly reselect ~`frac` of remaining.
      3. Build new reliability workbooks preserving post-comment schema.
      4. Write results under `{output_dir}/reselected_<rel_type>_reliability/`.

    Parameters
    ----------
    tiers : dict[str, Tier]
    input_dir, output_dir : Path or str
    rel_type : {"CU","WC"}, default "CU"
    frac : float, default 0.2
    """
    rel_type = rel_type.upper().strip()
    if rel_type not in {"CU", "WC"}:
        logger.error(f"Invalid rel_type '{rel_type}'. Must be 'CU' or 'WC'.")
        return

    input_dir, output_dir = Path(input_dir), Path(output_dir)
    out_dir_str = "cu_coding" if rel_type == "CU" else "word_count"
    out_dir = output_dir / f"reselected_{out_dir_str}_reliability"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_reliability_pairs(tiers, input_dir, rel_type)
    if not pairs:
        logger.warning(f"No {rel_type} files found for reselection.")
        return

    for org_file, rel_mates in tqdm(pairs.items(), desc=f"Reselecting {rel_type} reliability"):
        df_org, rel_dfs = _load_original_and_reliability(org_file, rel_mates, rel_type)
        if df_org is None or not rel_dfs:
            continue

        used_ids = set().union(*[set(rdf["sample_id"].dropna().astype(str)) for rdf in rel_dfs])
        new_ids = _select_new_samples(df_org, used_ids, frac)
        if not new_ids:
            continue

        new_df = _build_reliability_frame(df_org, rel_dfs[0], new_ids, rel_type)
        _write_reselected_reliability(new_df, org_file, out_dir, rel_type)

def reselect_cu_rel(tiers, input_dir, output_dir, frac=0.2):
    reselect_cu_wc_reliability(tiers, input_dir, output_dir, frac, rel_type="CU")

def reselect_wc_rel(tiers, input_dir, output_dir, frac=0.2):
    reselect_cu_wc_reliability(tiers, input_dir, output_dir, frac, rel_type="WC")
