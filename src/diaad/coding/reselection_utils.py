import random
from pathlib import Path

import pandas as pd

from diaad.core.logger import logger, _rel
from diaad.utils.sampling import calc_subset_size


def label_one(tier_obj, fname: str):
    """Safely get one tier label for a filename."""
    try:
        if hasattr(tier_obj, "match"):
            return tier_obj.match(fname)
    except Exception:
        pass
    return None


def labels_for(tiers, path: Path):
    """
    Return a tuple of tier labels for matching original/reliability files.

    If tiers are unavailable, fall back to the stem so files can still pair.
    """
    if not tiers:
        return (path.stem,)

    labels = []
    for t in tiers.values():
        try:
            labels.append(label_one(t, path.name))
        except Exception:
            labels.append(None)
    return tuple(labels)


def normalize_sample_ids(series):
    """Normalize sample_id values to stripped strings, excluding NaN."""
    return series.dropna().astype(str).str.strip()


def cols_to_comment(df):
    """
    Return columns up to and including the first comment-style boundary column.

    Preference order:
      1. exact 'comment'
      2. first column ending in '_comment'
      3. first column ending in '_com'
      4. all columns if no comment-like boundary exists
    """
    if "comment" in df.columns:
        idx = df.columns.get_loc("comment")
        return list(df.columns[: idx + 1])

    for col in df.columns:
        if col.endswith("_comment") or col.endswith("_com"):
            idx = df.columns.get_loc(col)
            return list(df.columns[: idx + 1])

    return list(df.columns)


def post_comment_cols(df):
    """Return columns after the first comment-style boundary column."""
    if "comment" in df.columns:
        start = df.columns.get_loc("comment") + 1
        return list(df.columns[start:])

    for col in df.columns:
        if col.endswith("_comment") or col.endswith("_com"):
            start = df.columns.get_loc(col) + 1
            return list(df.columns[start:])

    return []


def ordered_union(cols1, cols2):
    """Return ordered union preserving first appearance."""
    out = []
    seen = set()
    for col in list(cols1) + list(cols2):
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def discover_reliability_pairs(tiers, input_dir, coding_glob, rel_glob, rel_label):
    """
    Return dict of {original_file: [reliability_files]} matched by tier labels.
    """
    input_dir = Path(input_dir)
    coding_files = list(input_dir.rglob(coding_glob))
    rel_files = list(input_dir.rglob(rel_glob))
    rel_labels = {p: labels_for(tiers, p) for p in rel_files}

    matches = {}
    for org in coding_files:
        org_labels = labels_for(tiers, org)
        matched = [p for p, labs in rel_labels.items() if labs == org_labels]
        if not matched:
            logger.warning(f"[{rel_label}] No reliability files found for {org.name}")
        matches[org] = matched

    return matches


def load_original_and_reliability(org_file, rel_mates, rel_label):
    """
    Load original and reliability DataFrames.

    Returns
    -------
    tuple[pd.DataFrame | None, list[pd.DataFrame]]
        (original_df, list_of_reliability_dfs)
    """
    try:
        df_org = pd.read_excel(org_file)
    except Exception as e:
        logger.error(f"[{rel_label}] Failed reading {_rel(org_file)}: {e}")
        return None, []

    if "sample_id" not in df_org.columns:
        logger.warning(f"[{rel_label}] Missing sample_id in {org_file.name}")
        return None, []

    df_org = df_org.copy()
    df_org["sample_id"] = normalize_sample_ids(df_org["sample_id"])

    rel_dfs = []
    for rf in rel_mates:
        try:
            rdf = pd.read_excel(rf)
            if "sample_id" not in rdf.columns:
                logger.warning(f"[{rel_label}] Skipping {_rel(rf)} because sample_id is missing.")
                continue
            rdf = rdf.copy()
            rdf["sample_id"] = normalize_sample_ids(rdf["sample_id"])
            rel_dfs.append(rdf)
        except Exception as e:
            logger.warning(f"[{rel_label}] Failed reading {_rel(rf)}: {e}")

    return df_org, rel_dfs


def collect_used_ids(rel_dfs):
    """Collect already-used sample_ids from prior reliability files."""
    used_ids = set()
    for rdf in rel_dfs:
        used_ids.update(normalize_sample_ids(rdf["sample_id"]))
    return used_ids


def select_new_samples(df_org, used_ids, frac, rng=None):
    """
    Randomly select new sample_ids not already used for reliability.

    Parameters
    ----------
    df_org : pd.DataFrame
    used_ids : set[str]
    frac : float
    rng : random.Random | None

    Returns
    -------
    list[str]
    """
    rng = rng or random

    all_ids = set(normalize_sample_ids(df_org["sample_id"]))
    available = sorted(all_ids - used_ids)

    if not available:
        logger.warning("No unused samples available for reselection.")
        return []

    n = calc_subset_size(frac=frac, samples=all_ids)
    n = min(len(available), n)

    if n <= 0:
        logger.warning("Calculated reselection size is 0.")
        return []

    return rng.sample(available, n)


def write_reselected_reliability(df, org_file, out_dir, suffix, stem_token, rel_label):
    """
    Save reselected reliability DataFrame to Excel.

    Parameters
    ----------
    df : pd.DataFrame
    org_file : Path
    out_dir : Path
    suffix : str
        Final suffix, e.g. 'cu_reliability_coding' or 'word_count_reliability'
    stem_token : str
        Token to strip from original stem, e.g. 'cu_coding'
    rel_label : str
        Logging label, e.g. 'CU' or 'WC'
    """
    stem = org_file.stem
    base = stem.replace(stem_token, "").rstrip("_")
    out_path = Path(out_dir) / f"{base}_reselected_{suffix}.xlsx"

    try:
        df.to_excel(out_path, index=False)
        logger.info(f"[{rel_label}] Saved {_rel(out_path)}")
    except Exception as e:
        logger.error(f"[{rel_label}] Failed writing {_rel(out_path)}: {e}")
