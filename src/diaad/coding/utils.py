import random
import itertools
import numpy as np
import pandas as pd

from diaad.utils.logger import logger


UNINTELLIGIBLE = {"xxx", "yyy", "www"}
DEFAULT_STIM_COLS = ["narrative", "scene", "story", "stimulus"]


def resolve_stim_cols(narrative_field):
    """Use explicit narrative_field when provided; otherwise fall back to legacy stimulus columns."""
    return [narrative_field] if narrative_field else DEFAULT_STIM_COLS


def segment(x, n):
    """
    Segment a list x into n batches of roughly equal length.
    
    Parameters:
    - x (list): List to be segmented.
    - n (int): Number of segments to create.
    
    Returns:
    - list of lists: Segmented batches of roughly equal length.
    """
    if not x or n <= 0:
        return []
    seg_len = max(1, int(round(len(x) / n)))
    segments = [x[i:i + seg_len] for i in range(0, len(x), seg_len)]
    if len(segments) > n:
        last = segments.pop(-1)
        segments[-1] = segments[-1] + last
    return segments


def assign_coders(coders):
    """
    Assign each coder to each role (coder 1, coder 2, coder 3) in different segments.
    
    Parameters:
    - coders (list): List of coder names.
    
    Returns:
    - list of tuples: Each tuple contains an assignment of coders.
    """
    random.shuffle(coders)
    perms = list(itertools.permutations(coders))
    assignments = [perms[0]]
    for p in perms[1:]:
        if all(not any(np.array(p) == np.array(assn)) for assn in assignments):
            assignments.append(p)
    random.shuffle(assignments)
    return assignments


def normalize_coders(coders):
    """Return coder mode and trimmed coder list."""
    coders = [str(c) for c in coders if str(c).strip()]
    if len(coders) > 3:
        logger.warning("More than 3 coders provided; only the first 3 will be used.")
        coders = coders[:3]

    if len(coders) >= 3:
        return "three", coders
    if len(coders) == 2:
        return "two", coders
    if len(coders) == 1:
        return "single", coders
    return "zero", []


# for aggregation.
def utt_ct(x):
    """Count number of utterances."""
    no_utt = len(x.dropna())
    return no_utt if no_utt > 0 else np.nan

def ptotal(x):
    """Count number of positive scores."""
    return sum(x.dropna()) if len(x.dropna()) > 0 else np.nan

def ag_check(x):
    """Check agreement: at least 80% is in agreement."""
    total_cus = len(x.dropna())
    if total_cus > 0:
        return 1 if (sum(x == 1) / total_cus) >= 0.8 else 0
    else:
        return np.nan
    

def compute_cu_column(row):
    """
    Compute a single coder's CU value from paired SV/REL fields.

    Input
    -----
    row : pd.Series
        A two-element series ordered as [SV_col, REL_col] containing values {1, 0, NaN}.

    Returns
    -------
    int | float
        1 if SV == 1 and REL == 1 (coder marked the utterance as a CU on both dimensions)
        0 if both are present and at least one is not 1
        NaN if both entries are missing
        log error + NaN if only one is missing

    Notes
    -----
    - If exactly one of (SV, REL) is NaN while the other is not, an error is logged
      (neutrality inconsistency) and NaN is returned.
    """
    sv, rel = row.iloc[0], row.iloc[1]

    if (pd.isna(sv) and not pd.isna(rel)) or (pd.isna(rel) and not pd.isna(sv)):
        logger.error(f"Neutrality inconsistency in CU computation: SV={sv}, REL={rel}")
        return np.nan
    elif pd.isna(sv) and pd.isna(rel):
        return np.nan
    elif sv == rel == 1:
        return 1
    else:
        return 0
