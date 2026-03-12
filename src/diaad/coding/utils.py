import random
import itertools

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
