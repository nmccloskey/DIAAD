from __future__ import annotations
import math
from pathlib import Path
from typing import Sized
from diaad.utils.logger import logger, _rel


def find_matching_files(
    match_tiers=None,
    directories=None,
    search_base="",
    search_ext=".xlsx",
    deduplicate=True,
):
    """
    Recursively find files matching tier labels and a base pattern.

    Behavior
    --------
    • Searches all provided directories for filenames containing both
      `search_base` and every label in `match_tiers` (case-sensitive).
    • Returns a list[Path] of matches (empty if none found).
    • Optionally deduplicates identical filenames across directories,
      logging which duplicates were removed.

    Parameters
    ----------
    match_tiers : list[str] | None
        Tier labels (e.g., ["AC", "PreTx"]). None/empty ignored.
    directories : Path | str | list[Path | str] | None
        One or more directories to search (default: CWD).
    search_base : str
        Core substring to match in filenames.
    search_ext : str, default ".xlsx"
        File extension (with dot).
    deduplicate : bool, default True
        Remove duplicate filenames across directories.

    Returns
    -------
    list[Path]
        Matching file paths (may be empty).
    """
    match_tiers = [str(mt) for mt in (match_tiers or []) if mt]
    if directories is None:
        directories = [Path.cwd()]
    elif isinstance(directories, (str, Path)):
        directories = [directories]

    all_matches = []
    for d in directories:
        try:
            d = Path(d)
            if not d.exists():
                logger.warning(f"Directory not found: {_rel(d)} (skipping).")
                continue

            for f in d.rglob(f"*{search_base}*{search_ext}"):
                if all(mt in f.name for mt in match_tiers):
                    all_matches.append(f)
        except Exception as e:
            logger.error(f"Error searching in {_rel(d)}: {e}")

    if not all_matches:
        logger.warning(f"No matches found for base '{search_base}' with tiers {match_tiers}.")
        return []

    if deduplicate:
        seen = {}
        duplicates = {}
        for f in all_matches:
            if f.name in seen:
                duplicates.setdefault(f.name, []).append(f)
            else:
                seen[f.name] = f

        unique_matches = list(seen.values())

        if duplicates:
            logger.warning(
                f"Removed {sum(len(v) for v in duplicates.values())} duplicate filename(s) across directories."
            )
            for fname, paths in duplicates.items():
                logger.warning(f"Duplicate filename '{fname}' found in:")
                for p in [seen[fname], *paths]:
                    logger.warning(f"  - {_rel(p)}")

    else:
        unique_matches = all_matches

    if len(unique_matches) == 1:
        logger.info(f"Matched file for '{search_base}': {_rel(unique_matches[0])}")
    else:
        logger.info(
            f"Multiple ({len(unique_matches)}) files matched '{search_base}' and {match_tiers}."
        )
        for f in unique_matches:
            logger.debug(f"  - {_rel(f)}")

    return unique_matches


def calc_subset_size(frac: float, samples: Sized) -> int:
    """
    Calculate the minimum subset size required to satisfy a fractional
    sampling threshold, based on the number of available samples.

    The returned value is ceil(frac * n_samples), with a floor of 1.

    Parameters
    ----------
    frac : float
        Fractional proportion to sample. Must satisfy 0 < frac <= 1.

    samples : Sized
        Any object with a defined length (supports len(samples)).
        Examples: list of paths, pandas DataFrame, pandas Series, etc.

    Returns
    -------
    int
        Minimum required subset size (>= 1).

    Raises
    ------
    TypeError
        If frac is not numeric, or samples does not implement __len__.

    ValueError
        If frac is not within (0, 1], or if len(samples) == 0.
    """
    if not isinstance(frac, (float, int)):
        raise TypeError(f"frac must be numeric (float/int) in (0, 1]; got {type(frac)}")

    frac = float(frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must satisfy 0 < frac <= 1; got {frac}")

    try:
        n_samples = len(samples)
    except TypeError as e:
        raise TypeError("samples must be Sized (support len(samples))") from e

    if n_samples == 0:
        raise ValueError("samples must be non-empty (len(samples) > 0)")

    return max(1, math.ceil(frac * n_samples))
