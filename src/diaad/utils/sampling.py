from __future__ import annotations
import math
from typing import Sized


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
