from __future__ import annotations

import pytest

from diaad.coding.utils.sampling import calc_subset_size


def test_calc_subset_size_rounds_up():
    assert calc_subset_size(0.2, [1, 2, 3, 4, 5]) == 1
    assert calc_subset_size(0.5, [1, 2, 3]) == 2


def test_calc_subset_size_validates_inputs():
    with pytest.raises(ValueError, match="0 < frac <= 1"):
        calc_subset_size(0, [1])
    with pytest.raises(ValueError, match="non-empty"):
        calc_subset_size(0.5, [])
