from __future__ import annotations

from diaad.coding.utils import coders


def test_resolve_stim_cols_and_segment():
    assert coders.resolve_stim_cols("story") == ["story"]
    assert coders.resolve_stim_cols("") == coders.DEFAULT_STIM_COLS
    assert coders.segment([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_normalize_coders():
    assert coders.normalize_coders([" a ", "", "b"]) == ("two", [" a ", "b"])
