from __future__ import annotations

from diaad.coding.utils import coders


def test_segment():
    assert coders.segment([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_normalize_coders():
    assert coders.normalize_coders([" a ", "", "b"]) == ("two", [" a ", "b"])
