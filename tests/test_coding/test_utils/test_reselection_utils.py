from __future__ import annotations

import random

import pandas as pd

from diaad.coding.utils import reselection_utils


class FakeField:
    def match_path_parts(self, parts, source=None):
        return parts[0]


def test_metadata_values_for_and_column_partitioning(tmp_path):
    path = tmp_path / "group1" / "file.xlsx"
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")

    values = reselection_utils.metadata_values_for({"group": FakeField()}, path, input_dir=tmp_path)
    assert values == ("group1",)

    df = pd.DataFrame(columns=["sample_id", "comment", "after"])
    assert reselection_utils.cols_to_comment(df) == ["sample_id", "comment"]
    assert reselection_utils.post_comment_cols(df) == ["after"]
    assert reselection_utils.ordered_union(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


def test_select_new_samples_respects_used_ids():
    df = pd.DataFrame({"sample_id": ["S1", "S2", "S3", "S4"]})
    selected = reselection_utils.select_new_samples(
        df,
        used_ids={"S1"},
        frac=0.5,
        rng=random.Random(0),
    )

    assert len(selected) == 2
    assert "S1" not in selected
