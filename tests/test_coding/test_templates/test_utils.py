from __future__ import annotations

import pandas as pd
import pytest

from diaad.coding.templates import utils


def test_template_helpers_basic_behaviors():
    assert utils.normalize_coder_ids(None) == [""]
    assert utils.normalize_coder_ids(("1", " 2 ")) == ["1", "2"]
    assert utils.coerce_bin_labels(3) == [1, 2, 3]

    with pytest.raises(ValueError, match="num_bins must be >= 1"):
        utils.coerce_bin_labels(0)


def test_prepare_and_expand_template_data():
    df = pd.DataFrame({"sample_id": ["S1", "S2"], "story": ["A", "B"]})
    stim_df = utils.prepare_stimulus_lookup(df, "story")
    expanded = utils.expand_by_coder(stim_df, ["1", "2"], insert_after="sample_id")

    assert list(stim_df.columns) == ["sample_id", "stimulus"]
    assert list(expanded.columns) == ["sample_id", "coder_id", "stimulus"]
    assert len(expanded) == 4


def test_assign_template_coders_and_reliability_subset(monkeypatch):
    df = pd.DataFrame({"sample_id": ["S1", "S2", "S3", "S4"]})
    monkeypatch.setattr(utils, "assign_coders", lambda coders: [tuple(coders), tuple(reversed(coders))])
    monkeypatch.setattr(utils.random, "sample", lambda seq, k: list(seq)[:k])

    assigned, segments, assignments = utils.assign_template_coders(df, coder_ids=["1", "2"])
    rel_df = utils.build_reliability_subset(
        assigned,
        frac=0.5,
        coder_ids=["1", "2"],
        segments=segments,
        assignments=assignments,
    )

    assert assigned["coder_id"].notna().all()
    assert rel_df is not None
    assert set(rel_df["coder_id"]) <= {"1", "2"}
