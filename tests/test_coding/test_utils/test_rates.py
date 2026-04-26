from __future__ import annotations

import pandas as pd

from diaad.coding.utils import rates


def test_read_speaking_time_table_normalizes_duplicates(tmp_path):
    path = tmp_path / "speaking_times.xlsx"
    pd.DataFrame(
        {
            "sample_id": ["S1", "S1", "S2"],
            "secs": [30, 15, 60],
        }
    ).to_excel(path, index=False)

    df = rates.read_speaking_time_table(path, speaking_time_field="secs")

    assert list(df["sample_id"]) == ["S1", "S2"]
    assert list(df["speaking_time"]) == [45, 60]
    assert list(df["speaking_minutes"]) == [0.75, 1.0]


def test_merge_and_rate_helpers():
    analysis_df = pd.DataFrame({"sample_id": ["S1"], "count": [6]})
    speaking_time_df = pd.DataFrame(
        {"sample_id": ["S1"], "speaking_time": [120], "speaking_minutes": [2.0]}
    )

    merged = rates.merge_speaking_time(analysis_df, speaking_time_df)
    out = rates.add_rate_columns(merged, ["count"])

    assert merged.loc[0, "speaking_time"] == 120
    assert out.loc[0, "count_per_min"] == 3.0
