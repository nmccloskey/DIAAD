from __future__ import annotations

import pandas as pd

from diaad.coding.target_vocab import rates


def test_target_vocab_rates_accept_custom_sample_id():
    df = pd.DataFrame(
        {
            "expanded_sample_id": ["S1"],
            "narrative": ["StoryA"],
            "speaking_time": [60],
            "num_tokens": [5],
            "lexicon_coverage": [0.5],
        }
    )

    numerators = rates.infer_target_vocab_rate_numerators(
        df,
        sample_id_field="expanded_sample_id",
    )
    final = rates.finalize_target_vocab_rates_columns(
        df,
        numerator_cols=numerators,
        sample_id_field="expanded_sample_id",
    )

    assert "expanded_sample_id" in final.columns
    assert "num_tokens" in numerators
