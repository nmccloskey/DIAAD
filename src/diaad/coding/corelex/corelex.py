from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from diaad.coding.corelex.resources import get_builtin_resource
from diaad.coding.corelex.utils import (
    get_norm_columns,
    get_percentiles,
    id_core_words,
    prepare_corelex_inputs,
    preload_corelex_norms,
    reformat,
)
from diaad.core.logger import _rel, logger


SUMMARY_COLUMNS = [
    "sample_id",
    "narrative",
    "speaking_time",
    "num_tokens",
    "num_base_forms_produced",
    "num_core_token_matches",
    "lexicon_coverage",
    "core_tokens_per_min",
    "accuracy_pwa_percentile",
    "accuracy_control_percentile",
    "efficiency_pwa_percentile",
    "efficiency_control_percentile",
]

DETAIL_COLUMNS = [
    "sample_id",
    "narrative",
    "base_form",
    "num_tokens_matched",
    "score",
]

# Backward-compatible name for callers that imported the previous constant.
base_columns = SUMMARY_COLUMNS


def _coerce_speaking_time_seconds(speaking_time):
    try:
        return float(speaking_time) if pd.notnull(speaking_time) else np.nan
    except Exception:
        return np.nan


def _percentiles_for_metric(
    *,
    score: float,
    narrative: str,
    norm_lookup: dict,
    metric: str,
) -> tuple[float, float]:
    norms_for_narrative = norm_lookup.get(narrative, {}) if isinstance(norm_lookup, dict) else {}
    norm_df = norms_for_narrative.get(metric)
    if norm_df is None:
        logger.warning(
            f"Target vocabulary coverage: {metric} norms missing for narrative '{narrative}'."
        )
        return np.nan, np.nan

    try:
        columns = get_norm_columns(narrative, metric)
        pcts = get_percentiles(
            score,
            norm_df,
            columns["raw_score"],
            group_col=columns["group"],
        )
        return pcts.get("pwa_percentile", np.nan), pcts.get("control_percentile", np.nan)
    except Exception as e:
        logger.warning(
            f"Target vocabulary coverage: failed percentile lookup for "
            f"'{narrative}' {metric}: {e}"
        )
        return np.nan, np.nan


def _detail_rows_from_stats(sample_id, narrative: str, core_stats: dict) -> list[dict]:
    resource = get_builtin_resource(narrative)
    base_forms = resource.get("base_forms", []) if resource else []
    counts = core_stats.get("base_form_counts", {})

    return [
        {
            "sample_id": sample_id,
            "narrative": narrative,
            "base_form": base_form,
            "num_tokens_matched": int(counts.get(base_form, 0)),
            "score": int(counts.get(base_form, 0) > 0),
        }
        for base_form in base_forms
    ]


def compute_target_vocabulary_coverage_for_text(
    *,
    text: str,
    speaking_time,
    narrative: str,
    norm_lookup: dict | None = None,
    sample_id=None,
) -> tuple[dict, list[dict]]:
    """
    Compute target vocabulary coverage metrics for one sample.

    Built-in resources currently cover the canonical CoreLex-style tasks bundled
    with DIAAD. Norms are optional; missing norm data leaves percentile columns
    as NaN rather than suppressing coverage metrics.
    """
    if not isinstance(narrative, str) or not narrative.strip():
        logger.warning(
            "Target vocabulary coverage: missing/invalid narrative; returning empty metrics."
        )
        return {}, []

    resource = get_builtin_resource(narrative)
    if resource is None:
        logger.warning(
            f"Target vocabulary coverage: no built-in resource found for '{narrative}'."
        )
        return {}, []

    text = "" if text is None else str(text)
    reformatted_text = reformat(text)
    st = _coerce_speaking_time_seconds(speaking_time)
    core_stats = id_core_words(narrative, reformatted_text)

    minutes = (st / 60.0) if pd.notnull(st) and st > 0 else np.nan
    core_tokens_per_min = (
        core_stats["num_core_token_matches"] / minutes
        if pd.notnull(minutes) and minutes > 0
        else np.nan
    )

    norm_lookup = norm_lookup or {}
    accuracy_pwa, accuracy_control = _percentiles_for_metric(
        score=core_stats["num_base_forms_produced"],
        narrative=narrative,
        norm_lookup=norm_lookup,
        metric="accuracy",
    )
    if pd.notnull(core_tokens_per_min):
        efficiency_pwa, efficiency_control = _percentiles_for_metric(
            score=core_tokens_per_min,
            narrative=narrative,
            norm_lookup=norm_lookup,
            metric="efficiency",
        )
    else:
        efficiency_pwa = efficiency_control = np.nan

    summary = {
        "narrative": narrative,
        "speaking_time": st,
        "num_tokens": core_stats.get("num_tokens", np.nan),
        "num_base_forms_produced": core_stats.get("num_base_forms_produced", np.nan),
        "num_core_token_matches": core_stats.get("num_core_token_matches", np.nan),
        "lexicon_coverage": core_stats.get("lexicon_coverage", np.nan),
        "core_tokens_per_min": core_tokens_per_min,
        "accuracy_pwa_percentile": accuracy_pwa,
        "accuracy_control_percentile": accuracy_control,
        "efficiency_pwa_percentile": efficiency_pwa,
        "efficiency_control_percentile": efficiency_control,
    }
    detail_rows = _detail_rows_from_stats(sample_id, narrative, core_stats)
    return summary, detail_rows


def compute_corelex_for_text(
    *,
    text: str,
    speaking_time,
    narrative: str,
    norm_lookup: dict,
) -> dict:
    """
    Backward-compatible wrapper for target vocabulary coverage metrics.

    The returned metric names use base-form and target-vocabulary terminology.
    """
    try:
        summary, _ = compute_target_vocabulary_coverage_for_text(
            text=text,
            speaking_time=speaking_time,
            narrative=narrative,
            norm_lookup=norm_lookup,
        )
        return summary
    except Exception as e:
        logger.error(
            f"Failed target vocabulary coverage compute for narrative '{narrative}': {e}"
        )
        return {}


def extract_corelex_inputs_from_sample_df(sample_df: pd.DataFrame) -> dict:
    """
    Extract text, speaking_time, narrative, and sample_id from a DIAAD sample.

    The function name remains for compatibility with existing CoreLex callers.
    """
    try:
        if sample_df is None or sample_df.empty:
            return {}

        required = {"utterance", "narrative", "sample_id"}
        missing = [c for c in required if c not in sample_df.columns]
        if missing:
            logger.warning(
                f"Target vocabulary coverage: sample_df missing required columns: {missing}"
            )
            return {}

        sample_id = sample_df["sample_id"].iloc[0]
        narrative = sample_df["narrative"].iloc[0]
        speaking_time = (
            sample_df["speaking_time"].iloc[0]
            if "speaking_time" in sample_df.columns
            else np.nan
        )
        text = " ".join(
            u
            for u in sample_df["utterance"].astype(str).tolist()
            if u and str(u).strip()
        )

        return {
            "sample_id": sample_id,
            "narrative": narrative,
            "speaking_time": speaking_time,
            "text": text,
        }

    except Exception as e:
        logger.error(f"Target vocabulary coverage: failed extracting sample inputs: {e}")
        return {}


def _compute_corelex_for_sample(sample_df, norm_lookup, partition_tiers, tup):
    """
    Backward-compatible wrapper that returns a summary row and long detail rows.
    """
    try:
        extracted = extract_corelex_inputs_from_sample_df(sample_df)
        if not extracted:
            return {}, []

        row_prefix = {
            "sample_id": extracted["sample_id"],
            **(dict(zip(partition_tiers, tup)) if partition_tiers else {}),
        }

        summary, details = compute_target_vocabulary_coverage_for_text(
            text=extracted["text"],
            speaking_time=extracted["speaking_time"],
            narrative=extracted["narrative"],
            norm_lookup=norm_lookup,
            sample_id=extracted["sample_id"],
        )
        if not summary:
            return {}, []

        row_prefix.update(summary)
        return row_prefix, details

    except Exception as e:
        logger.error(f"Failed to compute target vocabulary coverage for sample_df: {e}")
        return {}, []


def _ordered_summary_columns(df: pd.DataFrame, partition_tiers: list[str]) -> list[str]:
    preferred = [SUMMARY_COLUMNS[0]] + partition_tiers + SUMMARY_COLUMNS[1:]
    ordered = [c for c in preferred if c in df.columns]
    return ordered + [c for c in df.columns if c not in ordered]


def run_corelex(
    tiers,
    input_dir,
    output_dir,
    exclude_participants=None,
    stimulus_field="narrative",
):
    """
    Execute target vocabulary coverage analysis using built-in CoreLex-style resources.
    """
    exclude_participants = set(exclude_participants or [])
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    corelex_dir = output_dir / "core_lex"
    corelex_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target vocabulary coverage output directory: {_rel(corelex_dir)}")

    utt_df, present_narratives = prepare_corelex_inputs(
        input_dir,
        output_dir,
        exclude_participants,
        stimulus_field=stimulus_field,
    )
    if utt_df is None:
        return

    partition_tiers = [t.name for t in tiers.values() if getattr(t, "partition", False)]
    norm_lookup = preload_corelex_norms(present_narratives)
    summary_rows = []
    detail_rows = []

    grouped = utt_df.groupby(by=partition_tiers) if partition_tiers else [((), utt_df)]

    for tup, subdf in grouped:
        tup = tup if isinstance(tup, tuple) else (tup,) if partition_tiers else ()
        for sample in tqdm(
            sorted(subdf["sample_id"].dropna().unique()),
            desc="Computing target vocabulary coverage",
        ):
            sample_df = subdf[subdf["sample_id"] == sample]
            if sample_df.empty:
                continue
            summary_row, sample_details = _compute_corelex_for_sample(
                sample_df,
                norm_lookup,
                partition_tiers,
                tup,
            )
            if summary_row:
                summary_rows.append(summary_row)
                detail_rows.extend(sample_details)

    if not summary_rows:
        logger.warning("No target vocabulary coverage rows produced; no output written.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[_ordered_summary_columns(summary_df, partition_tiers)]

    detail_df = pd.DataFrame(detail_rows, columns=DETAIL_COLUMNS)

    output_file = corelex_dir / f"core_lex_data_{timestamp}.xlsx"
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            detail_df.to_excel(writer, sheet_name="details", index=False)
        logger.info(f"Target vocabulary coverage results written to {_rel(output_file)}")
    except Exception as e:
        logger.error(f"Failed to write target vocabulary coverage results: {e}")

    logger.info("Target vocabulary coverage processing complete.")
