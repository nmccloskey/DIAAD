from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from diaad.coding.target_vocab.resources import get_resource, load_target_vocabulary_resources
from diaad.coding.target_vocab.utils import (
    get_norm_columns,
    get_percentiles,
    id_core_words,
    prepare_target_vocab_inputs,
    preload_target_vocab_norms,
    reformat,
)
from psair.core.logger import get_rel_path, logger


def summary_columns(sample_id_field: str = "sample_id") -> list[str]:
    return [
        sample_id_field,
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


def detail_columns(sample_id_field: str = "sample_id") -> list[str]:
    return [
        sample_id_field,
        "narrative",
        "base_form",
        "num_tokens_matched",
        "score",
    ]


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
    resources: dict | None = None,
) -> tuple[float, float]:
    norms_for_narrative = norm_lookup.get(narrative, {}) if isinstance(norm_lookup, dict) else {}
    norm_df = norms_for_narrative.get(metric)
    if norm_df is None:
        logger.warning(
            f"Target vocabulary coverage: {metric} norms missing for narrative '{narrative}'."
        )
        return np.nan, np.nan

    try:
        columns = get_norm_columns(narrative, metric, resources)
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


def _detail_rows_from_stats(
    sample_id,
    narrative: str,
    coverage_stats: dict,
    resources: dict | None = None,
    sample_id_field: str = "sample_id",
) -> list[dict]:
    resource = get_resource(narrative, resources)
    base_forms = resource.get("base_forms", []) if resource else []
    counts = coverage_stats.get("base_form_counts", {})

    return [
        {
            sample_id_field: sample_id,
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
    resources: dict | None = None,
    sample_id_field: str = "sample_id",
) -> tuple[dict, list[dict]]:
    """
    Compute target vocabulary coverage metrics for one sample.

    The default resources are bundled with DIAAD. A caller may pass a custom
    resource registry for project-specific target vocabularies. Norms are
    optional; missing norm data leaves percentile columns as NaN rather than
    suppressing coverage metrics.
    """
    if not isinstance(narrative, str) or not narrative.strip():
        logger.warning(
            "Target vocabulary coverage: missing/invalid narrative; returning empty metrics."
        )
        return {}, []

    resources = resources or load_target_vocabulary_resources()
    resource = get_resource(narrative, resources)
    if resource is None:
        logger.warning(
            f"Target vocabulary coverage: no resource found for '{narrative}'."
        )
        return {}, []

    text = "" if text is None else str(text)
    reformatted_text = reformat(text)
    st = _coerce_speaking_time_seconds(speaking_time)
    coverage_stats = id_core_words(narrative, reformatted_text, resources)

    minutes = (st / 60.0) if pd.notnull(st) and st > 0 else np.nan
    core_tokens_per_min = (
        coverage_stats["num_core_token_matches"] / minutes
        if pd.notnull(minutes) and minutes > 0
        else np.nan
    )

    norm_lookup = norm_lookup or {}
    accuracy_pwa, accuracy_control = _percentiles_for_metric(
        score=coverage_stats["num_base_forms_produced"],
        narrative=narrative,
        norm_lookup=norm_lookup,
        metric="accuracy",
        resources=resources,
    )
    if pd.notnull(core_tokens_per_min):
        efficiency_pwa, efficiency_control = _percentiles_for_metric(
            score=core_tokens_per_min,
            narrative=narrative,
            norm_lookup=norm_lookup,
            metric="efficiency",
            resources=resources,
        )
    else:
        efficiency_pwa = efficiency_control = np.nan

    summary = {
        "narrative": narrative,
        "speaking_time": st,
        "num_tokens": coverage_stats.get("num_tokens", np.nan),
        "num_base_forms_produced": coverage_stats.get("num_base_forms_produced", np.nan),
        "num_core_token_matches": coverage_stats.get("num_core_token_matches", np.nan),
        "lexicon_coverage": coverage_stats.get("lexicon_coverage", np.nan),
        "core_tokens_per_min": core_tokens_per_min,
        "accuracy_pwa_percentile": accuracy_pwa,
        "accuracy_control_percentile": accuracy_control,
        "efficiency_pwa_percentile": efficiency_pwa,
        "efficiency_control_percentile": efficiency_control,
    }
    detail_rows = _detail_rows_from_stats(
        sample_id,
        narrative,
        coverage_stats,
        resources,
        sample_id_field=sample_id_field,
    )
    return summary, detail_rows


def compute_target_vocab_for_text(
    *,
    text: str,
    speaking_time,
    narrative: str,
    norm_lookup: dict,
    resources: dict | None = None,
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
            resources=resources,
        )
        return summary
    except Exception as e:
        logger.error(
            f"Failed target vocabulary coverage compute for narrative '{narrative}': {e}"
        )
        return {}


def extract_target_vocab_inputs_from_sample_df(
    sample_df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> dict:
    """
    Extract text, speaking_time, narrative, and sample id from a DIAAD sample.

    The function name remains for compatibility with existing callers.
    """
    try:
        if sample_df is None or sample_df.empty:
            return {}

        required = {"utterance", "narrative", sample_id_field}
        missing = [c for c in required if c not in sample_df.columns]
        if missing:
            logger.warning(
                f"Target vocabulary coverage: sample_df missing required columns: {missing}"
            )
            return {}

        sample_id = sample_df[sample_id_field].iloc[0]
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
            sample_id_field: sample_id,
            "narrative": narrative,
            "speaking_time": speaking_time,
            "text": text,
        }

    except Exception as e:
        logger.error(f"Target vocabulary coverage: failed extracting sample inputs: {e}")
        return {}


def _compute_target_vocab_for_sample(
    sample_df,
    norm_lookup,
    resources=None,
    sample_id_field: str = "sample_id",
):
    """
    Backward-compatible wrapper that returns a summary row and long detail rows.
    """
    try:
        extracted = extract_target_vocab_inputs_from_sample_df(
            sample_df,
            sample_id_field=sample_id_field,
        )
        if not extracted:
            return {}, []

        row_prefix = {sample_id_field: extracted[sample_id_field]}

        summary, details = compute_target_vocabulary_coverage_for_text(
            text=extracted["text"],
            speaking_time=extracted["speaking_time"],
            narrative=extracted["narrative"],
            norm_lookup=norm_lookup,
            sample_id=extracted[sample_id_field],
            resources=resources,
            sample_id_field=sample_id_field,
        )
        if not summary:
            return {}, []

        row_prefix.update(summary)
        return row_prefix, details

    except Exception as e:
        logger.error(f"Failed to compute target vocabulary coverage for sample_df: {e}")
        return {}, []


def _ordered_summary_columns(
    df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> list[str]:
    ordered = [c for c in summary_columns(sample_id_field) if c in df.columns]
    return ordered + [c for c in df.columns if c not in ordered]


def run_target_vocab(
    metadata_fields,
    input_dir,
    output_dir,
    exclude_speakers=None,
    stimulus_field="narrative",
    resource_path=None,
    sample_id_field: str = "sample_id",
    transcript_table_filename: str = "transcript_tables.xlsx",
):
    """
    Execute target vocabulary coverage analysis using built-in or custom resources.

    Rows from configured excluded speaker tier labels are removed during input
    preparation when speaker labels are available.
    """
    exclude_speakers = set(exclude_speakers or [])
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    target_vocab_dir = output_dir / "target_vocab"
    target_vocab_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Target vocabulary coverage output directory: {get_rel_path(target_vocab_dir)}")
    resources = load_target_vocabulary_resources(resource_path)

    utt_df, present_narratives = prepare_target_vocab_inputs(
        input_dir,
        output_dir,
        exclude_speakers,
        stimulus_field=stimulus_field,
        resources=resources,
        sample_id_field=sample_id_field,
        transcript_table_filename=transcript_table_filename,
    )
    if utt_df is None:
        return

    norm_lookup = preload_target_vocab_norms(present_narratives, resources=resources)
    summary_rows = []
    detail_rows = []

    for sample in tqdm(
        sorted(utt_df[sample_id_field].dropna().unique()),
        desc="Computing target vocabulary coverage",
    ):
        sample_df = utt_df[utt_df[sample_id_field] == sample]
        if sample_df.empty:
            continue
        summary_row, sample_details = _compute_target_vocab_for_sample(
            sample_df,
            norm_lookup,
            resources,
            sample_id_field=sample_id_field,
        )
        if summary_row:
            summary_rows.append(summary_row)
            detail_rows.extend(sample_details)

    if not summary_rows:
        logger.warning("No target vocabulary coverage rows produced; no output written.")
        return

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[
        _ordered_summary_columns(summary_df, sample_id_field=sample_id_field)
    ]

    detail_df = pd.DataFrame(detail_rows, columns=detail_columns(sample_id_field))

    output_file = target_vocab_dir / f"target_vocab_data_{timestamp}.xlsx"
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            detail_df.to_excel(writer, sheet_name="details", index=False)
        logger.info(f"Target vocabulary coverage results written to {get_rel_path(output_file)}")
    except Exception as e:
        logger.error(f"Failed to write target vocabulary coverage results: {e}")

    logger.info("Target vocabulary coverage processing complete.")
