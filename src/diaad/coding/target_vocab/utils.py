import re
from pathlib import Path

import contractions
import num2words as n2w
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from diaad.coding.target_vocab.resources import (
    get_resource,
    get_resource_ids,
    load_builtin_resources,
)
from diaad.coding.utils import UNINTELLIGIBLE, resolve_stim_cols
from psair.core.logger import get_rel_path, logger
from psair.metadata.discovery import find_matching_files
from diaad.metadata.discovery import find_transcript_table, require_one_file
from diaad.transcripts.transcript_tables import extract_transcript_data


def generate_token_columns(present_narratives, resources: dict | None = None):
    """
    Build legacy surface-form column names per narrative and base form.

    The main target vocabulary coverage output now uses a canonical long detail
    table. This helper remains for older callers that still expect wide names.
    """
    resources = resources or load_builtin_resources()
    return [
        f"{scene[:3]}_{base_form}"
        for scene in present_narratives
        for base_form in resources.get(scene, {}).get("base_forms", [])
    ]


def _col(df, candidates):
    """Return the first matching column from a list of possible variants."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_target_vocab_inputs(
    input_dir: str,
    output_dir: str,
    transcript_table_filename: str = "transcript_tables.xlsx",
) -> tuple[str | None, pd.DataFrame | None]:
    """
    Locate valid target vocabulary coverage input files in priority order.

    Search order:
      1. *unblind_utterance_data*.xlsx, the preferred single summary file.
      2. *transcript_tables*.xlsx, the fallback source.
    """
    search_dirs = [Path(output_dir), Path(input_dir)]

    unblind_matches = find_matching_files(
        directories=search_dirs,
        search_base="unblind_utterance_data",
        search_ext=".xlsx",
        deduplicate=False,
    )
    if unblind_matches:
        p = require_one_file(
            unblind_matches,
            label="unblind utterance data file",
            configured_filename="unblind_utterance_data*.xlsx",
            directories=search_dirs,
        )
        df = read_excel_safely(p)
        if df is not None:
            logger.info(f"Using unblind utterance data: {get_rel_path(p)}")
            return "unblind", df

    transcript_table = find_transcript_table(
        directories=[input_dir, output_dir],
        filename=transcript_table_filename,
        required=False,
    )
    if transcript_table is None:
        logger.error(
            "No target vocabulary coverage input files found "
            "(*unblind_utterance_data*.xlsx or *transcript_tables*.xlsx)."
        )
        return None, None

    utt_df = extract_transcript_data(transcript_table)
    if utt_df is None:
        logger.error("Transcript tables found but none could be loaded.")
        return None, None

    logger.info(f"Loaded transcript table {get_rel_path(transcript_table)}.")
    return "transcripts", utt_df


def _drop_excluded_speaker_rows(
    df: pd.DataFrame,
    exclude_speakers,
) -> pd.DataFrame:
    """Remove rows from speakers excluded from target-vocabulary analysis."""
    if not exclude_speakers or "speaker" not in df.columns:
        return df

    exclude_set = {str(s).strip().lower() for s in exclude_speakers if str(s).strip()}
    if not exclude_set:
        return df

    speaker_labels = df["speaker"].astype(str).str.strip().str.lower()
    keep_mask = ~speaker_labels.isin(exclude_set)
    n_excluded = int((~keep_mask).sum())

    if n_excluded:
        logger.info(
            f"Excluded {n_excluded} target-vocabulary row(s) from analysis based on speaker label."
        )

    return df.loc[keep_mask].copy()


def prepare_target_vocab_inputs(
    input_dir,
    output_dir,
    exclude_speakers,
    stimulus_field="narrative",
    resources: dict | None = None,
    sample_id_field: str = "sample_id",
    transcript_table_filename: str = "transcript_tables.xlsx",
):
    """
    Load and normalize utterance-level target vocabulary coverage inputs.

    The function keeps the existing command path, but filters stimuli using the
    active target vocabulary resources and removes excluded speaker rows when
    speaker labels are available.
    """
    try:
        mode, utt_df = find_target_vocab_inputs(
            input_dir,
            output_dir,
            transcript_table_filename=transcript_table_filename,
        )
        if utt_df is None:
            return None, None
        if sample_id_field not in utt_df.columns:
            logger.error(
                f"Required sample identifier column missing in target vocabulary input: {sample_id_field}"
            )
            return None, None

        stim_cols = resolve_stim_cols(stimulus_field)
        resource_ids = get_resource_ids(resources)

        if mode == "unblind":
            narr_col = _col(utt_df, stim_cols)
            if narr_col is None:
                logger.error("Required stimulus/narrative column missing in unblind input.")
                return None, None

            utt_df = utt_df[utt_df[narr_col].isin(resource_ids)].copy()
            if narr_col != "narrative":
                utt_df = utt_df.rename(columns={narr_col: "narrative"})

            utt_df = _drop_excluded_speaker_rows(utt_df, exclude_speakers)

            cu_col = next((c for c in utt_df.columns if c.startswith("c2_cu")), None)
            wc_col = "word_count" if "word_count" in utt_df.columns else None
            filter_col = cu_col or wc_col
            if filter_col:
                utt_df = utt_df[~np.isnan(utt_df[filter_col])]
            else:
                logger.warning("No c2_cu or word_count column; continuing unfiltered.")
            present_narratives = set(utt_df["narrative"].dropna().unique())

        else:
            narr_col = _col(utt_df, stim_cols)
            utt_col = _col(utt_df, ["utterance", "text", "tokens"])
            time_col = _col(
                utt_df,
                [
                    "speaking_time",
                    "client_time",
                    "speech_time",
                    "time_s",
                    "time_sec",
                    "time_seconds",
                ],
            )

            if not all([narr_col, utt_col, time_col]):
                logger.error("Required columns missing in transcript table input.")
                return None, None

            utt_df = utt_df[utt_df[narr_col].isin(resource_ids)].copy()
            utt_df = _drop_excluded_speaker_rows(utt_df, exclude_speakers)

            present_narratives = set(utt_df[narr_col].dropna().unique())
            utt_df = utt_df.rename(columns={narr_col: "narrative"})
            if utt_col != "utterance":
                utt_df = utt_df.rename(columns={utt_col: "utterance"})
            if time_col != "speaking_time":
                utt_df = utt_df.rename(columns={time_col: "speaking_time"})

        return utt_df, present_narratives

    except Exception as e:
        logger.error(f"Failed to prepare target vocabulary coverage inputs: {e}")
        return None, None


def reformat(text: str) -> str:
    """
    Prepare a transcription text string for target vocabulary coverage analysis.

    - Expands contractions while approximately preserving possessive 's.
    - Converts standalone digits to words.
    - Preserves accepted CHAT replacements like "[: dogs] [*]" as "dogs".
    - Removes common CHAT/CLAN annotation containers and unintelligible markers.
    """
    try:
        text = text.lower().strip()

        text = re.sub(r"\b(he|it)'s got\b", r"\1 has got", text)

        expanded = []
        for tok in text.split():
            if re.fullmatch(r"\w+'s", tok):
                expanded.append(tok)
            else:
                expanded.append(contractions.fix(tok))
        text = " ".join(expanded)

        text = re.sub(r"\b\d+\b", lambda m: n2w.num2words(int(m.group())), text)
        text = re.sub(r"\[:\s*([^\]]+?)\s*\]\s*\[\*\]", r"\1", text)
        text = re.sub(r"\[[^\]]+\]", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\(\([^)]*\)\)", " ", text)
        text = re.sub(r"\{[^}]+\}", " ", text)
        text = re.sub(r"(?<!\S)(?!'s\b)[^\w\s']\S*", " ", text)
        text = re.sub(r"[^\w\s']", " ", text)

        toks = [t for t in text.split() if t not in UNINTELLIGIBLE]
        return " ".join(toks).strip()

    except Exception as e:
        logger.error(f"An error occurred while reformatting: {e}")
        return ""


def id_core_words(scene_name: str, reformatted_text: str, resources: dict | None = None) -> dict:
    """
    Identify target vocabulary base forms in a narrative sample.

    The function name is retained for compatibility with older callers.
    New code should interpret the results as target vocabulary coverage metrics.
    """
    resource = get_resource(scene_name, resources)
    base_forms = resource.get("base_forms", []) if resource else []
    reverse_lookup = resource.get("_reverse_variant_lookup", {}) if resource else {}

    tokens = reformatted_text.split()
    token_sets: dict[str, set[str]] = {}
    base_form_counts: dict[str, int] = {}
    num_core_token_matches = 0

    for token in tokens:
        base_form = reverse_lookup.get(token)
        if base_form is None:
            continue

        num_core_token_matches += 1
        base_form_counts[base_form] = base_form_counts.get(base_form, 0) + 1
        token_sets.setdefault(base_form, set()).add(token)

    num_tokens = len(tokens)
    num_base_forms_produced = len(base_form_counts)
    total_lexicon_size = len(base_forms)
    lexicon_coverage = (
        num_base_forms_produced / total_lexicon_size
        if total_lexicon_size > 0
        else 0.0
    )

    return {
        "num_tokens": num_tokens,
        "num_base_forms_produced": num_base_forms_produced,
        "num_core_token_matches": num_core_token_matches,
        "lexicon_coverage": lexicon_coverage,
        "base_form_counts": base_form_counts,
        "token_sets": token_sets,
        # Backward-compatible aliases for older in-process callers.
        "num_core_words": num_base_forms_produced,
        "num_cw_tokens": num_core_token_matches,
    }


def get_norm_columns(stimulus_name: str, metric: str, resources: dict | None = None) -> dict:
    """Return column metadata for a resource norm table, with legacy fallbacks."""
    resource = get_resource(stimulus_name, resources) or {}
    columns = (
        resource.get("norms", {})
        .get(metric, {})
        .get("columns", {})
    )
    if metric == "accuracy":
        default_raw_score = "CoreLex Score"
    else:
        default_raw_score = "CoreLex/min"
    return {
        "raw_score": columns.get("raw_score", default_raw_score),
        "group": columns.get("group", "Aphasia"),
    }


def load_norms_online(
    stimulus_name: str,
    metric: str = "accuracy",
    resources: dict | None = None,
) -> pd.DataFrame:
    """Load norm data declared by a target vocabulary resource."""
    resource = get_resource(stimulus_name, resources)
    if resource is None:
        raise ValueError(f"Unknown stimulus/resource '{stimulus_name}'")

    norm_spec = resource.get("norms", {}).get(metric)
    if not norm_spec:
        raise ValueError(f"Unknown norm metric '{metric}' for resource '{stimulus_name}'")
    if norm_spec.get("format", "csv") != "csv":
        raise ValueError(
            f"Unsupported norm format for resource '{stimulus_name}' metric '{metric}'"
        )

    try:
        return pd.read_csv(norm_spec["url"])
    except Exception as e:
        raise RuntimeError(f"Failed to load data from URL: {e}") from e


def load_target_vocab_norms_online(stimulus_name: str, metric: str = "accuracy") -> pd.DataFrame:
    """Compatibility wrapper for loading target vocabulary norm data."""
    return load_norms_online(stimulus_name, metric)


def preload_target_vocab_norms(present_narratives: set, resources: dict | None = None) -> dict:
    """
    Preload norms for target vocabulary resources in the current batch.
    """
    norm_data = {}
    resources = resources or load_builtin_resources()

    for scene in present_narratives:
        resource = resources.get(scene)
        if resource is None:
            logger.warning(f"No target vocabulary resource found for: {scene}")
            continue

        try:
            norm_data[scene] = {
                "accuracy": load_norms_online(scene, "accuracy", resources),
                "efficiency": load_norms_online(scene, "efficiency", resources),
                "__metadata__": resource.get("norms", {}),
            }
            logger.info(f"Loaded target vocabulary norms for: {scene}")
        except Exception as e:
            logger.warning(f"Failed to load norms for {scene}: {e}")
            norm_data[scene] = {
                "accuracy": None,
                "efficiency": None,
                "__metadata__": resource.get("norms", {}),
            }

    return norm_data


def get_percentiles(
    score: float,
    norm_df: pd.DataFrame,
    column: str,
    group_col: str = "Aphasia",
) -> dict:
    """
    Compute percentile rank of a score relative to control and PWA distributions.
    """
    control_scores = norm_df[norm_df[group_col] == 0][column]
    pwa_scores = norm_df[norm_df[group_col] == 1][column]

    return {
        "control_percentile": percentileofscore(control_scores, score, kind="weak"),
        "pwa_percentile": percentileofscore(pwa_scores, score, kind="weak"),
    }


def read_excel_safely(path):
    try:
        return pd.read_excel(path)
    except Exception as e:
        logger.warning(f"Failed reading {get_rel_path(path)}: {e}")
        return None


_read_excel_safely = read_excel_safely
