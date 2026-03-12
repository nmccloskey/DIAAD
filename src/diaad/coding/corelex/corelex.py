import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from diaad.utils.logger import logger, _rel
from diaad.coding.corelex.utils import (
    reformat, id_core_words, preload_corelex_norms, get_percentiles, prepare_corelex_inputs, generate_token_columns
)


base_columns = [
    "sample_id", "narrative", "speaking_time", "num_tokens",
    "num_core_words", "num_core_word_tokens", "lexicon_coverage", "core_words_per_min",
    "core_words_pwa_percentile", "core_words_control_percentile",
    "cwpm_pwa_percentile", "cwpm_control_percentile"
]


def compute_corelex_for_text(
    *,
    text: str,
    speaking_time,
    narrative: str,
    norm_lookup: dict,
) -> dict:
    """
    Compute CoreLex coverage, diversity, and percentile metrics for one sample,
    given raw text + speaking time + narrative name.

    Parameters
    ----------
    text : str
        The sample's full text (already assembled however your pipeline prefers).
    speaking_time : Any
        Speaking time in seconds (ideally numeric). If missing/invalid, CWPM is NaN.
    narrative : str
        Narrative/scene name used to select the correct CoreLex norms.
    norm_lookup : dict
        Preloaded CoreLex norms keyed by narrative.

    Returns
    -------
    dict
        Dictionary of CoreLex metrics and percentile results, plus token-set columns.
        Does NOT include sample_id or partition tier metadata; add those upstream.
    """
    out: dict = {}
    try:
        # ---- validate narrative / norms ----
        if not isinstance(narrative, str) or not narrative.strip():
            logger.warning("CoreLex: missing/invalid narrative; returning empty metrics.")
            return {}

        if not isinstance(norm_lookup, dict) or narrative not in norm_lookup:
            logger.warning(f"CoreLex: norms missing for narrative '{narrative}'; returning empty metrics.")
            return {}

        # ---- sanitize inputs ----
        text = "" if text is None else str(text)
        reformatted_text = reformat(text)

        # speaking_time -> float seconds or NaN
        try:
            st = float(speaking_time) if pd.notnull(speaking_time) else np.nan
        except Exception:
            st = np.nan

        core_stats = id_core_words(narrative, reformatted_text)

        # ---- efficiency (CWPM) ----
        minutes = (st / 60.0) if pd.notnull(st) and st > 0 else np.nan
        cwpm = (
            core_stats["num_core_words"] / minutes
            if pd.notnull(minutes) and minutes > 0
            else np.nan
        )

        # ---- percentiles: accuracy always, efficiency only if CWPM valid ----
        acc_df = norm_lookup[narrative].get("accuracy")
        if acc_df is None:
            logger.warning(f"CoreLex: accuracy norms missing for narrative '{narrative}'.")
            return {}

        acc_pcts = get_percentiles(core_stats["num_core_words"], acc_df, "CoreLex Score")

        if pd.notnull(cwpm):
            eff_df = norm_lookup[narrative].get("efficiency")
            if eff_df is None:
                logger.warning(f"CoreLex: efficiency norms missing for narrative '{narrative}'.")
                cwpm_pwa = cwpm_ctrl = np.nan
            else:
                eff_pcts = get_percentiles(cwpm, eff_df, "CoreLex/min")
                cwpm_pwa, cwpm_ctrl = eff_pcts["pwa_percentile"], eff_pcts["control_percentile"]
        else:
            cwpm_pwa = cwpm_ctrl = np.nan

        # ---- base metrics ----
        out.update(
            {
                "narrative": narrative,
                "speaking_time": st,
                "num_tokens": core_stats.get("num_tokens", np.nan),
                "num_core_words": core_stats.get("num_core_words", np.nan),
                "num_core_word_tokens": core_stats.get("num_cw_tokens", np.nan),
                "lexicon_coverage": core_stats.get("lexicon_coverage", np.nan),
                "core_words_per_min": cwpm,
                "core_words_pwa_percentile": acc_pcts.get("pwa_percentile", np.nan),
                "core_words_control_percentile": acc_pcts.get("control_percentile", np.nan),
                "cwpm_pwa_percentile": cwpm_pwa,
                "cwpm_control_percentile": cwpm_ctrl,
            }
        )

        # ---- token-set columns (scene-prefix + lemma) ----
        token_sets = core_stats.get("token_sets") or {}
        prefix = narrative[:3]
        for lemma, surfaces in token_sets.items():
            try:
                out[f"{prefix}_{lemma}"] = ", ".join(sorted(map(str, surfaces)))
            except Exception:
                # Keep going even if one lemma is weirdly formatted
                out[f"{prefix}_{lemma}"] = ""

        return out

    except Exception as e:
        logger.error(f"Failed CoreLex compute for narrative '{narrative}': {e}")
        return {}


def extract_corelex_inputs_from_sample_df(sample_df: pd.DataFrame) -> dict:
    """
    Extract (text, speaking_time, narrative, sample_id) from a DIAAD-style sample_df.

    Returns
    -------
    dict
        Keys: sample_id, narrative, speaking_time, text
        If required fields are missing, returns {}.
    """
    try:
        if sample_df is None or sample_df.empty:
            return {}

        required = {"utterance", "narrative", "sample_id"}
        missing = [c for c in required if c not in sample_df.columns]
        if missing:
            logger.warning(f"CoreLex: sample_df missing required columns: {missing}")
            return {}

        sample_id = sample_df["sample_id"].iloc[0]
        narrative = sample_df["narrative"].iloc[0]

        # speaking_time is optional
        speaking_time = np.nan
        if "speaking_time" in sample_df.columns:
            speaking_time = sample_df["speaking_time"].iloc[0]

        # join utterances
        text = " ".join(
            u for u in sample_df["utterance"].astype(str).tolist()
            if u and str(u).strip()
        )

        return {
            "sample_id": sample_id,
            "narrative": narrative,
            "speaking_time": speaking_time,
            "text": text,
        }

    except Exception as e:
        logger.error(f"CoreLex: failed extracting inputs from sample_df: {e}")
        return {}


def _compute_corelex_for_sample(sample_df, norm_lookup, partition_tiers, tup):
    """
    Backwards-compatible wrapper that:
      1) extracts inputs from sample_df
      2) computes CoreLex metrics from text/time/narrative
      3) prepends sample_id + partition tiers metadata
    """
    try:
        extracted = extract_corelex_inputs_from_sample_df(sample_df)
        if not extracted:
            return {}

        # start-of-row metadata
        row_prefix = {
            "sample_id": extracted["sample_id"],
            **(dict(zip(partition_tiers, tup)) if partition_tiers else {})
        }

        metrics = compute_corelex_for_text(
            text=extracted["text"],
            speaking_time=extracted["speaking_time"],
            narrative=extracted["narrative"],
            norm_lookup=norm_lookup,
        )
        if not metrics:
            return {}

        row_prefix.update(metrics)
        return row_prefix

    except Exception as e:
        logger.error(f"Failed to compute CoreLex metrics for sample_df: {e}")
        return {}


def run_corelex(tiers, input_dir, output_dir, exclude_participants=None):
    """
    Execute CoreLex analysis on aphasia narratives and export results.
    """
    exclude_participants = set(exclude_participants or [])
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    corelex_dir = output_dir / "core_lex"
    corelex_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"CoreLex output directory: {_rel(corelex_dir)}")

    utt_df, present_narratives = prepare_corelex_inputs(input_dir, output_dir, exclude_participants)
    if utt_df is None:
        return

    partition_tiers = [t.name for t in tiers.values() if getattr(t, "partition", False)]
    norm_lookup = preload_corelex_norms(present_narratives)
    token_columns = generate_token_columns(present_narratives)
    all_columns = [base_columns[0]] + partition_tiers + base_columns[1:] + token_columns
    rows = []

    # Handle no partition tiers gracefully
    grouped = utt_df.groupby(by=partition_tiers) if partition_tiers else [((), utt_df)]

    for tup, subdf in grouped:
        # robust handling when tup is not a tuple (groupby with one key gives scalar)
        tup = tup if isinstance(tup, tuple) else (tup,) if partition_tiers else ()
        for sample in tqdm(sorted(subdf["sample_id"].dropna().unique()), desc="Computing CoreLex"):
            sample_df = subdf[subdf["sample_id"] == sample]
            if sample_df.empty:
                continue
            row = _compute_corelex_for_sample(sample_df, norm_lookup, partition_tiers, tup)
            if row:
                rows.append(row)

    if not rows:
        logger.warning("No CoreLex rows produced; no output written.")
        return

    corelex_df = pd.DataFrame(rows)

    # Ensure requested column order when possible; keep extra cols if any
    ordered_cols = [c for c in all_columns if c in corelex_df.columns]
    extra_cols = [c for c in corelex_df.columns if c not in ordered_cols]
    corelex_df = corelex_df[ordered_cols + extra_cols]

    output_file = corelex_dir / f"core_lex_data_{timestamp}.xlsx"
    try:
        corelex_df.to_excel(output_file, index=False)
        logger.info(f"CoreLex results written to {_rel(output_file)}")
    except Exception as e:
        logger.error(f"Failed to write CoreLex results: {e}")

    logger.info("CoreLex processing complete.")
