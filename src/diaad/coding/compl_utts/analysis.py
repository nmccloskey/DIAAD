import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from src.diaad.core.logger import logger, _rel
from diaad.coding.utils import utt_ct, ptotal, compute_cu_column


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_label_suffix(value):
    """Return a clean suffix for output column naming."""
    return value if value else "base"


def _negative_total(x):
    """Count number of coded 0 values among non-missing entries."""
    n = utt_ct(x)
    p = ptotal(x)
    return n - p if n > 0 else np.nan


def _percent_positive(x):
    """Percent of positive (1) values among non-missing entries."""
    n = utt_ct(x)
    p = ptotal(x)
    return round((p / n) * 100, 3) if n > 0 else np.nan


def _count_missing(x):
    """Count missing values."""
    return int(x.isna().sum()) if len(x) > 0 else np.nan


def _count_pair_inconsistency(df, sv_col, rel_col):
    """
    Count rows where exactly one of SV/REL is missing.

    These rows are operationally inconsistent for CU derivation.
    """
    return int(((df[sv_col].isna()) ^ (df[rel_col].isna())).sum())


def _detect_coder_paradigm_pairs(columns, cu_paradigms=None):
    """
    Detect available (coder_prefix, paradigm) pairs from workbook columns.

    Supported column families
    -------------------------
    Unprefixed:
        sv, rel, sv_AAE, rel_AAE, ...

    Prefixed:
        c2_sv, c2_rel, c2_sv_AAE, c2_rel_AAE, ...
        c3_sv, c3_rel, c3_sv_AAE, c3_rel_AAE, ...

    Returns
    -------
    list[dict]
        Each entry has:
        {
            "coder_prefix": str | None,   # None for unprefixed schema
            "paradigm": str | None,
            "sv_col": str,
            "rel_col": str,
        }
    """
    columns = set(columns)
    pairs = []

    # Unprefixed base / paradigm columns
    if "sv" in columns and "rel" in columns:
        pairs.append(
            {
                "coder_prefix": None,
                "paradigm": None,
                "sv_col": "sv",
                "rel_col": "rel",
            }
        )

    unprefixed_sv = [c for c in columns if re.fullmatch(r"sv_.+", c)]
    for sv_col in sorted(unprefixed_sv):
        paradigm = sv_col[3:]
        rel_col = f"rel_{paradigm}"
        if rel_col in columns:
            pairs.append(
                {
                    "coder_prefix": None,
                    "paradigm": paradigm,
                    "sv_col": sv_col,
                    "rel_col": rel_col,
                }
            )

    # Prefixed coder columns, e.g. c2_sv, c3_sv_AAE
    prefixed_sv = [c for c in columns if re.fullmatch(r"c\d+_sv(?:_.+)?", c)]
    for sv_col in sorted(prefixed_sv):
        match = re.fullmatch(r"(c\d+)_sv(?:_(.+))?", sv_col)
        if not match:
            continue

        coder_prefix = match.group(1)
        paradigm = match.group(2) if match.group(2) else None
        rel_col = f"{coder_prefix}_rel" + (f"_{paradigm}" if paradigm else "")

        if rel_col in columns:
            pairs.append(
                {
                    "coder_prefix": coder_prefix,
                    "paradigm": paradigm,
                    "sv_col": sv_col,
                    "rel_col": rel_col,
                }
            )

    # Optional filter by explicit paradigms
    if cu_paradigms is not None:
        allowed = set(cu_paradigms)
        pairs = [p for p in pairs if p["paradigm"] in allowed]

    # Deduplicate while preserving order
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = (p["coder_prefix"], p["paradigm"], p["sv_col"], p["rel_col"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    return unique_pairs


def _make_output_cu_col(coder_prefix, paradigm):
    """
    Build a derived CU column name.

    Examples
    --------
    None, None   -> cu
    None, 'AAE'  -> cu_AAE
    'c2', None   -> c2_cu
    'c2', 'AAE'  -> c2_cu_AAE
    """
    if coder_prefix and paradigm:
        return f"{coder_prefix}_cu_{paradigm}"
    if coder_prefix and not paradigm:
        return f"{coder_prefix}_cu"
    if not coder_prefix and paradigm:
        return f"cu_{paradigm}"
    return "cu"


def _make_summary_columns(coder_prefix, paradigm):
    """
    Build output summary column names for a coder/paradigm combination.

    Uses 'base' for the unsuffixed paradigm in summary outputs to avoid
    names like 'no_utt_None'.
    """
    coder_tag = coder_prefix or "primary"
    paradigm_tag = _safe_label_suffix(paradigm)
    suffix = f"{coder_tag}_{paradigm_tag}"

    return {
        "no_utt": f"no_utt_{suffix}",
        "p_sv": f"p_sv_{suffix}",
        "m_sv": f"m_sv_{suffix}",
        "perc_sv": f"perc_sv_{suffix}",
        "miss_sv": f"miss_sv_{suffix}",
        "p_rel": f"p_rel_{suffix}",
        "m_rel": f"m_rel_{suffix}",
        "perc_rel": f"perc_rel_{suffix}",
        "miss_rel": f"miss_rel_{suffix}",
        "cu": f"cu_{suffix}",
        "perc_cu": f"perc_cu_{suffix}",
        "miss_cu": f"miss_cu_{suffix}",
        "sv_rel_inconsistent": f"sv_rel_inconsistent_{suffix}",
    }


# ---------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------

def _write_cu_analysis_outputs(cu_coding, summaries, out_dir, partition_labels):
    """Write utterance- and sample-level CU analysis files with relative-path logging."""
    label_str = "_".join(partition_labels) if partition_labels else "all_samples"
    utterance_path = Path(out_dir, f"{label_str}_cu_coding_by_utterance.xlsx")

    try:
        cu_coding.to_excel(utterance_path, index=False)
        logger.info(f"Saved utterance-level CU analysis: {_rel(utterance_path)}")
    except Exception as e:
        logger.error(f"Failed writing utterance-level file {_rel(utterance_path)}: {e}")
        return

    if not summaries:
        logger.warning(f"No valid CU summaries for {_rel(out_dir)}")
        return

    try:
        summary_long = pd.concat(summaries, ignore_index=True)
        summary_path = Path(out_dir, f"{label_str}_cu_coding_by_sample.xlsx")
        summary_long.to_excel(summary_path, index=False)
        logger.info(f"Saved CU summary file: {_rel(summary_path)}")
    except Exception as e:
        logger.error(f"Failed saving CU summary to {_rel(out_dir)}: {e}")

# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def analyze_cu_coding(tiers, input_dir, output_dir, cu_paradigms=None):
    """
    Summarize Complete Utterance (CU) coding by sample across all available
    coder schemas and paradigms in each workbook.

    Supported input schemas
    -----------------------
    Unprefixed:
        sv, rel
        sv_{paradigm}, rel_{paradigm}

    Prefixed:
        cN_sv, cN_rel
        cN_sv_{paradigm}, cN_rel_{paradigm}

    Behavior
    --------
    - Reads all *cu_coding*.xlsx files under `input_dir`.
    - Detects all valid coder/paradigm SV-REL pairs.
    - Computes CU = 1 if SV == REL == 1,
      0 if both are present but not both 1,
      NaN otherwise.
    - Writes:
        * utterance-level file with derived CU columns added
        * sample-level merged summary across all detected coder/paradigm pairs

    Parameters
    ----------
    tiers : dict[str, Tier]
    input_dir, output_dir : Path or str
    cu_paradigms : list[str] | None
        Optional explicit list of paradigms to include. If None, infer from columns.
    """
    cu_analysis_dir = Path(output_dir) / "cu_coding_analysis"
    cu_analysis_dir.mkdir(parents=True, exist_ok=True)

    coding_files = list(Path(input_dir).rglob("*cu_coding*.xlsx"))

    for cod in tqdm(coding_files, desc="Analyzing CU coding..."):
        try:
            cu_coding = pd.read_excel(cod)
            logger.info(f"Processing CU coding file: {_rel(cod)}")
        except Exception as e:
            logger.error(f"Failed reading {_rel(cod)}: {e}")
            continue

        cu_coding.drop(
            columns=[c for c in ["c1_id", "c1_comment", "c2_id", "c2_comment", "c3_id", "c3_comment"] if c in cu_coding],
            inplace=True,
            errors="ignore",
        )

        coder_pairs = _detect_coder_paradigm_pairs(cu_coding.columns, cu_paradigms=cu_paradigms)

        if not coder_pairs:
            logger.warning(f"No valid SV/REL coder-paradigm pairs found in {_rel(cod)}")
            continue

        summaries = []

        for pair in coder_pairs:
            coder_prefix = pair["coder_prefix"]
            paradigm = pair["paradigm"]
            sv_col = pair["sv_col"]
            rel_col = pair["rel_col"]
            cu_col = _make_output_cu_col(coder_prefix, paradigm)

            coder_label = coder_prefix or "primary"
            paradigm_label = paradigm or "base"

            try:
                cu_coding[[sv_col, rel_col]] = cu_coding[[sv_col, rel_col]].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                cu_coding[cu_col] = cu_coding[[sv_col, rel_col]].apply(compute_cu_column, axis=1)

                agg_df = cu_coding[["sample_id", sv_col, rel_col, cu_col]].copy()

                cu_sum = agg_df.groupby("sample_id").agg(
                    no_utt=(cu_col, utt_ct),

                    p_sv=(sv_col, ptotal),
                    m_sv=(sv_col, _negative_total),
                    perc_sv=(sv_col, _percent_positive),
                    miss_sv=(sv_col, _count_missing),

                    p_rel=(rel_col, ptotal),
                    m_rel=(rel_col, _negative_total),
                    perc_rel=(rel_col, _percent_positive),
                    miss_rel=(rel_col, _count_missing),

                    cu=(cu_col, ptotal),
                    perc_cu=(cu_col, _percent_positive),
                    miss_cu=(cu_col, _count_missing),
                ).reset_index()

                inconsistency_df = (
                    cu_coding.groupby("sample_id")
                    .apply(lambda df: _count_pair_inconsistency(df, sv_col, rel_col))
                    .reset_index(name="sv_rel_inconsistent")
                )

                cu_sum = pd.merge(cu_sum, inconsistency_df, on="sample_id", how="left")

                cu_sum["coder"] = coder_label
                cu_sum["paradigm"] = paradigm_label
                cu_sum["sv_col"] = sv_col
                cu_sum["rel_col"] = rel_col
                cu_sum["cu_col"] = cu_col

                # optional column ordering
                cu_sum = cu_sum[
                    [
                        "sample_id",
                        "coder",
                        "paradigm",
                        "sv_col",
                        "rel_col",
                        "cu_col",
                        "no_utt",
                        "p_sv",
                        "m_sv",
                        "perc_sv",
                        "miss_sv",
                        "p_rel",
                        "m_rel",
                        "perc_rel",
                        "miss_rel",
                        "cu",
                        "perc_cu",
                        "miss_cu",
                        "sv_rel_inconsistent",
                    ]
                ]

                summaries.append(cu_sum)

            except Exception as e:
                label = f"{coder_label} / {paradigm_label}"
                logger.error(f"Aggregation failed for {_rel(cod)} ({label}): {e}")

        partition_labels = [t.match(cod.name) for t in tiers.values() if t.partition]
        out_dir = Path(cu_analysis_dir, *partition_labels)
        out_dir.mkdir(parents=True, exist_ok=True)

        _write_cu_analysis_outputs(cu_coding, summaries, out_dir, partition_labels)
