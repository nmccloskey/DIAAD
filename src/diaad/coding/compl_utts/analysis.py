import re
import numpy as np
import pandas as pd
from pathlib import Path

from psair.core.logger import logger, _rel
from diaad.io.discovery import find_matching_files
from diaad.coding.utils import utt_ct, ptotal, compute_cu_column
from diaad.metadata.unblinding import maybe_unblind_dataframe


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
        sv, rel
        sv_AAE, rel_AAE, ...

    Prefixed:
        c2_sv, c2_rel, c2_sv_AAE, c2_rel_AAE, ...
        c3_sv, c3_rel, c3_sv_AAE, c3_rel_AAE, ...
    """
    columns = set(columns)
    pairs = []

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

    if cu_paradigms is not None:
        allowed = set(cu_paradigms)
        pairs = [p for p in pairs if p["paradigm"] in allowed]

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
    Build wide-format summary column names for a coder/paradigm combination.

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


def _drop_coder_admin_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop coder/admin columns not needed in analysis outputs."""
    df = df.copy()
    df.drop(
        columns=[
            c for c in
            ["id", "comment", "c1_id", "c1_comment", "c2_id", "c2_comment", "c3_id", "c3_comment"]
            if c in df.columns
        ],
        inplace=True,
        errors="ignore",
    )
    return df


def _summarize_pair(cu_coding, pair):
    """
    Summarize one coder/paradigm SV-REL pair at the sample level.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        summary_long, summary_wide
    """
    coder_prefix = pair["coder_prefix"]
    paradigm = pair["paradigm"]
    sv_col = pair["sv_col"]
    rel_col = pair["rel_col"]
    cu_col = _make_output_cu_col(coder_prefix, paradigm)

    coder_label = coder_prefix or "primary"
    paradigm_label = paradigm or "base"

    work_df = cu_coding[["sample_id", sv_col, rel_col]].copy()
    work_df[[sv_col, rel_col]] = work_df[[sv_col, rel_col]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    work_df[cu_col] = work_df[[sv_col, rel_col]].apply(
        compute_cu_column,
        axis=1,
    )

    cu_sum = work_df.groupby("sample_id").agg(
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
        work_df.groupby("sample_id")
        .apply(lambda df: _count_pair_inconsistency(df, sv_col, rel_col))
        .reset_index(name="sv_rel_inconsistent")
    )

    cu_sum = pd.merge(cu_sum, inconsistency_df, on="sample_id", how="left")

    summary_long = cu_sum.copy()
    summary_long["coder"] = coder_label
    summary_long["paradigm"] = paradigm_label
    summary_long["sv_col"] = sv_col
    summary_long["rel_col"] = rel_col
    summary_long["cu_col"] = cu_col

    summary_long = summary_long[
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

    wide_cols = _make_summary_columns(coder_prefix, paradigm)
    summary_wide = cu_sum.rename(columns=wide_cols)

    return summary_long, summary_wide, cu_col


def _combine_wide_summaries(summary_wides):
    """Merge wide summaries across coder/paradigm pairs by sample_id."""
    if not summary_wides:
        return None

    merged = summary_wides[0].copy()
    for df in summary_wides[1:]:
        merged = pd.merge(merged, df, on="sample_id", how="outer")
    return merged


def _maybe_unblind_cu_outputs(
    *,
    cu_coding: pd.DataFrame,
    summary_long: pd.DataFrame | None,
    summary_wide: pd.DataFrame | None,
    blinding_config=None,
    blind_codebook=None,
    input_dir=None,
    output_dir=None,
):
    """
    Unblind sample identifiers in CU analysis outputs if a coding-stage
    blind codebook is available.

    This function does not require transcript tables and does not reblind
    any outputs.
    """
    if blinding_config is None:
        return cu_coding, summary_long, summary_wide, None

    target_cols = ["sample_id"]

    unblinded_cu_coding, codebook_df = maybe_unblind_dataframe(
        df=cu_coding,
        config=blinding_config,
        blind_codebook=blind_codebook,
        target_cols=target_cols,
        directories=[input_dir, output_dir],
        strict=False,
    )

    unblinded_summary_long = None
    if summary_long is not None:
        unblinded_summary_long, _ = maybe_unblind_dataframe(
            df=summary_long,
            config=blinding_config,
            blind_codebook=codebook_df if codebook_df is not None else blind_codebook,
            target_cols=target_cols,
            directories=[input_dir, output_dir],
            strict=False,
        )

    unblinded_summary_wide = None
    if summary_wide is not None:
        unblinded_summary_wide, _ = maybe_unblind_dataframe(
            df=summary_wide,
            config=blinding_config,
            blind_codebook=codebook_df if codebook_df is not None else blind_codebook,
            target_cols=target_cols,
            directories=[input_dir, output_dir],
            strict=False,
        )

    return unblinded_cu_coding, unblinded_summary_long, unblinded_summary_wide, codebook_df


def _write_cu_analysis_outputs(
    cu_coding,
    summary_long,
    summary_wide,
    out_dir,
):
    """Write utterance- and sample-level CU analysis files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    utterance_path = out_dir / "cu_coding_by_utterance.xlsx"
    try:
        cu_coding.to_excel(utterance_path, index=False)
        logger.info(f"Saved utterance-level CU analysis: {_rel(utterance_path)}")
    except Exception as e:
        logger.error(f"Failed writing utterance-level file {_rel(utterance_path)}: {e}")
        return

    if summary_long is None or summary_long.empty:
        logger.warning(f"No valid CU long summaries for {_rel(out_dir)}")
    else:
        try:
            summary_long_path = out_dir / "cu_coding_by_sample_long.xlsx"
            summary_long.to_excel(summary_long_path, index=False)
            logger.info(f"Saved CU long summary file: {_rel(summary_long_path)}")
        except Exception as e:
            logger.error(f"Failed saving CU long summary to {_rel(summary_long_path)}: {e}")

    if summary_wide is None or summary_wide.empty:
        logger.warning(f"No valid CU wide summaries for {_rel(out_dir)}")
    else:
        try:
            summary_wide_path = out_dir / "cu_coding_by_sample.xlsx"
            summary_wide.to_excel(summary_wide_path, index=False)
            logger.info(f"Saved CU wide summary file: {_rel(summary_wide_path)}")
        except Exception as e:
            logger.error(f"Failed saving CU wide summary to {_rel(summary_wide_path)}: {e}")


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def analyze_cu_coding(
    input_dir,
    output_dir,
    cu_paradigms=None,
    blinding_config=None,
    blind_codebook=None,
):
    """
    Summarize Complete Utterance (CU) coding by sample across all available
    coder schemas and paradigms in a CU coding workbook.

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
    - Finds CU coding workbooks.
    - If multiple are found, warns and uses only the first returned file.
    - Detects valid coder/paradigm SV-REL pairs.
    - Computes CU = 1 if SV == REL == 1,
      0 if both are present but not both 1,
      NaN otherwise.
    - Writes:
        * utterance-level file with derived CU columns added
        * long sample-level summary
        * wide sample-level summary

    Unblinding
    ----------
    If a coding-stage blind codebook is available, sample identifiers are
    unblinded in the analysis outputs. No transcript tables are required.
    This function does not reblind outputs.
    """
    cu_analysis_dir = Path(output_dir) / "cu_coding_analysis"
    cu_analysis_dir.mkdir(parents=True, exist_ok=True)

    coding_files = find_matching_files(
        directories=[input_dir, output_dir],
        search_base="cu_coding",
    )

    if not coding_files:
        logger.warning("No CU coding files found for analysis.")
        return

    if len(coding_files) > 1:
        logger.warning(
            "Multiple CU coding files detected for analysis. "
            f"Using only the first returned file: {_rel(coding_files[0])}"
        )

    cod = coding_files[0]

    try:
        cu_coding = pd.read_excel(cod)
        logger.info(f"Processing CU coding file: {_rel(cod)}")
    except Exception as e:
        logger.error(f"Failed reading {_rel(cod)}: {e}")
        return

    cu_coding = _drop_coder_admin_cols(cu_coding)

    coder_pairs = _detect_coder_paradigm_pairs(
        cu_coding.columns,
        cu_paradigms=cu_paradigms,
    )

    if not coder_pairs:
        logger.warning(f"No valid SV/REL coder-paradigm pairs found in {_rel(cod)}")
        return

    summary_longs = []
    summary_wides = []

    for pair in coder_pairs:
        coder_label = pair["coder_prefix"] or "primary"
        paradigm_label = pair["paradigm"] or "base"

        try:
            summary_long, summary_wide, cu_col = _summarize_pair(cu_coding, pair)

            work_df = cu_coding[[pair["sv_col"], pair["rel_col"]]].apply(
                pd.to_numeric,
                errors="coerce",
            )
            cu_coding[cu_col] = work_df.apply(compute_cu_column, axis=1)

            summary_longs.append(summary_long)
            summary_wides.append(summary_wide)

        except Exception as e:
            label = f"{coder_label} / {paradigm_label}"
            logger.error(f"Aggregation failed for {_rel(cod)} ({label}): {e}")

    summary_long = pd.concat(summary_longs, ignore_index=True) if summary_longs else None
    summary_wide = _combine_wide_summaries(summary_wides)

    cu_coding, summary_long, summary_wide, _ = _maybe_unblind_cu_outputs(
        cu_coding=cu_coding,
        summary_long=summary_long,
        summary_wide=summary_wide,
        blinding_config=blinding_config,
        blind_codebook=blind_codebook,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    _write_cu_analysis_outputs(
        cu_coding=cu_coding,
        summary_long=summary_long,
        summary_wide=summary_wide,
        out_dir=cu_analysis_dir,
    )
