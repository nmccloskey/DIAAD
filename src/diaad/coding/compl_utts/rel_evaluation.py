import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from pingouin import intraclass_corr
from sklearn.metrics import cohen_kappa_score

from diaad.utils.logger import logger, _rel
from diaad.coding.utils import utt_ct, ptotal, ag_check, compute_cu_column


# ---------------------------------------------------------------------
# Reliability helpers
# ---------------------------------------------------------------------

def _paired_nonmissing(df, col1, col2):
    """Return rows where both columns are non-missing."""
    return df[df[col1].notna() & df[col2].notna()].copy()


def _raw_agreement(x, y):
    """
    Compute raw percent agreement for two aligned categorical series.

    Returns
    -------
    float
        Percent agreement in [0, 100], or np.nan if no paired values.
    """
    if len(x) == 0:
        return np.nan
    return round((x.to_numpy() == y.to_numpy()).mean() * 100, 3)


def _safe_kappa(x, y):
    """
    Compute Cohen's kappa for two aligned categorical series.

    Returns
    -------
    float
        Cohen's kappa, or np.nan if undefined / uncomputable.
    """
    if len(x) == 0:
        return np.nan

    # If there is no variability at all across paired ratings, kappa is not informative.
    observed = pd.concat([pd.Series(x), pd.Series(y)], ignore_index=True).dropna().unique()
    if len(observed) < 2:
        return np.nan

    try:
        return round(cohen_kappa_score(x, y), 6)
    except Exception as e:
        logger.warning(f"Failed to compute Cohen's kappa: {e}")
        return np.nan


def _compute_measure_stats(df, col1, col2):
    """
    Compute utterance-level reliability statistics for a pair of coder columns.

    Returns
    -------
    dict
        {
            'paired_n': int,
            'positive_1': float | np.nan,
            'positive_2': float | np.nan,
            'raw_agreement': float | np.nan,
            'kappa': float | np.nan,
        }
    """
    paired = _paired_nonmissing(df, col1, col2)

    if paired.empty:
        return {
            "paired_n": 0,
            "positive_1": np.nan,
            "positive_2": np.nan,
            "raw_agreement": np.nan,
            "kappa": np.nan,
        }

    x = paired[col1]
    y = paired[col2]

    return {
        "paired_n": int(len(paired)),
        "positive_1": float(np.nansum(x == 1)),
        "positive_2": float(np.nansum(y == 1)),
        "raw_agreement": _raw_agreement(x, y),
        "kappa": _safe_kappa(x, y),
    }


def _compute_icc_for_totals(summary_df, col2, col3):
    """
    Compute ICC(2,1) and ICC(2,k) on paired sample-level totals.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Sample-level summary dataframe.
    col2, col3 : str
        Column names holding coder-2 / coder-3 sample totals.

    Returns
    -------
    dict
        {
            'paired_samples': int,
            'icc2_1': float | np.nan,
            'icc2_k': float | np.nan,
        }
    """
    paired = summary_df[["sample_id", col2, col3]].dropna().copy()

    if len(paired) < 2:
        return {
            "paired_samples": int(len(paired)),
            "icc2_1": np.nan,
            "icc2_k": np.nan,
        }

    icc_long = pd.concat(
        [
            paired[["sample_id", col2]].rename(columns={col2: "score"}).assign(coder="coder_2"),
            paired[["sample_id", col3]].rename(columns={col3: "score"}).assign(coder="coder_3"),
        ],
        ignore_index=True,
    )

    try:
        icc = intraclass_corr(
            data=icc_long,
            targets="sample_id",
            raters="coder",
            ratings="score",
        )

        icc2_1 = icc.loc[icc["Type"] == "ICC2", "ICC"]
        icc2_k = icc.loc[icc["Type"] == "ICC2k", "ICC"]

        return {
            "paired_samples": int(len(paired)),
            "icc2_1": round(float(icc2_1.iloc[0]), 6) if not icc2_1.empty else np.nan,
            "icc2_k": round(float(icc2_k.iloc[0]), 6) if not icc2_k.empty else np.nan,
        }
    except Exception as e:
        logger.warning(f"Failed to compute ICC for {col2}/{col3}: {e}")
        return {
            "paired_samples": int(len(paired)),
            "icc2_1": np.nan,
            "icc2_k": np.nan,
        }


def _resolve_coder_columns(cu_coding, cu_rel, paradigm=None):
    """
    Resolve source columns for the two supported CU reliability schemas.

    Supported modes
    ---------------
    1. primary_vs_reliability
       - cu_coding: sv / rel or sv_{paradigm} / rel_{paradigm}
       - cu_rel:    sv / rel or sv_{paradigm} / rel_{paradigm}

    2. coder2_vs_coder3
       - cu_coding: c2_sv / c2_rel or c2_sv_{paradigm} / c2_rel_{paradigm}
       - cu_rel:    c3_sv / c3_rel or c3_sv_{paradigm} / c3_rel_{paradigm}

    Returns
    -------
    dict
        {
            'mode': str,
            'coding_sv': str,
            'coding_rel': str,
            'rel_sv': str,
            'rel_rel': str,
        }

    Raises
    ------
    KeyError
        If no supported column schema is found.
    """
    suffix = f"_{paradigm}" if paradigm else ""

    candidates = [
        {
            "mode": "primary_vs_reliability",
            "coding_sv": f"sv{suffix}",
            "coding_rel": f"rel{suffix}",
            "rel_sv": f"sv{suffix}",
            "rel_rel": f"rel{suffix}",
        },
        {
            "mode": "coder2_vs_coder3",
            "coding_sv": f"c2_sv{suffix}",
            "coding_rel": f"c2_rel{suffix}",
            "rel_sv": f"c3_sv{suffix}",
            "rel_rel": f"c3_rel{suffix}",
        },
    ]

    for cand in candidates:
        needed_left = {cand["coding_sv"], cand["coding_rel"]}
        needed_right = {cand["rel_sv"], cand["rel_rel"]}

        if needed_left.issubset(cu_coding.columns) and needed_right.issubset(cu_rel.columns):
            return cand

    raise KeyError(
        f"Could not resolve reliability columns for paradigm={paradigm!r}. "
        f"Available coding columns: {list(cu_coding.columns)}; "
        f"available reliability columns: {list(cu_rel.columns)}"
    )


# ---------------------------------------------------------------------
# Sample-level summary
# ---------------------------------------------------------------------

def summarize_cu_reliability(cu_rel_coding):
    """
    Aggregate utterance-level CU reliability to the sample level and compute
    overall reliability statistics.

    Expected input columns
    ----------------------
    Canonical utterance-level columns:
      - utterance_id
      - sample_id
      - c2_sv, c2_rel, c2_cu
      - c3_sv, c3_rel, c3_cu
      - agmt_sv, agmt_rel, agmt_cu

    Returns
    -------
    tuple[pd.DataFrame, dict]
        sample_summary, overall_stats

    Notes
    -----
    - Legacy percent-agreement and >=80% agreement flags are retained as
      descriptive QC outputs.
    - Primary reliability metrics are:
        * Cohen's kappa at the utterance level
        * ICC(2,1) and ICC(2,k) on sample totals
    """
    cu_rel_sum = cu_rel_coding.copy()
    cu_rel_sum.drop(columns=["utterance_id"], inplace=True, errors="ignore")

    try:
        cu_rel_sum = cu_rel_sum.groupby("sample_id").agg(
            num_utterances2=("c2_cu", utt_ct),
            plus_sv2=("c2_sv", ptotal),
            minus_sv2=("c2_sv", lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
            plus_rel2=("c2_rel", ptotal),
            minus_rel2=("c2_rel", lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
            plus_cu2=("c2_cu", ptotal),
            perc_cu2=("c2_cu", lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),

            num_utterances3=("c3_cu", utt_ct),
            plus_sv3=("c3_sv", ptotal),
            minus_sv3=("c3_sv", lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
            plus_rel3=("c3_rel", ptotal),
            minus_rel3=("c3_rel", lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
            plus_cu3=("c3_cu", ptotal),
            perc_cu3=("c3_cu", lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),

            total_agmt_sv=("agmt_sv", ptotal),
            perc_agmt_sv=("agmt_sv", lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),
            total_agmt_rel=("agmt_rel", ptotal),
            perc_agmt_rel=("agmt_rel", lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),
            total_agmt_cu=("agmt_cu", ptotal),
            perc_agmt_cu=("agmt_cu", lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),

            sample_agmt_sv=("agmt_sv", ag_check),
            sample_agmt_rel=("agmt_rel", ag_check),
            sample_agmt_cu=("agmt_cu", ag_check),
        ).reset_index()

        overall_stats = {
            "utterance_level": {
                "sv": _compute_measure_stats(cu_rel_coding, "c2_sv", "c3_sv"),
                "rel": _compute_measure_stats(cu_rel_coding, "c2_rel", "c3_rel"),
                "cu": _compute_measure_stats(cu_rel_coding, "c2_cu", "c3_cu"),
            },
            "sample_totals": {
                "sv": _compute_icc_for_totals(cu_rel_sum, "plus_sv2", "plus_sv3"),
                "rel": _compute_icc_for_totals(cu_rel_sum, "plus_rel2", "plus_rel3"),
                "cu": _compute_icc_for_totals(cu_rel_sum, "plus_cu2", "plus_cu3"),
            },
        }

        logger.info("Successfully aggregated CU reliability data.")
        return cu_rel_sum, overall_stats

    except Exception as e:
        logger.error(f"Failed during CU reliability aggregation: {e}")
        return pd.DataFrame(), {}


# ---------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------

def write_reliability_report(
    cu_rel_sum,
    overall_stats,
    report_path,
    partition_labels=None,
    comparison_mode=None,
    paradigm=None,
):
    """
    Write a plain-text CU reliability summary.

    Parameters
    ----------
    cu_rel_sum : pd.DataFrame
        Sample-level summary produced by `summarize_cu_reliability`.
    overall_stats : dict
        Nested dictionary with utterance-level kappa/raw agreement and
        sample-total ICC metrics.
    report_path : str | os.PathLike
        Destination .txt filepath.
    partition_labels : list[str] | None
        Optional partition labels for the report header.
    comparison_mode : str | None
        Resolved schema label, e.g. 'primary_vs_reliability' or 'coder2_vs_coder3'.
    paradigm : str | None
        Optional coder paradigm label.
    """
    partition_labels = partition_labels or []

    try:
        with open(report_path, "w", encoding="utf-8") as report:
            if partition_labels:
                report.write(f"CU Reliability Coding Report for {' '.join(partition_labels)}\n")
            else:
                report.write("CU Reliability Coding Report\n")

            if comparison_mode:
                report.write(f"Comparison mode: {comparison_mode}\n")
            if paradigm:
                report.write(f"Paradigm: {paradigm}\n")
            report.write("\n")

            # ---------------------------------------------------------
            # Primary metrics
            # ---------------------------------------------------------
            report.write("Primary reliability metrics\n")
            report.write("---------------------------\n\n")

            report.write("Utterance-level categorical reliability\n")
            for measure in ["sv", "rel", "cu"]:
                stats = overall_stats.get("utterance_level", {}).get(measure, {})
                report.write(
                    f"{measure.upper()}: "
                    f"paired utterances={stats.get('paired_n', np.nan)}, "
                    f"raw agreement={stats.get('raw_agreement', np.nan)}, "
                    f"Cohen's kappa={stats.get('kappa', np.nan)}\n"
                )
            report.write("\n")

            report.write("Sample-total reliability\n")
            for measure in ["sv", "rel", "cu"]:
                stats = overall_stats.get("sample_totals", {}).get(measure, {})
                report.write(
                    f"{measure.upper()} totals: "
                    f"paired samples={stats.get('paired_samples', np.nan)}, "
                    f"ICC(2,1)={stats.get('icc2_1', np.nan)}, "
                    f"ICC(2,k)={stats.get('icc2_k', np.nan)}\n"
                )
            report.write("\n")

            # ---------------------------------------------------------
            # Legacy descriptive metrics
            # ---------------------------------------------------------
            report.write("Legacy descriptive agreement metrics\n")
            report.write("-----------------------------------\n\n")

            for measure in ["sv", "rel", "cu"]:
                col = f"sample_agmt_{measure}"
                num_samples_agmt = np.nansum(cu_rel_sum[col]) if col in cu_rel_sum else np.nan
                denom = len(cu_rel_sum) if len(cu_rel_sum) > 0 else np.nan
                perc_samples_agmt = round(num_samples_agmt / denom * 100, 2) if denom and not pd.isna(denom) else np.nan

                report.write(
                    f"Samples with >=80% agreement on {measure.upper()}: "
                    f"{num_samples_agmt} out of {denom} ({perc_samples_agmt}%)\n"
                )

            report.write("\n")
            report.write(f"Average percent agreement on SV: {round(np.nanmean(cu_rel_sum['perc_agmt_sv']), 3)}\n")
            report.write(f"Average percent agreement on REL: {round(np.nanmean(cu_rel_sum['perc_agmt_rel']), 3)}\n")
            report.write(f"Average percent agreement on CU: {round(np.nanmean(cu_rel_sum['perc_agmt_cu']), 3)}\n")

        logger.info(f"Successfully wrote CU reliability report to {_rel(report_path)}")

    except Exception as e:
        logger.error(f"Failed to write reliability report to {_rel(report_path)}: {e}")


# ---------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------

def _write_cu_reliability_outputs(
    cu_rel_coding,
    partition_labels,
    base_dir,
    paradigm,
    comparison_mode,
):
    """
    Write utterance-level, sample-level, and report outputs.

    Notes
    -----
    `cu_rel_coding` is expected to contain canonical internal columns:
      c2_sv, c2_rel, c2_cu, c3_sv, c3_rel, c3_cu
    regardless of whether source files used unprefixed or c2/c3-prefixed names.
    """
    paradigm_str = f"_{paradigm}" if paradigm else ""
    output_path = Path(base_dir, *partition_labels)
    output_path.mkdir(parents=True, exist_ok=True)

    labels_str = "_".join(partition_labels) if partition_labels else "all_samples"

    utterance_path = output_path / f"{labels_str}{paradigm_str}_cu_reliability_coding_by_utterance.xlsx"
    cu_rel_coding.to_excel(utterance_path, index=False)
    logger.info(f"Wrote CU reliability utterance file to {_rel(utterance_path)}")

    cu_rel_sum, overall_stats = summarize_cu_reliability(cu_rel_coding)

    summary_path = output_path / f"{labels_str}{paradigm_str}_cu_reliability_coding_by_sample.xlsx"
    cu_rel_sum.to_excel(summary_path, index=False)
    logger.info(f"Wrote CU reliability summary file to {_rel(summary_path)}")

    report_path = output_path / f"{labels_str}{paradigm_str}_cu_reliability_coding_report.txt"
    write_reliability_report(
        cu_rel_sum=cu_rel_sum,
        overall_stats=overall_stats,
        report_path=report_path,
        partition_labels=partition_labels,
        comparison_mode=comparison_mode,
        paradigm=paradigm,
    )


# ---------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------

def evaluate_cu_reliability(tiers, input_dir, output_dir, cu_paradigms):
    """
    Compute and summarize Complete Utterance (CU) reliability across matched
    coding and reliability workbooks.

    Supported comparison modes
    --------------------------
    1. primary_vs_reliability
       - coding workbook uses: sv / rel or sv_{paradigm} / rel_{paradigm}
       - reliability workbook uses the same unprefixed structure

    2. coder2_vs_coder3
       - coding workbook uses: c2_sv / c2_rel or c2_sv_{paradigm} / c2_rel_{paradigm}
       - reliability workbook uses: c3_sv / c3_rel or c3_sv_{paradigm} / c3_rel_{paradigm}

    Outputs
    -------
    Written to:
      <output_dir>/cu_reliability[/<PARADIGM>]/<partition_labels>/

    Notes
    -----
    - Internal merged data are canonicalized to c2_* and c3_* columns so that
      downstream summarization/reporting is schema-agnostic.
    - If `cu_paradigms` is empty, the base (non-suffixed) columns are used.
    - If `cu_paradigms` contains one or more paradigms, each is evaluated.
    """
    cu_reliability_dir = Path(output_dir) / "cu_reliability"
    cu_reliability_dir.mkdir(parents=True, exist_ok=True)

    coding_files = list(Path(input_dir).rglob("*cu_coding.xlsx"))
    rel_files = list(Path(input_dir).rglob("*cu_reliability_coding.xlsx"))

    paradigms_to_run = cu_paradigms if cu_paradigms else [None]

    for rel in tqdm(rel_files, desc="Analyzing CU reliability..."):
        rel_labels = [t.match(rel.name, return_none=True) for t in tiers.values()]

        for cod in coding_files:
            cod_labels = [t.match(cod.name, return_none=True) for t in tiers.values()]
            if rel_labels != cod_labels:
                continue

            try:
                cu_coding = pd.read_excel(cod)
                cu_rel = pd.read_excel(rel)
                logger.info(f"Processing pair: {_rel(cod)} + {_rel(rel)}")
            except Exception as e:
                logger.error(f"Failed reading {_rel(cod)} or {_rel(rel)}: {e}")
                continue

            for paradigm in paradigms_to_run:
                try:
                    resolved = _resolve_coder_columns(cu_coding, cu_rel, paradigm=paradigm)
                    comparison_mode = resolved["mode"]

                    out_subdir = cu_reliability_dir / (paradigm or "")

                    left = cu_coding[
                        ["utterance_id", "sample_id", resolved["coding_sv"], resolved["coding_rel"]]
                    ].rename(
                        columns={
                            resolved["coding_sv"]: "c2_sv",
                            resolved["coding_rel"]: "c2_rel",
                        }
                    )

                    right = cu_rel[
                        ["utterance_id", resolved["rel_sv"], resolved["rel_rel"]]
                    ].rename(
                        columns={
                            resolved["rel_sv"]: "c3_sv",
                            resolved["rel_rel"]: "c3_rel",
                        }
                    )

                    merged = pd.merge(left, right, on="utterance_id", how="inner")

                    if len(right) != len(merged):
                        logger.warning(
                            f"Length mismatch in {_rel(rel)} "
                            f"({paradigm or 'base'}, {comparison_mode})"
                        )

                    merged["c2_cu"] = merged[["c2_sv", "c2_rel"]].apply(compute_cu_column, axis=1)
                    merged["c3_cu"] = merged[["c3_sv", "c3_rel"]].apply(compute_cu_column, axis=1)

                    for pair, new in [
                        (("c2_sv", "c3_sv"), "agmt_sv"),
                        (("c2_rel", "c3_rel"), "agmt_rel"),
                        (("c2_cu", "c3_cu"), "agmt_cu"),
                    ]:
                        merged[new] = merged.apply(
                            lambda r: int(
                                (r[pair[0]] == r[pair[1]])
                                or (pd.isna(r[pair[0]]) and pd.isna(r[pair[1]]))
                            ),
                            axis=1,
                        )

                    partition_labels = [t.match(rel.name) for t in tiers.values() if t.partition]

                    _write_cu_reliability_outputs(
                        cu_rel_coding=merged,
                        partition_labels=partition_labels,
                        base_dir=out_subdir,
                        paradigm=paradigm,
                        comparison_mode=comparison_mode,
                    )

                except KeyError as e:
                    logger.info(
                        f"Skipping paradigm {paradigm or 'base'} for {_rel(rel)} "
                        f"because required columns were not found: {e}"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Failed CU reliability for {paradigm or 'base'} on {_rel(rel)}: {e}"
                    )
                    continue
