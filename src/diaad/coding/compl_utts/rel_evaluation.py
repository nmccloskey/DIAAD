import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from diaad.utils.logger import logger, _rel
from diaad.coding.utils import utt_ct, ptotal, ag_check, compute_cu_column


def summarize_cu_reliability(cu_rel_coding, sv2, rel2, sv3, rel3):
    """
    Aggregate utterance-level CU reliability to the sample level.

    Input
    -----
    cu_relcod : pd.DataFrame
        Merged utterance-level dataframe containing:
        - identifiers: 'utterance_id', 'sample_id'
        - coder-2 columns: sv2, rel2 (names passed in), and computed 'c2_cu'
        - coder-3 columns: sv3, rel3 (names passed in), and computed 'c3_cu'
        - agreement flags: 'agreement_sv', 'agmt_rel', 'agmt_cu' (1 if equal or both NaN, else 0)

    sv2, rel2, sv3, rel3 : str
        Column names for the respective SV/REL fields used in aggregation.

    Returns
    -------
    pd.DataFrame
        One row per sample_id with:
        - Counts per coder (no_utt2/no_utt3, cu_2/cu_3, p_sv*/m_sv*, p_rel*/m_rel*)
        - Percent CU per coder (perc_cu_2, perc_cu_3)
        - Percent agreement on SV/REL/CU (perc_agmt_sv, perc_agmt_rel, perc_agmt_cu)
        - Binary sample-level agreement indicators (sample_agmt_sv, sample_agmt_rel, sample_agmt_cu),
          where 1 indicates ≥80% agreement across utterances, 0 otherwise.
    """
    cu_rel_sum = cu_rel_coding.copy()
    cu_rel_sum.drop(columns=['utterance_id'], inplace=True, errors='ignore')

    try:
        cu_rel_sum = cu_rel_sum.groupby(['sample_id']).agg(
            num_utterances2=('c2_cu', utt_ct),
            plus_sv2=(sv2, ptotal),
            minus_sv2=(sv2, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
            plus_rel2=(rel2, ptotal),
            minus_rel2=(rel2, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
            plus_cu2=('c2_cu', ptotal),
            perc_cu2=('c2_cu', lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),

            num_utterances3=('c3_cu', utt_ct),
            plus_sv3=(sv3, ptotal),
            minus_sv3=(sv3, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
            plus_rel3=(rel3, ptotal),
            minus_rel3=(rel3, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
            plus_cu3=('c3_cu', ptotal),
            perc_cu3=('c3_cu', lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),

            total_ag_sv=('agmt_sv', ptotal),
            perc_agmt_sv=('agmt_sv', lambda x: (ptotal(x) / utt_ct(x)) * 100 if utt_ct(x) > 0 else np.nan),
            total_agmt_rel=('agmt_rel', ptotal),
            perc_agmt_rel=('agmt_rel', lambda x: (ptotal(x) / utt_ct(x)) * 100 if utt_ct(x) > 0 else np.nan),
            total_agmt_cu=('agmt_cu', ptotal),
            perc_agmt_cu=('agmt_cu', lambda x: (ptotal(x) / utt_ct(x)) * 100 if utt_ct(x) > 0 else np.nan),

            sample_agmt_sv=('agmt_sv', ag_check),
            sample_agmt_rel=('agmt_rel', ag_check),
            sample_agmt_cu=('agmt_cu', ag_check)
        ).reset_index()
        logger.info("Successfully aggregated CU reliability data.")
        return cu_rel_sum
    except Exception as e:
        logger.error(f"Failed during CU reliability aggregation: {e}")
        return pd.DataFrame()  # Fail-safe return

def write_reliability_report(cu_rel_sum, report_path, partition_labels=[]):
    """
    Write a plain-text CU reliability summary.

    Input
    -----
    cu_rel_sum : pd.DataFrame
        Sample-level summary produced by `summarize_cu_reliability`, including:
        'sample_agmt_cu', 'perc_agmt_sv', 'perc_agmt_rel', 'perc_agmt_cu'.
    report_path : str | os.PathLike
        Destination .txt filepath.
    partition_labels : list[str] | []
        Optional tier labels (e.g., site/test/participant) to render in the header.

    Side Effects
    ------------
    Creates a text file with:
      - Count and percent of samples meeting ≥80% CU agreement (sample_agmt_cu == 1)
      - Average SV/REL/CU percent agreement across samples

    Logging
    -------
    Logs success or an error if the report cannot be written.
    """
    try:
        num_samples_agmt = np.nansum(cu_rel_sum['sample_agmt_cu'])
        perc_samples_agmt = round(num_samples_agmt / len(cu_rel_sum) * 100, 2)

        with open(report_path, 'w') as report:
            if partition_labels:
                report.write(f"CU Reliability Coding Report for {' '.join(partition_labels)}\n\n")
            else:
                report.write("CU Reliability Coding Report\n\n")

            report.write(f"Coders agree on at least 80% of CUs in {num_samples_agmt} out of {len(cu_rel_sum)} total samples: {perc_samples_agmt}%\n\n")
            report.write(f"Average agreement on SV: {round(np.nanmean(cu_rel_sum['perc_agmt_sv']), 3)}\n")
            report.write(f"Average agreement on REL: {round(np.nanmean(cu_rel_sum['perc_agmt_rel']), 3)}\n")
            report.write(f"Average agreement on CU: {round(np.nanmean(cu_rel_sum['perc_agmt_cu']), 3)}\n")

        logger.info(f"Successfully wrote CU reliability report to {_rel(report_path)}")
    except Exception as e:
        logger.error(f"Failed to write reliability report to {_rel(report_path)}: {e}")

def _write_cu_reliability_outputs(
    cu_rel_coding, partition_labels, base_dir, paradigm, sv2, rel2, sv3, rel3
):
    """Write utterance, summary, and report outputs with relative-path logging."""
    paradigm_str = f"_{paradigm}" if paradigm else ""
    output_path = Path(base_dir, *partition_labels)
    output_path.mkdir(parents=True, exist_ok=True)

    labels_str = "_".join(partition_labels)
    utterance_path = output_path / f"{labels_str}{paradigm_str}_cu_reliability_coding_by_utterance.xlsx"
    cu_rel_coding.to_excel(utterance_path, index=False)
    logger.info(f"Wrote CU reliability utterance file to {_rel(utterance_path)}")

    cu_rel_sum = summarize_cu_reliability(cu_rel_coding, sv2, rel2, sv3, rel3)

    summary_path = output_path / f"{labels_str}{paradigm_str}_cu_reliability_coding_by_sample.xlsx"
    cu_rel_sum.to_excel(summary_path, index=False)
    logger.info(f"Wrote CU reliability summary file to {_rel(summary_path)}")

    report_path = output_path / f"{labels_str}{paradigm_str}_cu_reliability_coding_report.txt"
    write_reliability_report(cu_rel_sum, report_path, partition_labels)
    logger.info(f"Successfully wrote CU reliability report to {_rel(report_path)}")

def evaluate_cu_reliability(tiers, input_dir, output_dir, cu_paradigms):
    """
    Compute and summarize Complete Utterance (CU) reliability across matched
    coder-2 and coder-3 workbooks.

    For each pair of files with identical tier labels:
      - Merge utterance data (c2 vs c3)
      - Compute CU flags and agreement on SV, REL, and CU
      - Write utterance-level, sample-level, and text reports

    Behavior
    --------
    • If 0–1 paradigms: use base columns ('c2_sv', 'c2_rel', 'c3_sv', 'c3_rel').
    • If ≥2 paradigms: iterate per paradigm, using suffixed variants.
    • Outputs written to:
        <output_dir>/cu_reliability[/<PARADIGM>]/<tier_labels>/

    Parameters
    ----------
    tiers : dict[str, Tier]
    input_dir, output_dir : Path or str
    cu_paradigms : list[str]
    """
    cu_reliability_dir = Path(output_dir) / "cu_reliability"
    cu_reliability_dir.mkdir(parents=True, exist_ok=True)

    coding_files = list(Path(input_dir).rglob("*cu_coding.xlsx"))
    rel_files = list(Path(input_dir).rglob("*cu_reliability_coding.xlsx"))

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

            paradigms_to_run = cu_paradigms if len(cu_paradigms) >= 2 else [None]
            for paradigm in paradigms_to_run:
                try:
                    sv2, rel2, sv3, rel3 = (
                        (f"c2_sv_{paradigm}", f"c2_rel_{paradigm}",
                         f"c3_sv_{paradigm}", f"c3_rel_{paradigm}")
                        if paradigm else ("c2_sv", "c2_rel", "c3_sv", "c3_rel")
                    )
                    out_subdir = cu_reliability_dir / (paradigm or "")
                    cu_cod_sub = cu_coding[["utterance_id", "sample_id", sv2, rel2]]
                    cu_rel_sub = cu_rel[["utterance_id", sv3, rel3]]
                    merged = pd.merge(cu_cod_sub, cu_rel_sub, on="utterance_id", how="inner")

                    if len(cu_rel_sub) != len(merged):
                        logger.warning(f"Length mismatch in {_rel(rel)} ({paradigm or 'base'})")

                    merged["c2_cu"] = merged[[sv2, rel2]].apply(compute_cu_column, axis=1)
                    merged["c3_cu"] = merged[[sv3, rel3]].apply(compute_cu_column, axis=1)
                    for pair, new in [
                        ((sv2, sv3), "agmt_sv"),
                        ((rel2, rel3), "agmt_rel"),
                        (("c2_cu", "c3_cu"), "agmt_cu"),
                    ]:
                        merged[new] = merged.apply(
                            lambda r: int((r[pair[0]] == r[pair[1]])
                                          or (pd.isna(r[pair[0]]) and pd.isna(r[pair[1]]))),
                            axis=1,
                        )

                    partition_labels = [t.match(rel.name)
                                        for t in tiers.values() if t.partition]
                    _write_cu_reliability_outputs(
                        merged, partition_labels, out_subdir, paradigm, sv2, rel2, sv3, rel3
                    )

                except Exception as e:
                    logger.error(f"Failed CU reliability for {paradigm or 'base'} on {_rel(rel)}: {e}")
                    continue
