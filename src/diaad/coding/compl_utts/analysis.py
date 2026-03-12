import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from diaad.utils.logger import logger, _rel
from diaad.coding.utils import utt_ct, ptotal, ag_check, compute_cu_column
                    

def _write_cu_analysis_outputs(cu_coding, summaries, out_dir, partition_labels):
    """Write utterance- and sample-level CU analysis files with relative-path logging."""
    label_str = "_".join(partition_labels)
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
        merged = summaries[0]
        for df in summaries[1:]:
            merged = pd.merge(merged, df, on="sample_id", how="outer")
        summary_path = Path(out_dir, f"{label_str}_cu_coding_by_sample.xlsx")
        merged.to_excel(summary_path, index=False)
        logger.info(f"Saved CU summary file: {_rel(summary_path)}")
    except Exception as e:
        logger.error(f"Failed merging or saving CU summary to {_rel(out_dir)}: {e}")

def analyze_cu_coding(tiers, input_dir, output_dir, cu_paradigms=None):
    """
    Summarize coder-2 Complete Utterance (CU) coding by sample and paradigm.

    Behavior
    --------
    • Reads all *cu_coding*.xlsx files under `input_dir`.
    • Computes CU = 1 if SV==REL==1, 0 if both present but not both 1, else NaN.
    • If no paradigms provided, infers from suffixed columns (c2_sv_*).
    • For each paradigm (or None for base), writes:
        - <labels>_cu_coding_by_utterance.xlsx
        - <labels>_cu_coding_by_sample.xlsx

    Parameters
    ----------
    tiers : dict[str, Tier]
    input_dir, output_dir : Path or str
    cu_paradigms : list[str] | None
        Optional explicit list of CU paradigms.
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

        cu_coding.drop(columns=[c for c in ["c1_id", "c1_comment", "c2_id"]
                                if c in cu_coding], inplace=True, errors="ignore")

        # Infer paradigms if none given
        paradigms = cu_paradigms or sorted(
            {c.split("_")[-1] for c in cu_coding if c.startswith("c2_sv_")}
        ) or [None]

        summaries = []
        for paradigm in paradigms:
            sv_col = f"c2_sv_{paradigm}" if paradigm else "c2_sv"
            rel_col = f"c2_rel_{paradigm}" if paradigm else "c2_rel"
            cu_col = f"c2_cu_{paradigm}" if paradigm else "c2_cu"

            if sv_col not in cu_coding or rel_col not in cu_coding:
                logger.warning(f"Skipping {paradigm or 'base'}: columns missing in {_rel(cod)}")
                continue

            cu_coding[cu_col] = cu_coding[[sv_col, rel_col]].apply(compute_cu_column, axis=1)
            # Create summary stats 
            agg_df = cu_coding[['sample_id', sv_col, rel_col, cu_col]].copy()
            agg_df[[sv_col, rel_col, cu_col]] = agg_df[[sv_col, rel_col, cu_col]].apply(pd.to_numeric, errors='coerce')

            try:
                cu_sum = agg_df.groupby("sample_id").agg(
                    **{
                        f"no_utt_{paradigm}": (cu_col, utt_ct),
                        f"p_sv_{paradigm}": (sv_col, ptotal),
                        f"m_sv_{paradigm}": (sv_col, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
                        f"p_rel_{paradigm}": (rel_col, ptotal),
                        f"m_rel_{paradigm}": (rel_col, lambda x: utt_ct(x) - ptotal(x) if utt_ct(x) > 0 else np.nan),
                        f"cu_{paradigm}": (cu_col, ptotal),
                        f"perc_cu_{paradigm}": (cu_col, lambda x: round((ptotal(x) / utt_ct(x)) * 100, 3) if utt_ct(x) > 0 else np.nan),
                    }
                ).reset_index()
                summaries.append(cu_sum)
            except Exception as e:
                logger.error(f"Aggregation failed for {_rel(cod)} ({paradigm or 'base'}): {e}")

        partition_labels = [t.match(cod.name) for t in tiers.values() if t.partition]
        out_dir = Path(cu_analysis_dir, *partition_labels)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_cu_analysis_outputs(cu_coding, summaries, out_dir, partition_labels)
