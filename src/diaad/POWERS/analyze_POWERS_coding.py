import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pingouin import intraclass_corr
from sklearn.metrics import cohen_kappa_score

def match_reliability_files(input_dir, output_dir):
    """
    Match and merge POWERS coding files with reliability coding files.

    This function searches `input_dir` for baseline POWERS coding Excel files
    (*POWERS_Coding*.xlsx) and their corresponding reliability re-coding files
    (*POWERS_ReliabilityCoding*.xlsx). For each matched pair, it merges data on
    utterance_id and sample_id, drops coder-1 columns, and writes a new merged
    file into `{output_dir}/POWERS_ReliabilityAnalysis`.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing both coding and reliability coding files.
    output_dir : str or Path
        Root directory where merged files will be saved.

    Returns
    -------
    None
        Writes merged Excel files named
        *POWERS_ReliabilityCoding_Merged*.xlsx into the reliability directory.
    """

    # Make POWERS Reliability Analysis folder.
    POWERS_Reliability_dir = os.path.join(output_dir, 'POWERS_ReliabilityAnalysis')
    try:
        os.makedirs(POWERS_Reliability_dir, exist_ok=True)
        logging.info(f"Created directory: {POWERS_Reliability_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory {POWERS_Reliability_dir}: {e}")
        return

    coding_files = [f for f in Path(input_dir).rglob('*POWERS_Coding*.xlsx')]
    rel_files = [f for f in Path(input_dir).rglob('*POWERS_ReliabilityCoding*.xlsx')]

    # Match original coding and reliability files.
    for rel in tqdm(rel_files, desc="Analyzing POWERS reliability coding..."):
        for cod in coding_files:
            if rel.name.replace("POWERS_Coding", "POWERS_ReliabilityCoding") == cod.name:
                try:
                    PCcod = pd.read_excel(cod)
                    PCrel = pd.read_excel(rel)
                    logging.info(f"Processing coding file: {cod} and reliability file: {rel}")
                except Exception as e:
                    logging.error(f"Failed to read files {cod} or {rel}: {e}")
                    continue
    
                merged = PCcod.merge(PCrel, on=["utterance_id", "sample_id"], how="inner")
                merged.drop(columns=[col for col in merged.columns if col.startswith("c1")], inplace=True)

                merged_filename = os.path.join(POWERS_Reliability_dir, rel.name.replace("POWERS_ReliabilityCoding", "POWERS_ReliabilityCoding_Merged"))

                try:
                    os.makedirs(os.path.dirname(merged_filename), exist_ok=True)
                    merged.to_excel(merged_filename, index=False)
                    logging.info(f"Wrote merged POWERS coding & reliability file: {merged_filename}")
                except Exception as e:
                    logging.error(f"Failed to write merged POWERS coding & reliability file {merged_filename}: {e}")

def number_turns(turn_type_col):
    """Assign sequential numeric labels to turns of type T, MT, or ST."""
    new_col = []
    turn_counts = {"T": 0, "MT": 0, "ST": 0}
    for t in turn_type_col:
        if t not in turn_counts:
            try:
                new_t = new_col[-1]
            except Exception as e:
                print(
                    f"Blank turn cell cannot inherit previous cell's value: {e}. Marking as error (X)"
                )
                new_t = "X"
        else:
            turn_counts[t] += 1
            new_t = f"{t}{turn_counts.get(t, 'X')}"
        new_col.append(new_t)
    return new_col

def count_value(val):
    def inner(series):
        return np.sum(series == val)
    return inner

TURN_AGG_COLS = ["speech_units", "content_words", "num_nouns", "filled_pauses"]


def analyze_POWERS_coding(input_dir, output_dir, reliability=False):
    """
    Analyze POWERS coding files to generate turn-, speaker-, and dialog-level
    aggregates and assess inter-coder reliability.

    This function scans `input_dir` recursively for POWERS coding files
    (`*POWERS_Coding*.xlsx`), computes aggregated metrics, and writes results
    into a structured analysis directory under `output_dir`.

    Processing steps
    ----------------
    1. **Turn labeling**: Adds sequential turn labels (e.g., T1, MT2) for
       coders one and two, enabling grouping of utterances into discrete turns.
    2. **Aggregations**:
       - **Turn-level summary**: Aggregates speech unit metrics by sample,
         speaker, and labeled turn.
       - **Speaker-level summary**: Aggregates metrics per sample × speaker.
       - **Dialog-level summary**: Aggregates metrics across the entire sample.
    3. **Reliability analysis**:
       - Computes ICC(2,1) for coders one and two at the utterance level
         (metric by metric).
       - Reliability could also be computed at speaker and dialog levels, but
         interpretation must be cautious: finer granularity (utterances) is
         standard, while aggregations collapse variability and may inflate or
         obscure coder agreement.
    4. **Output**: For each input file, creates an Excel workbook with sheets:
       "Turns", "Speakers", "Dialogs", and "Reliability".

    Parameters
    ----------
    input_dir : str or Path
        Directory to search for POWERS coding files.
    output_dir : str or Path
        Root directory for saving analysis outputs.

    Returns
    -------
    None
        Results are written to Excel files in
        `{output_dir}/POWERS_CodingAnalysis`.
    """
    out_folder = "POWERS_CodingAnalysis" if not reliability else "POWERS_ReliabilityAnalysis"
    PCanalysis_dir = os.path.join(output_dir, out_folder)
    try:
        os.makedirs(PCanalysis_dir, exist_ok=True)
        logging.info(f"Output directory: {PCanalysis_dir}")
    except Exception as e:
        logging.error(f"Failed to create POWERS analysis directory {PCanalysis_dir}: {e}")
        return

    if not reliability:
        pc_files = list(Path(input_dir).rglob("*POWERS_Coding*.xlsx"))
        coders = ["c1", "c2"]
    else:
        pc_files = list(Path(PCanalysis_dir).rglob("*POWERS_ReliabilityCoding_Merged*.xlsx"))
        coders = ["c2", "c3"]
    c1, c2 = coders

    for pc_file in tqdm(pc_files, desc="Analyzing POWERS coding..."):
        try:
            utt_df = pd.read_excel(pc_file)
            logging.info(f"Processing POWERS coding file: {pc_file}")
        except Exception as e:
            logging.error(f"Failed to read POWERS coding file {pc_file}: {e}")
            continue

        # Add turn labels
        for coder in coders:
            utt_df.insert(
                utt_df.columns.to_list().index(f"{coder}_turn_type") + 1,
                f"{coder}_turn_label",
                number_turns(utt_df[f"{coder}_turn_type"]),
            )

        # Automated counts sum dict
        auto_summed = {
            f"{coder}_{col}_sum": (f"{coder}_{col}", "sum")
                for coder in coders
                for col in TURN_AGG_COLS
        }

        # Turn-level summary
        turn_df = utt_df.groupby(
            by=["sample_id", "speaker", f"{c1}_turn_label", f"{c2}_turn_label"]
        ).agg(**auto_summed).reset_index()

        # Speaker-level summary
        speaker_aggs = {
            **auto_summed,
            f"{c1}_total_turns": (f"{c1}_turn_label", "nunique"),
            f"{c2}_total_turns": (f"{c2}_turn_label", "nunique"),
            f"{c1}_num_T": (f"{c1}_turn_type", count_value("T")),
            f"{c2}_num_T": (f"{c2}_turn_type", count_value("T")),
            f"{c1}_num_MT": (f"{c1}_turn_type", count_value("MT")),
            f"{c2}_num_MT": (f"{c2}_turn_type", count_value("MT")),
            f"{c1}_num_ST": (f"{c1}_turn_type", count_value("ST")),
            f"{c2}_num_ST": (f"{c2}_turn_type", count_value("ST")),
        }

        speaker_df = utt_df.groupby(["sample_id", "speaker"]).agg(**speaker_aggs).reset_index()
        speaker_df[f"{c1}_mean_turn_length"] = (
            speaker_df[f"{c1}_speech_units_sum"] / speaker_df[f"{c1}_total_turns"]
        )
        speaker_df[f"{c2}_mean_turn_length"] = (
            speaker_df[f"{c2}_speech_units_sum"] / speaker_df[f"{c2}_total_turns"]
        )

        # Dialog-level summary
        sample_df = utt_df.groupby("sample_id").agg(**auto_summed).reset_index()

        # Reliability at utterance level
        icc_rows = []
        for col in TURN_AGG_COLS:
            try:
                tmp = utt_df[[f"{c1}_{col}", f"{c2}_{col}"]]
                if tmp.dropna().shape[0] < 2 or tmp.nunique().min() <= 1:
                    logging.warning(f"Not enough variability to compute ICC for {col} in {pc_file.name}")
                    continue
                tmp_long = tmp.melt(var_name="coder", value_name="score")
                tmp_long["target"] = list(np.arange(len(tmp))) * len(coders)
                res = intraclass_corr(data=tmp_long, targets="target", raters="coder", ratings="score", nan_policy='omit')
                icc_rows.append({"metric": col, "ICC2": res.query('Type == "ICC2"').iloc[0]["ICC"]})
            except Exception as e:
                logging.error(f"Failed to compute ICC2 reliability for {col} in {pc_file.name}: {e}. Skipping")
        icc_df = pd.DataFrame(icc_rows)

        # Turn type reliability
        y1 = utt_df[f"{c1}_turn_type"].fillna("MISSING").astype(str)
        y2 = utt_df[f"{c2}_turn_type"].fillna("MISSING").astype(str)

        kappa_turn = cohen_kappa_score(y1, y2)
        agree_turn = (y1 == y2).mean()

        # Collaborative repair reliability
        c1_bin = (~utt_df[f"{c1}_collab_repair"].isna()).astype(int)
        c2_bin = (~utt_df[f"{c2}_collab_repair"].isna()).astype(int)

        kappa_repair, agree_repair = np.nan, np.nan
        if len(np.unique(c1_bin)) > 1 or len(np.unique(c2_bin)) > 1:
            kappa_repair = cohen_kappa_score(c1_bin, c2_bin)
            agree_repair = (c1_bin == c2_bin).mean()

        cat_rel = pd.DataFrame([
            {"metric": "turn_type", "kappa": kappa_turn, "agreement": agree_turn},
            {"metric": "collab_repair", "kappa": kappa_repair, "agreement": agree_repair}
        ])

        # Write results
        out_file = os.path.join(
            PCanalysis_dir, Path(pc_file).stem.replace("Coding", "Analysis") + ".xlsx"
        )
        try:
            with pd.ExcelWriter(out_file, engine="xlsxwriter") as writer:
                turn_df.to_excel(writer, sheet_name="Turns", index=False)
                speaker_df.to_excel(writer, sheet_name="Speakers", index=False)
                sample_df.to_excel(writer, sheet_name="Dialogs", index=False)
                icc_df.to_excel(writer, sheet_name="ContinuousReliability", index=False)
                cat_rel.to_excel(writer, sheet_name="CategoricalReliability", index=False)
            logging.info(f"Wrote analysis workbook: {out_file}")
        except Exception as e:
            logging.error(f"Failed to write analysis file {out_file}: {e}")
