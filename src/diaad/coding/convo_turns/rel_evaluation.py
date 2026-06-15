from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from psair.core.logger import get_rel_path, logger
from diaad.metadata.discovery import find_one_matching_file
from diaad.coding.convo_turns.analysis import extract_turn_counts
from diaad.coding.utils.rel_eval_utils import (
    calculate_icc_from_pingouin,
    coverage_summary,
    variance_pair_stats,
    write_coverage_section,
)
from diaad.transcripts.transcription_reliability_evaluation import (
    _format_alignment_output,
    _levenshtein_metrics,
    _needleman_wunsch_global,
)


TURN_KEY_COLS = ["sample_id", "session", "bin"]


def _turn_key_cols(sample_id_field: str = "sample_id") -> list[str]:
    return [sample_id_field, "session", "bin"]


def _normalize_turn_file(
    df: pd.DataFrame,
    *,
    label: str,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Normalize one conversation-turn dataframe to a consistent evaluation schema.
    """
    out = df.copy()
    if sample_id_field not in out.columns and "group" in out.columns:
        out = out.rename(columns={"group": sample_id_field})
    if sample_id_field not in out.columns and sample_id_field != "sample_id" and "sample_id" in out.columns:
        out = out.rename(columns={"sample_id": sample_id_field})

    required = [sample_id_field, "turns"]
    missing = [col for col in required if col not in out.columns]
    if missing:
        raise KeyError(f"{label} is missing required columns: {missing}")

    turn_key_cols = _turn_key_cols(sample_id_field)
    for col in turn_key_cols:
        if col not in out.columns:
            out[col] = ""

    out[sample_id_field] = out[sample_id_field].fillna("").astype(str).str.strip()
    out["session"] = out["session"].fillna("").astype(str).str.strip()
    out["bin"] = out["bin"].fillna("").astype(str).str.strip()
    out["turns"] = out["turns"].fillna("").astype(str).str.strip()

    dupes = out.duplicated(subset=turn_key_cols, keep=False)
    if dupes.any():
        logger.warning(
            "%s contains duplicate sample/session/bin rows; keeping the first occurrence for each key.",
            label,
        )
        out = out.drop_duplicates(subset=turn_key_cols, keep="first")

    return out.loc[:, turn_key_cols + ["turns"]]


def _count_percent_agreement(count_main: int, count_rel: int) -> float:
    """Return an intuitive count-level percent agreement in [0, 100]."""
    top = max(int(count_main), int(count_rel))
    if top == 0:
        return 100.0
    return round((min(int(count_main), int(count_rel)) / top) * 100, 3)


def _build_counts_sheet(
    merged: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """Expand one merged row per sample/session/bin into participant count rows."""
    rows: list[dict] = []

    for _, row in merged.iterrows():
        main_counts = extract_turn_counts(row.get("turns_main", ""))
        rel_counts = extract_turn_counts(row.get("turns_rel", ""))
        participants = sorted(set(main_counts) | set(rel_counts))

        for participant in participants:
            count_main = int(main_counts.get(participant, 0))
            count_rel = int(rel_counts.get(participant, 0))
            rows.append(
                {
                    sample_id_field: row[sample_id_field],
                    "session": row["session"],
                    "bin": row["bin"],
                    "participant": participant,
                    "count_main": count_main,
                    "count_rel": count_rel,
                    "perc_agmt": _count_percent_agreement(count_main, count_rel),
                }
            )

    counts_df = pd.DataFrame(
        rows,
        columns=[
            sample_id_field,
            "session",
            "bin",
            "participant",
            "count_main",
            "count_rel",
            "perc_agmt",
        ],
    )
    if counts_df.empty:
        return counts_df

    counts_df["target_id"] = counts_df.apply(
        lambda r: f"{r[sample_id_field]}|{r['session']}|{r['bin']}|{r['participant']}",
        axis=1,
    )
    return counts_df


def _safe_alignment_token(value: object, fallback: str) -> str:
    text = str(value).strip()
    if text == "":
        return fallback
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return cleaned or fallback


def _save_turn_alignment(
    row: pd.Series,
    out_dir: Path,
    sample_id_field: str = "sample_id",
) -> None:
    """Write one global alignment text file for a turns sequence pair."""
    filename = (
        f"{_safe_alignment_token(row[sample_id_field], 'sample')}_"
        f"{_safe_alignment_token(row['session'], 'session')}_"
        f"{_safe_alignment_token(row['bin'], 'bin')}_turns_alignment.txt"
    )
    path = out_dir / "global_alignments" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    turns_main = row["turns_main"]
    turns_rel = row["turns_rel"]

    if turns_main == "" or turns_rel == "":
        path.write_text(
            "\n".join(
                [
                    "Global alignment unavailable because one sequence is blank.",
                    "",
                    f"Sequence 1: {turns_main}",
                    f"Sequence 2: {turns_rel}",
                ]
            ),
            encoding="utf-8",
        )
        return

    nw = _needleman_wunsch_global(turns_main, turns_rel)
    path.write_text(
        _format_alignment_output(
            nw["alignment"],
            nw["needleman_wunsch_score"],
            nw["needleman_wunsch_norm"],
        ),
        encoding="utf-8",
    )


def _build_sequences_sheet(
    merged: pd.DataFrame,
    out_dir: Path,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """Compute Levenshtein sequence metrics and save optional alignments."""
    rows: list[dict] = []

    for _, row in merged.iterrows():
        metrics = _levenshtein_metrics(row.get("turns_main", ""), row.get("turns_rel", ""))
        rows.append(
            {
                sample_id_field: row[sample_id_field],
                "session": row["session"],
                "bin": row["bin"],
                "levenshtein_distance": metrics["levenshtein_distance"],
                "levenshtein_similarity": metrics["levenshtein_similarity"],
            }
        )
        try:
            _save_turn_alignment(
                row,
                out_dir,
                sample_id_field=sample_id_field,
            )
        except Exception as e:
            logger.warning(
                "Failed to write turns alignment for %s / %s / %s: %s",
                row[sample_id_field],
                row["session"],
                row["bin"],
                e,
            )

    return pd.DataFrame(
        rows,
        columns=[
            sample_id_field,
            "session",
            "bin",
            "levenshtein_distance",
            "levenshtein_similarity",
        ],
    )


def _build_sample_sheet(
    counts_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """Aggregate count and sequence descriptive metrics to the sample level."""
    count_summary = pd.DataFrame(columns=[sample_id_field, "avg_perc_agmt"])
    if not counts_df.empty:
        count_summary = (
            counts_df.groupby(sample_id_field, as_index=False)["perc_agmt"]
            .mean()
            .rename(columns={"perc_agmt": "avg_perc_agmt"})
        )

    seq_summary = pd.DataFrame(columns=[sample_id_field, "avg_dist", "avg_sim"])
    if not sequences_df.empty:
        seq_summary = (
            sequences_df.groupby(sample_id_field, as_index=False)
            .agg(
                avg_dist=("levenshtein_distance", "mean"),
                avg_sim=("levenshtein_similarity", "mean"),
            )
        )

    if count_summary.empty:
        return seq_summary if not seq_summary.empty else pd.DataFrame(
            columns=[sample_id_field, "avg_perc_agmt", "avg_dist", "avg_sim"]
        )
    if seq_summary.empty:
        return count_summary
    return count_summary.merge(seq_summary, on=sample_id_field, how="outer")


def _write_turns_reliability_report(
    *,
    counts_df: pd.DataFrame,
    sequences_df: pd.DataFrame,
    report_path: Path,
    rel_name: str,
    coverage: dict | None = None,
) -> None:
    """Write a plain-text reliability report for digital conversation turns."""
    icc_value = np.nan
    paired_targets = 0
    mean_perc_agmt = np.nan

    if not counts_df.empty:
        paired_targets = len(counts_df)
        mean_perc_agmt = round(float(counts_df["perc_agmt"].mean()), 3)
        icc_value = calculate_icc_from_pingouin(
            df=counts_df,
            target_col="target_id",
            col_org="count_main",
            col_rel="count_rel",
            rater_labels=("main", "rel"),
        )
    variance = variance_pair_stats(counts_df, "count_main", "count_rel")

    sims = (
        sequences_df["levenshtein_similarity"].astype(float).dropna()
        if not sequences_df.empty and "levenshtein_similarity" in sequences_df.columns
        else pd.Series(dtype=float)
    )
    dists = (
        sequences_df["levenshtein_distance"].astype(float).dropna()
        if not sequences_df.empty and "levenshtein_distance" in sequences_df.columns
        else pd.Series(dtype=float)
    )

    n_sequences = len(sims)
    similarity_lines = []
    if n_sequences > 0:
        bands = {
            "Excellent (>= .90)": int((sims >= 0.90).sum()),
            "Sufficient (.80 - .89)": int(((sims >= 0.80) & (sims < 0.90)).sum()),
            "Min. acceptable (.70 - .79)": int(((sims >= 0.70) & (sims < 0.80)).sum()),
            "Below .70": int((sims < 0.70).sum()),
        }
        similarity_lines = [
            f"Paired session-bin rows: {n_sequences}",
            f"Average Levenshtein similarity: {sims.mean():.3f}",
            f"Similarity standard deviation: {sims.std():.3f}",
            f"Minimum similarity: {sims.min():.3f}",
            f"Maximum similarity: {sims.max():.3f}",
            f"Average Levenshtein distance: {dists.mean():.3f}",
            "",
            "Similarity bands:",
        ]
        for label, count in bands.items():
            pct = count / n_sequences * 100 if n_sequences else 0.0
            similarity_lines.append(f"  {label}: {count} ({pct:.1f}%)")
    else:
        similarity_lines = ["No paired sequence rows available."]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("Digital Conversation Turns Reliability Report\n\n")
        f.write(f"Source reliability file: {rel_name}\n\n")
        write_coverage_section(f, coverage)

        f.write("Primary reliability metrics\n")
        f.write("---------------------------\n\n")
        f.write("Count totals by participant within sample/session/bin\n")
        f.write(f"Paired participant targets: {paired_targets}\n")
        f.write(f"Mean percent agreement: {mean_perc_agmt}\n")
        f.write(
            f"ICC(2,1): {icc_value} "
            f"(main_var={variance['org_var']}, rel_var={variance['rel_var']}, "
            f"pooled_var={variance['pooled_var']})\n\n"
        )

        f.write("Sequence consistency\n")
        f.write("--------------------\n")
        f.write("\n".join(similarity_lines))
        f.write("\n")


def evaluate_digital_convo_turns_reliability(
    metadata_fields,
    input_dir,
    output_dir,
    sample_id_field: str = "sample_id",
    dct_coding_filename: str = "conversation_turns.xlsx",
    dct_coding_reliability: str = "conversation_turns_reliability.xlsx",
):
    """
    Evaluate digital conversation turn reliability using counts and sequence similarity.
    """
    del metadata_fields

    out_dir = Path(output_dir) / "turns_reliability"
    out_dir.mkdir(parents=True, exist_ok=True)

    org_file = find_one_matching_file(
        directories=[input_dir, output_dir],
        filename=dct_coding_filename,
        label="conversation turns coding file",
    )
    rel_file = find_one_matching_file(
        directories=[input_dir, output_dir],
        filename=dct_coding_reliability,
        label="conversation turns reliability file",
    )

    try:
        org_df = _normalize_turn_file(
            pd.read_excel(org_file),
            label=org_file.name,
            sample_id_field=sample_id_field,
        )
        rel_df = _normalize_turn_file(
            pd.read_excel(rel_file),
            label=rel_file.name,
            sample_id_field=sample_id_field,
        )
        logger.info(f"Processing pair: {get_rel_path(org_file)} + {get_rel_path(rel_file)}")
    except Exception as e:
        logger.error(f"Failed reading or normalizing {get_rel_path(org_file)} or {get_rel_path(rel_file)}: {e}")
        return

    turn_key_cols = _turn_key_cols(sample_id_field)
    merged = pd.merge(
        org_df,
        rel_df,
        on=turn_key_cols,
        how="outer",
        suffixes=("_main", "_rel"),
    )
    merged["turns_main"] = merged["turns_main"].fillna("").astype(str)
    merged["turns_rel"] = merged["turns_rel"].fillna("").astype(str)

    if merged.empty:
        logger.warning("Merged turns reliability dataframe is empty.")
        return

    counts_df = _build_counts_sheet(merged, sample_id_field=sample_id_field)
    sequences_df = _build_sequences_sheet(
        merged,
        out_dir,
        sample_id_field=sample_id_field,
    )
    samples_df = _build_sample_sheet(
        counts_df,
        sequences_df,
        sample_id_field=sample_id_field,
    )

    results_path = out_dir / "conversation_turns_reliability_results.xlsx"
    try:
        with pd.ExcelWriter(results_path, engine="xlsxwriter") as writer:
            counts_df.drop(columns=["target_id"], errors="ignore").to_excel(
                writer,
                sheet_name="counts",
                index=False,
            )
            sequences_df.to_excel(writer, sheet_name="sequences", index=False)
            samples_df.to_excel(writer, sheet_name="samples", index=False)
        logger.info(f"Wrote turns reliability results: {get_rel_path(results_path)}")
    except Exception as e:
        logger.error(f"Failed writing turns reliability results {get_rel_path(results_path)}: {e}")
        return

    report_path = out_dir / "conversation_turns_reliability_report.txt"
    try:
        _write_turns_reliability_report(
            counts_df=counts_df,
            sequences_df=sequences_df,
            report_path=report_path,
            rel_name=rel_file.name,
            coverage=coverage_summary(
                org_df,
                merged,
                sample_id_field=sample_id_field,
                utterance_id_field=None,
                unit_label="session-bin rows",
                unit_key_cols=turn_key_cols,
            ),
        )
        logger.info(f"Successfully wrote turns reliability report to {get_rel_path(report_path)}")
    except Exception as e:
        logger.error(f"Failed writing turns reliability report {get_rel_path(report_path)}: {e}")
