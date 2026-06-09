from __future__ import annotations

from pathlib import Path

import pandas as pd

from psair.core.logger import logger
from diaad.coding.templates.samples import _sort_sample_template
from diaad.coding.templates.utils import (
    TEMPLATE_SUBDIR,
    apply_optional_identifier_blinding,
    assign_template_coders,
    build_reliability_subset,
    coerce_bin_labels,
    expand_by_coder,
    find_transcript_table,
    resolve_template_coder_ids,
    write_template_exports,
)
from diaad.transcripts.transcript_tables import extract_transcript_data


TURNS_TEMPLATE_FILENAME = "conversation_turns_template.xlsx"
TURNS_RELIABILITY_FILENAME = "conversation_turns_reliability_template.xlsx"
TURNS_CODEBOOK_FILENAME = "conversation_turns_template_codebook.xlsx"


def _build_turns_template_base(
    transcript_table_path: str | Path,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """Build the blank turns template before coder and reliability expansion."""
    sample_df = extract_transcript_data(transcript_table_path, kind="sample")
    if sample_id_field not in sample_df.columns:
        raise ValueError(
            f"Transcript sample table must include '{sample_id_field}'."
        )

    out = (
        sample_df.loc[:, [sample_id_field]]
        .copy()
        .drop_duplicates(subset=[sample_id_field])
    )
    out["session"] = pd.NA
    out["bin"] = pd.NA
    out["turns"] = pd.NA
    return out.loc[:, [sample_id_field, "session", "bin", "turns"]]


def _expand_turn_bins(df: pd.DataFrame, *, num_bins: int) -> pd.DataFrame:
    """Repeat each sample row once per bin label."""
    frames: list[pd.DataFrame] = []
    for label in coerce_bin_labels(num_bins):
        part = df.copy()
        part["bin"] = label
        frames.append(part)

    if not frames:
        return df.iloc[0:0].copy()

    return pd.concat(frames, ignore_index=True)


def _sort_turns_template(
    df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """Sort turns templates for stable, readable output."""
    ordered = _sort_sample_template(df, sample_id_field=sample_id_field)
    cols = [
        col
        for col in [sample_id_field, "coder_id", "session", "bin", "turns"]
        if col in ordered.columns
    ]
    return ordered.loc[:, cols]


def make_digital_convo_turn_files(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    frac: float,
    num_bins: int,
    num_coders: int,
    blinding_config=None,
    seed: int = 99,
    sample_id_field: str = "sample_id",
    transcript_table_filename: str = "transcript_tables.xlsx",
) -> Path | None:
    """
    Create primary and reliability coding templates for digital conversation turns.
    """
    transcript_table = find_transcript_table(
        input_dir,
        output_dir,
        transcript_table_filename=transcript_table_filename,
    )
    if transcript_table is None:
        return None

    template_dir = Path(output_dir) / TEMPLATE_SUBDIR
    template_dir.mkdir(parents=True, exist_ok=True)

    df = _build_turns_template_base(
        transcript_table,
        sample_id_field=sample_id_field,
    )
    df = expand_by_coder(df, [""], insert_after=sample_id_field)
    df = _expand_turn_bins(df, num_bins=num_bins)

    coder_ids = resolve_template_coder_ids(num_coders)
    df, segments, assignments = assign_template_coders(
        df,
        coder_ids=coder_ids,
        sample_id_field=sample_id_field,
    )
    df = _sort_turns_template(df, sample_id_field=sample_id_field)

    rel_df = build_reliability_subset(
        df,
        frac=frac,
        coder_ids=coder_ids,
        segments=segments,
        assignments=assignments,
        sample_id_field=sample_id_field,
    )
    if rel_df is not None:
        rel_df = _sort_turns_template(rel_df, sample_id_field=sample_id_field)

    codebook_df = pd.DataFrame()
    export_df = df
    export_rel_df = rel_df

    if blinding_config is not None and blinding_config.should_blind("coding"):
        export_df, codebook_df = apply_optional_identifier_blinding(
            df,
            blind=True,
            blinding_config=blinding_config,
            directories=[input_dir, output_dir],
            seed=seed,
        )
        export_df = _sort_turns_template(export_df, sample_id_field=sample_id_field)

        if rel_df is not None:
            export_rel_df, _ = apply_optional_identifier_blinding(
                rel_df,
                blind=True,
                blinding_config=blinding_config,
                existing_codebook=codebook_df,
                directories=[input_dir, output_dir],
                seed=seed,
            )
            export_rel_df = _sort_turns_template(
                export_rel_df,
                sample_id_field=sample_id_field,
            )

    logger.info(
        "Prepared digital conversation turn template exports: primary=%s rows, reliability=%s rows.",
        len(export_df),
        0 if export_rel_df is None else len(export_rel_df),
    )

    return write_template_exports(
        primary_df=export_df,
        primary_path=template_dir / TURNS_TEMPLATE_FILENAME,
        reliability_df=export_rel_df,
        reliability_path=template_dir / TURNS_RELIABILITY_FILENAME,
        codebook_df=codebook_df,
        codebook_path=template_dir / TURNS_CODEBOOK_FILENAME,
    )
