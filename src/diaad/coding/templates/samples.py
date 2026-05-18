from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from psair.core.logger import logger
from diaad.transcripts.transcript_tables import extract_transcript_data
from diaad.coding.templates.utils import (
    DEFAULT_NUM_BINS,
    DEFAULT_STIMULUS_FIELD,
    SAMPLE_CODEBOOK_FILENAME,
    SAMPLE_RELIABILITY_FILENAME,
    SAMPLE_TEMPLATE_FILENAME,
    TEMPLATE_SUBDIR,
    StimulusTemplateConfig,
    apply_optional_identifier_blinding,
    assign_template_coders,
    build_reliability_subset,
    coerce_bin_labels,
    expand_by_coder,
    find_transcript_table,
    normalize_coder_ids,
    require_columns,
    resolve_available_stimulus_field,
    resolve_template_coder_ids,
    write_coding_template,
    write_template_exports,
)


@dataclass(frozen=True)
class SampleTemplateConfig(StimulusTemplateConfig):
    """
    Settings for sample-level coding templates.
    """
    num_bins: int = DEFAULT_NUM_BINS


def _prepare_sample_template(
    sample_df: pd.DataFrame,
    config: SampleTemplateConfig,
) -> pd.DataFrame:
    """
    Build the base sample-level coding template.
    """
    sample_id_field = config.sample_id_field
    require_columns(sample_df, [sample_id_field], "sample_df")

    cols = [sample_id_field]
    stimulus_field = resolve_available_stimulus_field(
        sample_df,
        config.stimulus_field,
    )
    if stimulus_field:
        cols.append(stimulus_field)

    out = sample_df.loc[:, cols].copy().drop_duplicates(subset=[sample_id_field])

    if stimulus_field:
        out = out.rename(columns={stimulus_field: "stimulus"})
        out["bin"] = pd.NA
        out = out.loc[:, [sample_id_field, "stimulus", "bin"]]
    else:
        out["bin"] = pd.NA
        out = out.loc[:, [sample_id_field, "bin"]]

    return out


def _expand_sample_template_bins(
    df: pd.DataFrame,
    *,
    num_bins: int,
    bin_col: str = "bin",
) -> pd.DataFrame:
    """
    Expand each sample row into one row per configured bin label.
    """
    if bin_col not in df.columns:
        raise ValueError(f"'{bin_col}' not found in dataframe.")

    bin_labels = coerce_bin_labels(num_bins)
    frames: list[pd.DataFrame] = []

    for label in bin_labels:
        part = df.copy()
        part[bin_col] = label
        frames.append(part)

    if not frames:
        return df.iloc[0:0].copy()

    return pd.concat(frames, ignore_index=True)


def _sort_sample_template(
    df: pd.DataFrame,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Sort sample templates for readable output.
    """
    sort_cols = [col for col in [sample_id_field, "bin"] if col in df.columns]
    if not sort_cols:
        return df.copy()
    return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def build_sample_coding_template(
    transcript_table_path: str | Path,
    *,
    coder_ids: Optional[list[str] | tuple[str, ...] | str] = None,
    template_config: SampleTemplateConfig | None = None,
    blind: bool = False,
    blinding_config=None,
    existing_codebook: pd.DataFrame | None = None,
    seed: int = 99,
    sample_id_field: str = "sample_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a sample-level blank coding template from transcript tables.
    """
    template_config = template_config or SampleTemplateConfig(
        sample_id_field=sample_id_field,
    )
    coder_ids = normalize_coder_ids(coder_ids)

    sample_df = extract_transcript_data(transcript_table_path, kind="sample")
    out = _prepare_sample_template(sample_df=sample_df, config=template_config)
    out = expand_by_coder(out, coder_ids, insert_after=template_config.sample_id_field)
    out = _sort_sample_template(out, sample_id_field=template_config.sample_id_field)

    if blind:
        if blinding_config is None:
            raise ValueError("blinding_config is required when blind=True.")
        out, codebook_df = apply_optional_identifier_blinding(
            out,
            blind=True,
            blinding_config=blinding_config,
            existing_codebook=existing_codebook,
            directories=[Path(transcript_table_path).parent],
            seed=seed,
        )
    else:
        codebook_df = pd.DataFrame()

    logger.info("Prepared sample-level coding template with %s row(s).", len(out))
    return out, codebook_df


def add_balanced_bins(
    df: pd.DataFrame,
    *,
    num_bins: int = DEFAULT_NUM_BINS,
    sample_id_col: str = "sample_id",
    bin_col: str = "bin",
) -> pd.DataFrame:
    """
    Fill bin assignments across unique samples as evenly as possible.
    """
    if sample_id_col not in df.columns:
        raise ValueError(f"'{sample_id_col}' not found in dataframe.")

    bin_labels = coerce_bin_labels(num_bins)

    out = df.copy()
    sample_order = (
        out.loc[:, [sample_id_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    sample_order[bin_col] = [
        bin_labels[i % num_bins] for i in range(len(sample_order))
    ]

    out = out.drop(columns=[bin_col], errors="ignore")
    out = out.merge(sample_order, on=sample_id_col, how="left", validate="m:1")

    return _sort_sample_template(out)


def make_sample_coding_template(
    transcript_table_path: str | Path,
    output_path: str | Path,
    *,
    coder_ids: Optional[list[str] | tuple[str, ...] | str] = None,
    num_bins: int = DEFAULT_NUM_BINS,
    stimulus_field: str = DEFAULT_STIMULUS_FIELD,
    prefill_bins: bool = False,
    blind: bool = False,
    blinding_config=None,
    existing_codebook: pd.DataFrame | None = None,
    codebook_path: str | Path | None = None,
    seed: int = 99,
    sample_id_field: str = "sample_id",
) -> Path:
    """
    Convenience wrapper: build and write a sample-level coding template.
    """
    config = SampleTemplateConfig(
        num_bins=num_bins,
        stimulus_field=stimulus_field,
        sample_id_field=sample_id_field,
    )
    df, codebook_df = build_sample_coding_template(
        transcript_table_path,
        coder_ids=coder_ids,
        template_config=config,
        blind=blind,
        blinding_config=blinding_config,
        existing_codebook=existing_codebook,
        seed=seed,
        sample_id_field=sample_id_field,
    )

    if prefill_bins:
        df = add_balanced_bins(
            df,
            num_bins=config.num_bins,
            sample_id_col=config.sample_id_field,
        )

    return write_coding_template(
        df,
        output_path,
        codebook_df=codebook_df,
        codebook_path=codebook_path,
    )


def make_sample_template_files(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    frac: float,
    num_bins: int = DEFAULT_NUM_BINS,
    num_coders: int,
    stimulus_field: str = DEFAULT_STIMULUS_FIELD,
    blinding_config=None,
    seed: int = 99,
    sample_id_field: str = "sample_id",
) -> Path | None:
    """
    Create sample-level primary and reliability template workbooks.
    """
    transcript_table = find_transcript_table(input_dir, output_dir)
    if transcript_table is None:
        return None

    template_dir = Path(output_dir) / TEMPLATE_SUBDIR
    template_dir.mkdir(parents=True, exist_ok=True)

    config = SampleTemplateConfig(
        num_bins=num_bins,
        stimulus_field=stimulus_field,
        sample_id_field=sample_id_field,
    )

    df, _ = build_sample_coding_template(
        transcript_table_path=transcript_table,
        coder_ids=None,
        template_config=config,
        blind=False,
        seed=seed,
        sample_id_field=sample_id_field,
    )

    df = _expand_sample_template_bins(df, num_bins=config.num_bins)
    df = _sort_sample_template(df, sample_id_field=config.sample_id_field)

    coder_ids = resolve_template_coder_ids(num_coders)
    df, segments, assignments = assign_template_coders(
        df,
        coder_ids=coder_ids,
        sample_id_field=config.sample_id_field,
    )
    df = _sort_sample_template(df, sample_id_field=config.sample_id_field)

    rel_df = build_reliability_subset(
        df,
        frac=frac,
        coder_ids=coder_ids,
        segments=segments,
        assignments=assignments,
        sample_id_field=config.sample_id_field,
    )
    if rel_df is not None:
        rel_df = _sort_sample_template(rel_df, sample_id_field=config.sample_id_field)

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
        export_df = _sort_sample_template(export_df, sample_id_field=config.sample_id_field)

        if rel_df is not None:
            export_rel_df, _ = apply_optional_identifier_blinding(
                rel_df,
                blind=True,
                blinding_config=blinding_config,
                existing_codebook=codebook_df,
                directories=[input_dir, output_dir],
                seed=seed,
            )
            export_rel_df = _sort_sample_template(
                export_rel_df,
                sample_id_field=config.sample_id_field,
            )

    logger.info(
        "Prepared sample template exports: primary=%s rows, reliability=%s rows.",
        len(export_df),
        0 if export_rel_df is None else len(export_rel_df),
    )

    return write_template_exports(
        primary_df=export_df,
        primary_path=template_dir / SAMPLE_TEMPLATE_FILENAME,
        reliability_df=export_rel_df,
        reliability_path=template_dir / SAMPLE_RELIABILITY_FILENAME,
        codebook_df=codebook_df,
        codebook_path=template_dir / SAMPLE_CODEBOOK_FILENAME,
    )
