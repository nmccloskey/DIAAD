from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from psair.core.logger import logger
from diaad.transcripts.transcript_tables import extract_transcript_data
from diaad.coding.templates.utils import (
    DEFAULT_STIMULUS_FIELD,
    TEMPLATE_SUBDIR,
    UTTERANCE_CODEBOOK_FILENAME,
    UTTERANCE_RELIABILITY_FILENAME,
    UTTERANCE_TEMPLATE_FILENAME,
    StimulusTemplateConfig,
    apply_optional_identifier_blinding,
    assign_template_coders,
    build_reliability_subset,
    expand_by_coder,
    find_transcript_table,
    normalize_coder_ids,
    prepare_stimulus_lookup,
    require_columns,
    resolve_available_stimulus_field,
    resolve_template_coder_ids,
    write_coding_template,
    write_template_exports,
)


@dataclass(frozen=True)
class UtteranceTemplateConfig(StimulusTemplateConfig):
    """
    Settings for utterance-level coding templates.
    """


def _prepare_utterance_template(
    utt_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    config: UtteranceTemplateConfig,
) -> pd.DataFrame:
    """
    Build the base utterance-level coding template.
    """
    require_columns(utt_df, ["sample_id", "utterance_id", "utterance"], "utt_df")

    out = utt_df.loc[:, ["sample_id", "utterance_id", "utterance"]].copy()

    stimulus_field = resolve_available_stimulus_field(
        sample_df,
        config.stimulus_field,
    )

    if stimulus_field:
        stim_df = prepare_stimulus_lookup(sample_df, stimulus_field)
        out = out.merge(stim_df, on="sample_id", how="left", validate="m:1")
        out = out.loc[:, ["sample_id", "utterance_id", "stimulus", "utterance"]]

    return out


def build_utterance_coding_template(
    transcript_table_path: str | Path,
    *,
    coder_ids: Optional[list[str] | tuple[str, ...] | str] = None,
    template_config: UtteranceTemplateConfig | None = None,
    blind: bool = False,
    blinding_config=None,
    existing_codebook: pd.DataFrame | None = None,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build an utterance-level blank coding template from transcript tables.
    """
    template_config = template_config or UtteranceTemplateConfig()
    coder_ids = normalize_coder_ids(coder_ids)

    sample_df = extract_transcript_data(transcript_table_path, kind="sample")
    utt_df = extract_transcript_data(transcript_table_path, kind="utterance")

    out = _prepare_utterance_template(
        utt_df=utt_df,
        sample_df=sample_df,
        config=template_config,
    )
    out = expand_by_coder(out, coder_ids, insert_after="utterance_id")

    if blind:
        if blinding_config is None:
            raise ValueError("blinding_config is required when blind=True.")
        out, codebook_df = apply_optional_identifier_blinding(
            out,
            blind=True,
            blinding_config=blinding_config,
            existing_codebook=existing_codebook,
            seed=seed,
        )
    else:
        codebook_df = pd.DataFrame()

    logger.info("Prepared utterance-level coding template with %s row(s).", len(out))
    return out, codebook_df


def make_utterance_coding_template(
    transcript_table_path: str | Path,
    output_path: str | Path,
    *,
    coder_ids: Optional[list[str] | tuple[str, ...] | str] = None,
    stimulus_field: str = DEFAULT_STIMULUS_FIELD,
    blind: bool = False,
    blinding_config=None,
    existing_codebook: pd.DataFrame | None = None,
    codebook_path: str | Path | None = None,
    seed: int = 99,
) -> Path:
    """
    Convenience wrapper: build and write an utterance-level coding template.
    """
    config = UtteranceTemplateConfig(stimulus_field=stimulus_field)
    df, codebook_df = build_utterance_coding_template(
        transcript_table_path,
        coder_ids=coder_ids,
        template_config=config,
        blind=blind,
        blinding_config=blinding_config,
        existing_codebook=existing_codebook,
        seed=seed,
    )
    return write_coding_template(
        df,
        output_path,
        codebook_df=codebook_df,
        codebook_path=codebook_path,
    )


def make_utterance_template_files(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    frac: float,
    num_coders: int,
    stimulus_field: str = DEFAULT_STIMULUS_FIELD,
    blinding_config=None,
    seed: int = 99,
) -> Path | None:
    """
    Create utterance-level primary and reliability template workbooks.
    """
    transcript_table = find_transcript_table(input_dir, output_dir)
    if transcript_table is None:
        return None

    template_dir = Path(output_dir) / TEMPLATE_SUBDIR
    template_dir.mkdir(parents=True, exist_ok=True)

    config = UtteranceTemplateConfig(stimulus_field=stimulus_field)
    df, _ = build_utterance_coding_template(
        transcript_table_path=transcript_table,
        coder_ids=None,
        template_config=config,
        blind=False,
        seed=seed,
    )

    coder_ids = resolve_template_coder_ids(num_coders)
    df, segments, assignments = assign_template_coders(df, coder_ids=coder_ids)
    rel_df = build_reliability_subset(
        df,
        frac=frac,
        coder_ids=coder_ids,
        segments=segments,
        assignments=assignments,
    )

    codebook_df = pd.DataFrame()
    export_df = df
    export_rel_df = rel_df

    if blinding_config is not None and blinding_config.should_blind("coding"):
        export_df, codebook_df = apply_optional_identifier_blinding(
            df,
            blind=True,
            blinding_config=blinding_config,
            seed=seed,
        )

        if rel_df is not None:
            export_rel_df, _ = apply_optional_identifier_blinding(
                rel_df,
                blind=True,
                blinding_config=blinding_config,
                existing_codebook=codebook_df,
                seed=seed,
            )

    logger.info(
        "Prepared utterance template exports: primary=%s rows, reliability=%s rows.",
        len(export_df),
        0 if export_rel_df is None else len(export_rel_df),
    )

    return write_template_exports(
        primary_df=export_df,
        primary_path=template_dir / UTTERANCE_TEMPLATE_FILENAME,
        reliability_df=export_rel_df,
        reliability_path=template_dir / UTTERANCE_RELIABILITY_FILENAME,
        codebook_df=codebook_df,
        codebook_path=template_dir / UTTERANCE_CODEBOOK_FILENAME,
    )
