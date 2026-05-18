from __future__ import annotations

from pathlib import Path

import pandas as pd

from psair.core.logger import logger
from diaad.coding.templates.utils import (
    TEMPLATE_SUBDIR,
    find_transcript_table,
    require_columns,
    write_coding_template,
)
from diaad.transcripts.transcript_tables import extract_transcript_data


TIME_TEMPLATE_FILENAME = "speaking_times.xlsx"


def build_speaking_time_template(
    transcript_table_path: str | Path,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Build a blank speaking-time template from transcript tables.
    """
    sample_df = extract_transcript_data(transcript_table_path, kind="sample")
    require_columns(sample_df, [sample_id_field], "sample_df")

    out = (
        sample_df.loc[:, [sample_id_field]]
        .copy()
        .drop_duplicates(subset=[sample_id_field])
    )
    out["speaking_time"] = pd.NA
    out = out.sort_values([sample_id_field], kind="stable").reset_index(drop=True)

    logger.info("Prepared speaking-time template with %s row(s).", len(out))
    return out


def make_speaking_time_template(
    transcript_table_path: str | Path,
    output_path: str | Path,
    sample_id_field: str = "sample_id",
) -> Path:
    """
    Convenience wrapper: build and write a speaking-time template.
    """
    df = build_speaking_time_template(
        transcript_table_path,
        sample_id_field=sample_id_field,
    )
    return write_coding_template(df, output_path)


def make_speaking_time_template_files(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    sample_id_field: str = "sample_id",
) -> Path | None:
    """
    Create a speaking-time template workbook keyed by sample_id.
    """
    transcript_table = find_transcript_table(input_dir, output_dir)
    if transcript_table is None:
        return None

    template_dir = Path(output_dir) / TEMPLATE_SUBDIR
    template_dir.mkdir(parents=True, exist_ok=True)

    return make_speaking_time_template(
        transcript_table_path=transcript_table,
        output_path=template_dir / TIME_TEMPLATE_FILENAME,
        sample_id_field=sample_id_field,
    )
