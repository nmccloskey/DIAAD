from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Optional

import pandas as pd

from psair.core.logger import logger, get_rel_path
from diaad.coding.utils import assign_coders, resolve_stim_cols, segment
from diaad.coding.utils.sampling import calc_subset_size
from diaad.metadata.discovery import find_one_matching_file
from diaad.metadata.blinding import blind_file_identifiers, write_blind_codebook
from diaad.metadata.utils import validate_columns


DEFAULT_NUM_BINS = 4
DEFAULT_STIMULUS_FIELD = "narrative"
TEMPLATE_SUBDIR = "coding_templates"
UTTERANCE_TEMPLATE_FILENAME = "utterance_coding_template.xlsx"
UTTERANCE_RELIABILITY_FILENAME = "utterance_reliability_template.xlsx"
UTTERANCE_CODEBOOK_FILENAME = "utterance_template_codebook.xlsx"
SAMPLE_TEMPLATE_FILENAME = "sample_coding_template.xlsx"
SAMPLE_RELIABILITY_FILENAME = "sample_reliability_template.xlsx"
SAMPLE_CODEBOOK_FILENAME = "sample_template_codebook.xlsx"


@dataclass(frozen=True)
class StimulusTemplateConfig:
    """
    Shared stimulus-related settings for coding templates.
    """
    stimulus_field: str = DEFAULT_STIMULUS_FIELD
    sample_id_field: str = "sample_id"
    utterance_id_field: str = "utterance_id"

    @property
    def use_stimulus(self) -> bool:
        return bool(str(self.stimulus_field or "").strip())


def normalize_coder_ids(
    coder_ids: Optional[list[str] | tuple[str, ...] | str],
) -> list[str]:
    """
    Normalize coder IDs to a non-empty list.
    """
    if coder_ids is None:
        return [""]

    if isinstance(coder_ids, str):
        coder_ids = [coder_ids]

    normalized = [str(x).strip() for x in coder_ids]
    normalized = [x for x in normalized if x != ""]

    return normalized or [""]


def require_columns(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
    """
    Validate required columns are present.
    """
    validate_columns(df, cols, df_name=df_name)


def coerce_bin_labels(num_bins: int) -> list[int]:
    """
    Return canonical bin labels as 1..num_bins.
    """
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}.")
    return list(range(1, num_bins + 1))


def prepare_stimulus_lookup(
    sample_df: pd.DataFrame,
    stimulus_field: str,
    sample_id_field: str = "sample_id",
) -> pd.DataFrame:
    """
    Return sample_id + stimulus lookup table.
    """
    require_columns(sample_df, [sample_id_field, stimulus_field], "sample_df")

    stim_df = sample_df.loc[:, [sample_id_field, stimulus_field]].copy()
    stim_df = stim_df.rename(columns={stimulus_field: "stimulus"})
    stim_df = stim_df.drop_duplicates(subset=[sample_id_field])

    return stim_df


def resolve_available_stimulus_field(
    sample_df: pd.DataFrame,
    stimulus_field: str,
) -> str | None:
    """
    Resolve the first stimulus-like column available in sample_df.
    """
    candidate_cols = resolve_stim_cols(stimulus_field)

    for col in candidate_cols:
        if col in sample_df.columns:
            return col

    if candidate_cols:
        logger.warning(
            "No requested stimulus columns were found in transcript tables: %s",
            candidate_cols,
        )

    return None


def expand_by_coder(
    df: pd.DataFrame,
    coder_ids: list[str],
    insert_after: str,
) -> pd.DataFrame:
    """
    Replicate rows once per coder and insert coder_id after insert_after.
    """
    frames = []

    for coder_id in coder_ids:
        part = df.copy()
        part["coder_id"] = coder_id
        frames.append(part)

    out = pd.concat(frames, ignore_index=True)

    cols = list(out.columns)
    cols.remove("coder_id")
    insert_idx = cols.index(insert_after) + 1
    cols = cols[:insert_idx] + ["coder_id"] + cols[insert_idx:]

    return out.loc[:, cols]


def apply_optional_identifier_blinding(
    df: pd.DataFrame,
    *,
    blind: bool,
    blinding_config,
    existing_codebook: pd.DataFrame | None = None,
    directories=None,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optionally blind identifiers for manual coding files.
    """
    if not blind:
        return df.copy(), pd.DataFrame()

    blinded_df, codebook_df = blind_file_identifiers(
        df=df,
        config=blinding_config,
        existing_codebook=existing_codebook,
        directories=directories,
        seed=seed,
    )
    return blinded_df, codebook_df


def find_transcript_table(
    input_dir: str | Path,
    output_dir: str | Path,
) -> Path | None:
    """
    Locate the transcript table workbook for template generation.
    """
    return find_one_matching_file(
        directories=[input_dir, output_dir],
        filename="transcript_tables.xlsx",
        label="transcript table file",
    )


def resolve_template_coder_ids(num_coders: int) -> list[str]:
    """
    Return normalized template coder IDs.
    """
    if num_coders <= 0:
        return [""]
    return [str(i) for i in range(1, num_coders + 1)]


def assign_template_coders(
    df: pd.DataFrame,
    *,
    coder_ids: list[str],
    sample_id_field: str = "sample_id",
) -> tuple[pd.DataFrame, list[list[str]], list[tuple[str, ...]]]:
    """
    Assign primary coders by sample while keeping sample rows together.
    """
    require_columns(df, [sample_id_field], "template_df")

    out = df.copy()
    sample_ids = list(out[sample_id_field].drop_duplicates())

    if not sample_ids:
        return out, [], []

    if len(coder_ids) <= 1:
        out["coder_id"] = coder_ids[0] if coder_ids else ""
        return out, [sample_ids], [(coder_ids[0] if coder_ids else "",)]

    segments = segment(sample_ids, n=len(coder_ids))
    assignments = assign_coders(coder_ids.copy())

    for seg, assn in zip(segments, assignments):
        if not seg:
            continue
        out.loc[out[sample_id_field].isin(seg), "coder_id"] = assn[0]

    return out, segments, assignments


def build_reliability_subset(
    df: pd.DataFrame,
    *,
    frac: float,
    coder_ids: list[str],
    segments: list[list[str]],
    assignments: list[tuple[str, ...]],
    sample_id_field: str = "sample_id",
) -> pd.DataFrame | None:
    """
    Build a sample-preserving reliability subset from a primary template.
    """
    if frac == 0:
        logger.info("frac=0 detected; no reliability subset will be generated.")
        return None

    require_columns(df, [sample_id_field], "template_df")

    sample_ids = list(df[sample_id_field].drop_duplicates())
    if not sample_ids:
        logger.warning("No sample IDs available for reliability subset generation.")
        return None

    if len(coder_ids) <= 1:
        n_rel = calc_subset_size(frac=frac, samples=sample_ids)
        rel_samples = random.sample(sample_ids, k=n_rel)
        rel_df = df[df[sample_id_field].isin(rel_samples)].copy()
        rel_df["coder_id"] = coder_ids[0] if coder_ids else ""
        return rel_df

    rel_subsets: list[pd.DataFrame] = []

    for seg, assn in zip(segments, assignments):
        if not seg:
            continue

        n_rel = calc_subset_size(frac=frac, samples=seg)
        rel_samples = random.sample(seg, k=n_rel)
        rel_df = df[df[sample_id_field].isin(rel_samples)].copy()
        rel_df["coder_id"] = assn[1]
        rel_subsets.append(rel_df)

    if not rel_subsets:
        return None

    return pd.concat(rel_subsets, ignore_index=True)


def write_coding_template(
    df: pd.DataFrame,
    path: str | Path,
    *,
    codebook_df: pd.DataFrame | None = None,
    codebook_path: str | Path | None = None,
) -> Path:
    """
    Write a coding template to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="coding_template")
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Template path must end in .xlsx or .csv")

    logger.info(f"Wrote coding template: {get_rel_path(path)}")

    if codebook_df is not None and not codebook_df.empty:
        if codebook_path is None:
            stem = path.with_suffix("")
            codebook_path = f"{stem}_codebook.xlsx"
        write_blind_codebook(codebook_df, codebook_path)

    return path


def write_template_exports(
    *,
    primary_df: pd.DataFrame,
    primary_path: str | Path,
    reliability_df: pd.DataFrame | None = None,
    reliability_path: str | Path | None = None,
    codebook_df: pd.DataFrame | None = None,
    codebook_path: str | Path | None = None,
) -> Path:
    """
    Write primary template plus optional reliability subset and codebook.
    """
    written_path = write_coding_template(
        primary_df,
        primary_path,
        codebook_df=codebook_df,
        codebook_path=codebook_path,
    )

    if reliability_df is not None and reliability_path is not None:
        write_coding_template(reliability_df, reliability_path)

    return written_path
