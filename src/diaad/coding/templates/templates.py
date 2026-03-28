from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Optional

import pandas as pd

from diaad.core.logger import logger, _rel
from diaad.coding.utils import assign_coders, resolve_stim_cols, segment
from diaad.coding.utils.sampling import calc_subset_size
from diaad.io.discovery import find_matching_files
from diaad.transcripts.transcript_tables import extract_transcript_data
from diaad.metadata.utils import validate_columns
from diaad.metadata.blinding import blind_file_identifiers, write_blind_codebook


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
class CodingTemplateConfig:
    """
    Normalized settings for generic manual coding template generation.
    """
    num_bins: int = DEFAULT_NUM_BINS
    stimulus_field: str = DEFAULT_STIMULUS_FIELD

    @property
    def use_stimulus(self) -> bool:
        return bool(str(self.stimulus_field or "").strip())


def _normalize_coder_ids(
    coder_ids: Optional[list[str] | tuple[str, ...] | str],
) -> list[str]:
    """
    Normalize coder IDs to a non-empty list.

    Blank / None becomes a single blank coder_id column.
    """
    if coder_ids is None:
        return [""]

    if isinstance(coder_ids, str):
        coder_ids = [coder_ids]

    normalized = [str(x).strip() for x in coder_ids]
    normalized = [x for x in normalized if x != ""]

    return normalized or [""]


def _require_columns(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
    """
    Validate required columns are present.
    """
    validate_columns(df, cols, df_name=df_name)


def _coerce_bin_labels(num_bins: int) -> list[int]:
    """
    Return canonical bin labels as 1..num_bins.
    """
    if num_bins < 1:
        raise ValueError(f"num_bins must be >= 1, got {num_bins}.")
    return list(range(1, num_bins + 1))


def _prepare_stimulus_lookup(
    sample_df: pd.DataFrame,
    stimulus_field: str,
) -> pd.DataFrame:
    """
    Return sample_id + stimulus lookup table.

    The output stimulus column is renamed to 'stimulus'.
    """
    _require_columns(sample_df, ["sample_id", stimulus_field], "sample_df")

    stim_df = sample_df.loc[:, ["sample_id", stimulus_field]].copy()
    stim_df = stim_df.rename(columns={stimulus_field: "stimulus"})
    stim_df = stim_df.drop_duplicates(subset=["sample_id"])

    return stim_df


def _resolve_available_stimulus_field(
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


def _prepare_utterance_template(
    utt_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    config: CodingTemplateConfig,
) -> pd.DataFrame:
    """
    Build the base utterance-level coding template.

    Columns:
        sample_id, utterance_id, stimulus?, utterance
    """
    _require_columns(utt_df, ["sample_id", "utterance_id", "utterance"], "utt_df")

    out = utt_df.loc[:, ["sample_id", "utterance_id", "utterance"]].copy()

    stimulus_field = _resolve_available_stimulus_field(
        sample_df,
        config.stimulus_field,
    )

    if stimulus_field:
        stim_df = _prepare_stimulus_lookup(sample_df, stimulus_field)
        out = out.merge(stim_df, on="sample_id", how="left", validate="m:1")
        out = out.loc[:, ["sample_id", "utterance_id", "stimulus", "utterance"]]

    return out


def _prepare_sample_template(
    sample_df: pd.DataFrame,
    config: CodingTemplateConfig,
) -> pd.DataFrame:
    """
    Build the base sample-level coding template.

    Columns:
        sample_id, stimulus?, bin
    """
    _require_columns(sample_df, ["sample_id"], "sample_df")

    cols = ["sample_id"]
    stimulus_field = _resolve_available_stimulus_field(
        sample_df,
        config.stimulus_field,
    )
    if stimulus_field:
        cols.append(stimulus_field)

    out = sample_df.loc[:, cols].copy().drop_duplicates(subset=["sample_id"])

    if stimulus_field:
        out = out.rename(columns={stimulus_field: "stimulus"})
        out["bin"] = pd.NA
        out = out.loc[:, ["sample_id", "stimulus", "bin"]]
    else:
        out["bin"] = pd.NA
        out = out.loc[:, ["sample_id", "bin"]]

    return out


def _expand_by_coder(
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


def _apply_optional_identifier_blinding(
    df: pd.DataFrame,
    *,
    blind: bool,
    blinding_config,
    existing_codebook: pd.DataFrame | None = None,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Optionally blind identifiers for manual coding files.

    Returns
    -------
    blinded_or_raw_df, codebook_df
    """
    if not blind:
        return df.copy(), pd.DataFrame()

    blinded_df, codebook_df = blind_file_identifiers(
        df=df,
        config=blinding_config,
        existing_codebook=existing_codebook,
        seed=seed,
    )
    return blinded_df, codebook_df


def _find_transcript_table(
    input_dir: str | Path,
    output_dir: str | Path,
) -> Path | None:
    """
    Locate the transcript table workbook for template generation.
    """
    transcript_tables = find_matching_files(
        directories=[input_dir, output_dir],
        search_base="transcript_tables",
    )

    if not transcript_tables:
        logger.error("No transcript_tables file found.")
        return None

    if len(transcript_tables) > 1:
        logger.warning(
            "Multiple transcript tables detected. "
            f"Processing only the first returned file: {_rel(transcript_tables[0])}"
        )

    return transcript_tables[0]


def _resolve_template_coder_ids(num_coders: int) -> list[str]:
    """
    Return normalized template coder IDs.
    """
    if num_coders <= 0:
        return [""]
    return [str(i) for i in range(1, num_coders + 1)]


def _assign_template_coders(
    df: pd.DataFrame,
    *,
    coder_ids: list[str],
) -> tuple[pd.DataFrame, list[list[str]], list[tuple[str, ...]]]:
    """
    Assign primary coders by sample while keeping sample rows together.
    """
    out = df.copy()
    sample_ids = list(out["sample_id"].drop_duplicates())

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
        out.loc[out["sample_id"].isin(seg), "coder_id"] = assn[0]

    return out, segments, assignments


def _build_reliability_subset(
    df: pd.DataFrame,
    *,
    frac: float,
    coder_ids: list[str],
    segments: list[list[str]],
    assignments: list[tuple[str, ...]],
) -> pd.DataFrame | None:
    """
    Build a sample-preserving reliability subset from a primary template.
    """
    if frac == 0:
        logger.info("frac=0 detected; no reliability subset will be generated.")
        return None

    sample_ids = list(df["sample_id"].drop_duplicates())
    if not sample_ids:
        logger.warning("No sample IDs available for reliability subset generation.")
        return None

    if len(coder_ids) <= 1:
        n_rel = calc_subset_size(frac=frac, samples=sample_ids)
        rel_samples = random.sample(sample_ids, k=n_rel)
        rel_df = df[df["sample_id"].isin(rel_samples)].copy()
        rel_df["coder_id"] = coder_ids[0] if coder_ids else ""
        return rel_df

    rel_subsets: list[pd.DataFrame] = []

    for seg, assn in zip(segments, assignments):
        if not seg:
            continue

        n_rel = calc_subset_size(frac=frac, samples=seg)
        rel_samples = random.sample(seg, k=n_rel)
        rel_df = df[df["sample_id"].isin(rel_samples)].copy()
        rel_df["coder_id"] = assn[1]
        rel_subsets.append(rel_df)

    if not rel_subsets:
        return None

    return pd.concat(rel_subsets, ignore_index=True)


def _write_template_exports(
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


def build_utterance_coding_template(
    transcript_table_path: str | Path,
    *,
    coder_ids: Optional[list[str] | tuple[str, ...] | str] = None,
    template_config: CodingTemplateConfig | None = None,
    blind: bool = False,
    blinding_config=None,
    existing_codebook: pd.DataFrame | None = None,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build an utterance-level blank coding template from transcript tables.

    Output columns:
        sample_id, utterance_id, coder_id, stimulus?, utterance
    """
    template_config = template_config or CodingTemplateConfig()
    coder_ids = _normalize_coder_ids(coder_ids)

    sample_df = extract_transcript_data(transcript_table_path, kind="sample")
    utt_df = extract_transcript_data(transcript_table_path, kind="utterance")

    out = _prepare_utterance_template(
        utt_df=utt_df,
        sample_df=sample_df,
        config=template_config,
    )
    out = _expand_by_coder(out, coder_ids, insert_after="utterance_id")

    if blind:
        if blinding_config is None:
            raise ValueError("blinding_config is required when blind=True.")
        out, codebook_df = _apply_optional_identifier_blinding(
            out,
            blind=True,
            blinding_config=blinding_config,
            existing_codebook=existing_codebook,
            seed=seed,
        )
    else:
        codebook_df = pd.DataFrame()

    logger.info(
        f"Prepared utterance-level coding template with {len(out)} row(s)."
    )
    return out, codebook_df


def build_sample_coding_template(
    transcript_table_path: str | Path,
    *,
    coder_ids: Optional[list[str] | tuple[str, ...] | str] = None,
    template_config: CodingTemplateConfig | None = None,
    blind: bool = False,
    blinding_config=None,
    existing_codebook: pd.DataFrame | None = None,
    seed: int = 99,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a sample-level blank coding template from transcript tables.

    Output columns:
        sample_id, coder_id, stimulus?, bin
    """
    template_config = template_config or CodingTemplateConfig()
    coder_ids = _normalize_coder_ids(coder_ids)

    sample_df = extract_transcript_data(transcript_table_path, kind="sample")
    out = _prepare_sample_template(sample_df=sample_df, config=template_config)
    out = _expand_by_coder(out, coder_ids, insert_after="sample_id")

    if blind:
        if blinding_config is None:
            raise ValueError("blinding_config is required when blind=True.")
        out, codebook_df = _apply_optional_identifier_blinding(
            out,
            blind=True,
            blinding_config=blinding_config,
            existing_codebook=existing_codebook,
            seed=seed,
        )
    else:
        codebook_df = pd.DataFrame()

    logger.info(
        f"Prepared sample-level coding template with {len(out)} row(s)."
    )
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

    Useful if the user wants an initial blank sample-level template with
    pre-assigned coding bins.
    """
    if sample_id_col not in df.columns:
        raise ValueError(f"'{sample_id_col}' not found in dataframe.")

    bin_labels = _coerce_bin_labels(num_bins)

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

    return out


def write_coding_template(
    df: pd.DataFrame,
    path: str | Path,
    *,
    codebook_df: pd.DataFrame | None = None,
    codebook_path: str | Path | None = None,
) -> Path:
    """
    Write a coding template to disk.

    Supports .xlsx and .csv. Optionally writes a paired codebook.
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

    logger.info(f"Wrote coding template: {_rel(path)}")

    if codebook_df is not None and not codebook_df.empty:
        if codebook_path is None:
            stem = path.with_suffix("")
            codebook_path = f"{stem}_codebook.xlsx"
        write_blind_codebook(codebook_df, codebook_path)

    return path


def make_utterance_coding_template(
    transcript_table_path: str | Path,
    output_path: str | Path,
    *,
    coder_ids: Optional[list[str] | tuple[str, ...] | str] = None,
    num_bins: int = DEFAULT_NUM_BINS,
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
    config = CodingTemplateConfig(
        num_bins=num_bins,
        stimulus_field=stimulus_field,
    )
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
) -> Path:
    """
    Convenience wrapper: build and write a sample-level coding template.
    """
    config = CodingTemplateConfig(
        num_bins=num_bins,
        stimulus_field=stimulus_field,
    )
    df, codebook_df = build_sample_coding_template(
        transcript_table_path,
        coder_ids=coder_ids,
        template_config=config,
        blind=blind,
        blinding_config=blinding_config,
        existing_codebook=existing_codebook,
        seed=seed,
    )

    if prefill_bins:
        df = add_balanced_bins(df, num_bins=config.num_bins)

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
    transcript_table = _find_transcript_table(input_dir, output_dir)
    if transcript_table is None:
        return None

    template_dir = Path(output_dir) / TEMPLATE_SUBDIR
    template_dir.mkdir(parents=True, exist_ok=True)

    config = CodingTemplateConfig(
        stimulus_field=stimulus_field,
    )

    df, _ = build_utterance_coding_template(
        transcript_table_path=transcript_table,
        coder_ids=None,
        template_config=config,
        blind=False,
        seed=seed,
    )

    coder_ids = _resolve_template_coder_ids(num_coders)
    df, segments, assignments = _assign_template_coders(df, coder_ids=coder_ids)
    rel_df = _build_reliability_subset(
        df,
        frac=frac,
        coder_ids=coder_ids,
        segments=segments,
        assignments=assignments,
    )

    codebook_df = pd.DataFrame()
    export_df = df
    export_rel_df = rel_df

    if blinding_config is not None and getattr(blinding_config, "blind_files", False):
        export_df, codebook_df = _apply_optional_identifier_blinding(
            df,
            blind=True,
            blinding_config=blinding_config,
            seed=seed,
        )

        if rel_df is not None:
            export_rel_df, _ = _apply_optional_identifier_blinding(
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

    return _write_template_exports(
        primary_df=export_df,
        primary_path=template_dir / UTTERANCE_TEMPLATE_FILENAME,
        reliability_df=export_rel_df,
        reliability_path=template_dir / UTTERANCE_RELIABILITY_FILENAME,
        codebook_df=codebook_df,
        codebook_path=template_dir / UTTERANCE_CODEBOOK_FILENAME,
    )


def make_sample_template_files(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    frac: float,
    num_coders: int,
    stimulus_field: str = DEFAULT_STIMULUS_FIELD,
    prefill_bins: bool = False,
    blinding_config=None,
    seed: int = 99,
) -> Path | None:
    """
    Create sample-level primary and reliability template workbooks.
    """
    transcript_table = _find_transcript_table(input_dir, output_dir)
    if transcript_table is None:
        return None

    template_dir = Path(output_dir) / TEMPLATE_SUBDIR
    template_dir.mkdir(parents=True, exist_ok=True)

    config = CodingTemplateConfig(
        stimulus_field=stimulus_field,
    )

    df, _ = build_sample_coding_template(
        transcript_table_path=transcript_table,
        coder_ids=None,
        template_config=config,
        blind=False,
        seed=seed,
    )

    if prefill_bins:
        df = add_balanced_bins(df, num_bins=config.num_bins)

    coder_ids = _resolve_template_coder_ids(num_coders)
    df, segments, assignments = _assign_template_coders(df, coder_ids=coder_ids)
    rel_df = _build_reliability_subset(
        df,
        frac=frac,
        coder_ids=coder_ids,
        segments=segments,
        assignments=assignments,
    )

    codebook_df = pd.DataFrame()
    export_df = df
    export_rel_df = rel_df

    if blinding_config is not None and getattr(blinding_config, "blind_files", False):
        export_df, codebook_df = _apply_optional_identifier_blinding(
            df,
            blind=True,
            blinding_config=blinding_config,
            seed=seed,
        )

        if rel_df is not None:
            export_rel_df, _ = _apply_optional_identifier_blinding(
                rel_df,
                blind=True,
                blinding_config=blinding_config,
                existing_codebook=codebook_df,
                seed=seed,
            )

    logger.info(
        "Prepared sample template exports: primary=%s rows, reliability=%s rows.",
        len(export_df),
        0 if export_rel_df is None else len(export_rel_df),
    )

    return _write_template_exports(
        primary_df=export_df,
        primary_path=template_dir / SAMPLE_TEMPLATE_FILENAME,
        reliability_df=export_rel_df,
        reliability_path=template_dir / SAMPLE_RELIABILITY_FILENAME,
        codebook_df=codebook_df,
        codebook_path=template_dir / SAMPLE_CODEBOOK_FILENAME,
    )
