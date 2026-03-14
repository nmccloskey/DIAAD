import random
import re
import contractions
import pandas as pd
from tqdm import tqdm
import num2words as n2w
from pathlib import Path

from diaad.utils.logger import logger, _rel
from diaad.utils.auxiliary import (
    calc_subset_size,
    find_matching_files,
    extract_transcript_data,
)
from diaad.coding.utils import segment, assign_coders


# ------------------------------------------------------------------
# Text normalization / automated first-pass counting
# ------------------------------------------------------------------

FILLER_TOKENS = {
    "uh", "um", "er", "erm", "hm", "hmm", "mm", "mhm",
    "uhh", "umm", "eh", "ah", "oh",
    "xxx", "yyy", "www",
}


def count_words(text: str) -> int:
    """
    Generate an automated first-pass word count for one utterance.

    This is intentionally heuristic rather than dictionary-based so it works
    better on aphasia data, phonetic spellings, dialect forms, and irregular
    transcript text.

    Parameters
    ----------
    text : str

    Returns
    -------
    int
        Automated first-pass word count.
    """
    if pd.isna(text):
        return 0

    text = str(text).strip().lower()
    if not text:
        return 0

    text = text.replace("\xa0", " ")

    # Expand contractions tokenwise where possible.
    text = " ".join(contractions.fix(tok) for tok in text.split())

    # Convert integers to words.
    text = re.sub(
        r"\d+",
        lambda m: n2w.num2words(int(m.group(0))),
        text,
    )

    # Remove or simplify common CHAT / annotation-like content.
    text = re.sub(r"\[[^\]]*\]", " ", text)            # [=! ...], [//], etc.
    text = re.sub(r"<[^>]*>", " ", text)               # angle-bracket groupings
    text = re.sub(r"\(([^)]*)\)", r" \1 ", text)       # keep contents, drop parens
    text = re.sub(r"&[A-Za-z0-9_:+-]+", " ", text)     # &-um, &+xxx, etc.
    text = re.sub(r"@[\w:]+", " ", text)               # @o, @q, etc.
    text = re.sub(r"[+\/]+", " ", text)
    text = re.sub(r"[_=~^*]", " ", text)

    # Normalize punctuation / separators.
    text = re.sub(r"[-–—]", " ", text)
    text = re.sub(r"[^\w\s']", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return 0

    tokens = []
    for tok in text.split():
        tok = tok.strip("'")
        if not tok:
            continue
        if tok in FILLER_TOKENS:
            continue
        if re.fullmatch(r"x+", tok):
            continue
        if tok == "cl":
            continue
        if re.search(r"[A-Za-z0-9]", tok):
            tokens.append(tok)

    return len(tokens)


# ------------------------------------------------------------------
# Input discovery / reading
# ------------------------------------------------------------------

def _shuffle_by_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Shuffle rows at the sample level while preserving within-sample order."""
    if "sample_id" not in df.columns:
        raise KeyError("Expected column 'sample_id' in input dataframe.")

    subdfs = [subdf for _, subdf in df.groupby("sample_id", sort=False)]
    random.shuffle(subdfs)
    return pd.concat(subdfs, ignore_index=True) if subdfs else df.copy()


def _find_input_files(input_dir, output_dir):
    """
    Prefer CU coding files. If none are found, fall back to transcript tables.
    """
    cu_files = find_matching_files(
        directories=[input_dir, output_dir],
        search_base="cu_coding_by_utterance",
    )

    if cu_files:
        logger.info(
            f"Found {len(cu_files)} CU coding file(s); using these for word-count prep."
        )
        return "cu", cu_files

    transcript_tables = find_matching_files(
        directories=[input_dir, output_dir],
        search_base="transcript_tables",
    )

    if transcript_tables:
        logger.info(
            f"No CU coding files found. Found {len(transcript_tables)} transcript table file(s); "
            "using these for automated first-pass word-count prep."
        )
        return "transcript", transcript_tables

    return None, []


def _read_source_file(file: Path, source_type: str) -> pd.DataFrame:
    """Read one CU or transcript-table file and shuffle by sample."""
    if source_type == "cu":
        df = pd.read_excel(file)
    elif source_type == "transcript":
        df = extract_transcript_data(file, type="utterance")
    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    df = _shuffle_by_sample(df)
    logger.info(f"Read and shuffled {_rel(file)}")
    return df


def _extract_labels(file: Path, tiers) -> list[str]:
    """Extract tier labels from filename."""
    labels = [t.match(file.name, return_none=True) for t in tiers.values()]
    return [lab for lab in labels if lab]


# ------------------------------------------------------------------
# CU neutrality logic
# ------------------------------------------------------------------

def _is_neutral_value(value) -> bool:
    """
    Return True if a CU cell should be treated as neutral / non-countable.

    Neutral values include:
      - missing values
      - blank strings
      - NA-like strings
      - neutral / n / 0 / false-like strings
    """
    if pd.isna(value):
        return True

    text = str(value).strip().lower()
    if not text:
        return True

    neutral_values = {
        "na",
        "n/a",
        "nan",
        "none",
        "neutral",
        "neu",
        "n",
        "0",
        "false",
        "no",
    }
    return text in neutral_values


def _get_cu_columns(df: pd.DataFrame) -> list[str]:
    """
    Find CU coding columns matching:
      - cu
      - *_cu
      - cu_*
      - *_cu_*
    """
    cu_cols = []
    for col in df.columns:
        col_l = col.lower()
        if (
            col_l == "cu"
            or col_l.startswith("cu_")
            or col_l.endswith("_cu")
            or "_cu_" in col_l
        ):
            cu_cols.append(col)
    return cu_cols


def _row_is_countable_from_cu(row: pd.Series, cu_cols: list[str]) -> bool:
    """
    An utterance is countable if any CU column is non-neutral.
    If all CU columns are neutral, the utterance gets word_count = 'NA'.
    """
    if not cu_cols:
        return True

    return any(not _is_neutral_value(row.get(col)) for col in cu_cols)


# ------------------------------------------------------------------
# Data preparation / blinding
# ------------------------------------------------------------------

BLINDED_COLUMNS = [
    "sample_id",
    "utterance_id",
    "speaker",
    "utterance",
    "comment",
    "id",
    "word_count",
    "wc_comment",
]


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all blinded output columns exist.
    """
    df = df.copy()
    for col in BLINDED_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def _prepare_wc_df(
    df: pd.DataFrame,
    source_type: str,
    exclude_participants: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build blinded primary word-count dataframe.

    Rules
    -----
    - If speaker is in exclude_participants -> word_count = 'NA'
    - Else if source_type == 'cu' and all CU columns are neutral -> word_count = 'NA'
    - Else compute automated first-pass count from utterance text
    """
    df = df.copy()
    df = _ensure_required_columns(df)

    exclude_set = {str(x).strip().lower() for x in (exclude_participants or [])}
    cu_cols = _get_cu_columns(df) if source_type == "cu" else []

    def compute_wc(row):
        speaker = str(row.get("speaker", "")).strip().lower()
        if speaker in exclude_set:
            return "NA"

        if source_type == "cu" and not _row_is_countable_from_cu(row, cu_cols):
            return "NA"

        return count_words(row.get("utterance", ""))

    df["word_count"] = df.apply(compute_wc, axis=1)
    df["wc_comment"] = ""

    # Blind / restrict output columns.
    df = df[BLINDED_COLUMNS].copy()
    return df


# ------------------------------------------------------------------
# Coder assignment
# ------------------------------------------------------------------

def _sample_reliability_subset(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Sample a whole-sample reliability subset."""
    unique_ids = list(df["sample_id"].drop_duplicates())
    n_rel_samples = calc_subset_size(frac=frac, samples=unique_ids)

    if n_rel_samples <= 0:
        return df.iloc[0:0].copy()

    rel_samples = random.sample(unique_ids, k=n_rel_samples)
    return df[df["sample_id"].isin(rel_samples)].copy()


def _assign_wc_coders(wc_df: pd.DataFrame, coders: list[str], frac: float):
    """
    Assign coder IDs for primary and reliability word-count workbooks.

    Rules
    -----
    frac == 0:
        no reliability subset
    len(coders) == 0:
        blank id in both primary and reliability
    len(coders) == 1:
        same id in both primary and reliability
    len(coders) >= 2:
        segment samples and assign via imported helpers
    """
    wc_df = wc_df.copy()

    if "id" not in wc_df.columns:
        wc_df["id"] = ""

    # frac = 0 -> empty reliability workbook
    if frac == 0:
        wc_rel_df = wc_df.iloc[0:0].copy()
        logger.info("frac=0, so no reliability subset was created.")
        return wc_df, wc_rel_df

    # 0 coders -> blank IDs
    if len(coders) == 0:
        wc_df["id"] = ""
        wc_rel_df = _sample_reliability_subset(wc_df, frac=frac)
        wc_rel_df["id"] = ""
        logger.info("No coders supplied; created primary/reliability files with blank ID column.")
        return wc_df, wc_rel_df

    # 1 coder -> same ID everywhere
    if len(coders) == 1:
        coder = coders[0]
        wc_df["id"] = coder
        wc_rel_df = _sample_reliability_subset(wc_df, frac=frac)
        wc_rel_df["id"] = coder
        logger.info(
            f"One coder supplied ({coder}); created reliability subset with the same ID values."
        )
        return wc_df, wc_rel_df

    # 2+ coders -> segment samples and assign IDs
    assignments = assign_coders(coders)
    unique_ids = list(wc_df["sample_id"].drop_duplicates())
    sample_segments = segment(unique_ids, n=len(coders))

    rel_subsets = []

    for seg, assn in zip(sample_segments, assignments):
        if not seg:
            continue

        primary_coder = assn[0]
        rel_coder = assn[1]

        wc_df.loc[wc_df["sample_id"].isin(seg), "id"] = primary_coder

        seg_df = wc_df[wc_df["sample_id"].isin(seg)].copy()
        rel_df = _sample_reliability_subset(seg_df, frac=frac)
        rel_df["id"] = rel_coder
        rel_subsets.append(rel_df)

    wc_rel_df = (
        pd.concat(rel_subsets, ignore_index=True)
        if rel_subsets
        else wc_df.iloc[0:0].copy()
    )

    logger.info(f"Reliability subset contains {len(wc_rel_df)} utterance rows.")
    return wc_df, wc_rel_df


# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------

def _write_wc_outputs(
    wc_df: pd.DataFrame,
    wc_rel_df: pd.DataFrame,
    word_count_dir: Path,
    labels: list[str],
):
    """Write primary and reliability word-count workbooks."""
    out_dir = Path(word_count_dir, *labels)
    out_dir.mkdir(parents=True, exist_ok=True)

    lab_str = "_".join(labels) + "_" if labels else ""

    outputs = {
        f"{lab_str}word_counting.xlsx": wc_df,
        f"{lab_str}word_count_reliability.xlsx": wc_rel_df,
    }

    for fname, df in outputs.items():
        fpath = out_dir / fname
        try:
            df.to_excel(fpath, index=False)
            logger.info(f"Wrote {_rel(fpath)}")
        except Exception as e:
            logger.error(f"Failed to write {_rel(fpath)}: {e}")


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def make_word_count_files(
    tiers,
    frac,
    coders,
    input_dir,
    output_dir,
    exclude_participants: list[str] | None = None,
):
    """
    Create blinded word-count coding and reliability workbooks from either:
      1. CU coding-by-utterance files (preferred), or
      2. transcript tables (fallback).

    Behavior
    --------
    - CU files are preferred so neutral utterances can be excluded from counting.
    - If no CU files are found, transcript tables are used for automated first-pass counts.
    - Output columns are restricted to:
        sample_id, utterance_id, speaker, utterance, comment, id, word_count, wc_comment
    - Any utterance from exclude_participants gets word_count = 'NA'.
    - For CU files, if all CU columns for an utterance are neutral, word_count = 'NA'.

    Reliability behavior
    --------------------
    - frac == 0 -> empty reliability workbook
    - 0 coders  -> blank id column
    - 1 coder   -> same id in primary and reliability
    - 2+ coders -> segmented sample assignment via helpers

    Parameters
    ----------
    tiers : dict[str, Tier]
    frac : float
    coders : list[str]
    input_dir : Path | str
    output_dir : Path | str
    exclude_participants : list[str] | None
        Speakers whose utterances should be explicitly marked neutral ('NA').
    """
    word_count_dir = Path(output_dir) / "word_counts"
    word_count_dir.mkdir(parents=True, exist_ok=True)

    source_type, files = _find_input_files(input_dir, output_dir)

    if not files:
        logger.warning("No CU coding files or transcript tables were found for word-count prep.")
        return

    for file in tqdm(files, desc="Generating word count files"):
        try:
            df = _read_source_file(file, source_type=source_type)
            labels = _extract_labels(file, tiers)

            wc_df = _prepare_wc_df(
                df=df,
                source_type=source_type,
                exclude_participants=exclude_participants,
            )

            wc_df, wc_rel_df = _assign_wc_coders(
                wc_df=wc_df,
                coders=coders,
                frac=frac,
            )

            # Reassert blinded column order after coder assignment.
            wc_df = _ensure_required_columns(wc_df)[BLINDED_COLUMNS].copy()
            wc_rel_df = _ensure_required_columns(wc_rel_df)[BLINDED_COLUMNS].copy()

            _write_wc_outputs(
                wc_df=wc_df,
                wc_rel_df=wc_rel_df,
                word_count_dir=word_count_dir,
                labels=labels,
            )

        except Exception as e:
            logger.error(f"Failed processing {_rel(file)}: {e}")
