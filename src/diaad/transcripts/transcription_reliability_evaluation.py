from __future__ import annotations

import re
from pathlib import Path
from typing import Union, List

import pandas as pd
import pylangacq
from tqdm import tqdm

from Levenshtein import distance
from Bio.Align import PairwiseAligner

from psair.core.logger import logger, get_rel_path


ALIGNMENTS_SUBDIR = "global_alignments"


def percent_difference(a, b):
    try:
        a, b = float(a), float(b)
        if a == 0 and b == 0:
            return 0.0
        denom = (abs(a) + abs(b)) / 2.0
        return (abs(a - b) / denom) * 100.0 if denom != 0 else 0.0
    except Exception:
        return float("nan")


def scrub_clan(text: str) -> str:
    """
    Remove CLAN markup while keeping only speech-relevant material.

    - Keep common disfluencies like &um, &uh, &h (→ 'um', 'uh', 'h')
    - Remove gesture and non-speech codes (e.g., &=points:leg, =laughs, <...>, ((...)), {...}, [/], [//])
    - Remove any remaining bracketed or symbolic markup
    - Preserve ordinary words, punctuation (.!?), and apostrophes

    Example
    -------
    Input : "but &-um &-uh &+h hurt &=points:leg oh well"
    Output: "but um uh h hurt oh well"
    """
    # normalize speech-like tokens (&um, &-uh, &+h → um, uh, h)
    text = re.sub(r"(?<!\S)&[-+]?([a-zA-Z]+)\b", r"\1", text)
    # remove all other &-prefixed tokens
    text = re.sub(r"(?<!\S)&\S+", " ", text)

    # remove structural / paralinguistic markup
    text = re.sub(r"\(\([^)]*\)\)", " ", text)
    text = re.sub(r"\{[^}]+\}", " ", text)
    text = re.sub(r"\[\/*\]", " ", text)
    text = re.sub(r"\[[^]]*\]", " ", text)

    # remove =codes (e.g., =laughs)
    text = re.sub(r"(?<!\S)=[^\s]+", " ", text)

    # remove non-speech symbols except .!? and apostrophes
    text = re.sub(r"[^\w\s'!.?]", " ", text)

    # tidy whitespace
    text = re.sub(r"\s+(?=[.!?])", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def process_corrections(text: str, prefer_correction: bool = True) -> str:
    """
    Handle CLAN correction notation ([: correction] [*]) according to preference.

    prefer_correction=True  -> replace with correction
    prefer_correction=False -> keep original (remove correction markup)
    """
    if prefer_correction:
        # Replace "orig [: corr] [*]" with "corr"
        text = re.sub(r"(\S+)\s*\[:\s*([^\]]+?)\s*\]\s*\[\*\]", r"\2", text)
    else:
        # Replace "orig [: corr] [*]" with "orig"
        text = re.sub(r"(\S+)\s*\[:\s*([^\]]+?)\s*\]\s*\[\*\]", r"\1", text)

    # Clean up spacing
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def extract_cha_text(
    source: Union[str, pylangacq.Reader],
    exclude_participants: List[str] = None,
) -> str:
    """
    Extract utterance text only when a pylangacq.Reader is provided.

    For DIAAD: accepts a Reader and returns concatenated utterances.
    For DIAAD: if input is already a text string, it is returned unchanged
    (no pylangacq parsing).

    Parameters
    ----------
    source : str or pylangacq.Reader
        - pylangacq.Reader → extract utterances
        - str → returned unchanged (already plain text)
    exclude_participants : list[str], optional
        Participant codes to exclude (e.g., ['INV']).
    """
    exclude_participants = exclude_participants or []

    try:
        if isinstance(source, pylangacq.Reader):
            parts = []
            for line in source.utterances():
                if line.participant in exclude_participants:
                    continue
                utterance = line.tiers.get(line.participant, "")
                utterance = re.sub(r"\s+(?=[.!?])", "", utterance)
                parts.append(utterance)
            return " ".join(parts).strip()

        elif isinstance(source, str):
            # Return string unchanged — already text
            return source.strip()

        else:
            raise TypeError(
                f"Unsupported input type for extract_cha_text: {type(source)}"
            )

    except Exception as e:
        logger.error(f"extract_cha_text failed: {e}")
        return ""


def process_utterances(
    chat_data: Union[str, pylangacq.Reader],
    *,
    exclude_participants: List[str] = None,
    strip_clan: bool = True,
    prefer_correction: bool = True,
    lowercase: bool = True,
) -> str:
    """
    Unified utterance-processing pipeline for both DIAAD (Reader input)
    and DIAAD (plain text input).

    Behavior
    --------
    - If `chat_data` is a pylangacq.Reader, extract and process utterances.
    - If `chat_data` is already a string, skip pylangacq and process directly.
    - Optionally remove CLAN markup and/or apply correction preferences.

    Parameters
    ----------
    chat_data : str or pylangacq.Reader
        CHAT text (string) or Reader object.
    exclude_participants : list[str], optional
        Participants to omit (used only for Reader input).
    strip_clan : bool
        If True, scrub CLAN markup.
    prefer_correction : bool
        Policy for handling [: correction] [*].
    lowercase : bool
        Lowercase final output.
    """
    # 1. Extract text (Reader → concatenated utterances; str → unchanged)
    text = extract_cha_text(chat_data, exclude_participants)
    if not text:
        return ""

    # 2. Handle corrections
    text = process_corrections(text, prefer_correction)

    # 3. Optionally strip CLAN markup
    if strip_clan:
        text = scrub_clan(text)
    else:
        text = re.sub(r"[ \t]+", " ", text)

    # 4. Final normalization
    text = re.sub(r"\s+", " ", text).strip()
    if lowercase:
        text = text.lower()

    return text


# Helper function to wrap lines at approximately 80 characters or based on delimiters
def _wrap_text(text, width=80):
    """
    Wrap text to a specified width or based on utterance delimiters for better readability.
    """
    words = text.split()
    lines = []
    current_line = words[0]
    
    for word in words[1:]:
        # Add the word to the current line if it doesn't exceed the width limit
        if len(current_line) + len(word) + 1 <= width:
            current_line += ' ' + word
        else:
            # If the width limit is exceeded, append the current line and start a new one
            lines.append(current_line)
            current_line = word

    # Append the last line if there is any content left
    if current_line:
        lines.append(current_line)

    return lines


def write_reliability_report(transc_rel_subdf, report_path):
    """
    Write a plain-text transcription-reliability report.

    Parameters
    ----------
    transc_rel_subdf : pandas.DataFrame
        One row per sample. Must contain a numeric column
        'levenshtein_similarity' whose values lie in [0, 1].
    report_path : str | pathlib.Path
        Full path to the output .txt file.
    """

    try:
        # ── sanity checks ──────────────────────────────────────────────────────
        if 'levenshtein_similarity' not in transc_rel_subdf.columns:
            raise KeyError("'levenshtein_similarity' column is missing.")

        ls = transc_rel_subdf['levenshtein_similarity'].astype(float).dropna()
        n_samples = len(ls)
        mean_ls   = ls.mean()
        sd_ls     = ls.std()
        min_ls    = ls.min()
        max_ls    = ls.max()

        # ── similarity bands ───────────────────────────────────────────────────
        bands = {
            "Excellent (≥ .90)":        (ls >= 0.90),
            "Sufficient (.80 – .89)":   ((ls >= 0.80) & (ls < 0.90)),
            "Min. acceptable (.70 – .79)": ((ls >= 0.70) & (ls < 0.80)),
            "Below .70":               (ls < 0.70),
        }
        counts = {label: mask.sum() for label, mask in bands.items()}

        # ── compose the report text ────────────────────────────────────────────
        header = "Transcription Reliability Report"

        lines = [
            header,
            "=" * len(header),
            f"Number of samples: {n_samples}",
            "",
            f"Levenshtein similarity score summary stats:",
            f"  • Average: {mean_ls:.3f}",
            f"  • Standard Deviation: {sd_ls:.3f}",
            f"  • Min: {min_ls:.3f}",
            f"  • Max: {max_ls:.3f}",
            "",
            "Similarity bands:",
        ]
        for label, count in counts.items():
            pct = count / n_samples * 100 if n_samples else 0
            lines.append(f"  • {label}: {count} ({pct:.1f}%)")

        report_text = "\n".join(lines)

        # ── write to disk ──────────────────────────────────────────────────────
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info("Successfully wrote transcription reliability report to %s", get_rel_path(report_path))

    except Exception as e:
        logger.error("Failed to write transcription reliability report to %s: %s", get_rel_path(report_path), e)
        raise


def _compute_simple_stats(org_text: str, rel_text: str):
    org_tokens = org_text.split()
    rel_tokens = rel_text.split()
    org_num_tokens = len(org_tokens)
    rel_num_tokens = len(rel_tokens)
    pdiff_num_tokens = percent_difference(org_num_tokens, rel_num_tokens)

    org_num_chars = len(org_text)
    rel_num_chars = len(rel_text)
    pdiff_num_chars = percent_difference(org_num_chars, rel_num_chars)

    return {
        "org_num_tokens": org_num_tokens,
        "rel_num_tokens": rel_num_tokens,
        "perc_diff_num_tokens": pdiff_num_tokens,
        "org_num_chars": org_num_chars,
        "rel_num_chars": rel_num_chars,
        "perc_diff_num_chars": pdiff_num_chars,
    }


def _levenshtein_metrics(org_text: str, rel_text: str):
    Ldist = distance(org_text, rel_text)
    max_len = max(len(org_text), len(rel_text)) or 1
    Lscore = 1 - (Ldist / max_len)
    return {"levenshtein_distance": Ldist, "levenshtein_similarity": Lscore}


def _needleman_wunsch_global(org_text: str, rel_text: str):
    aligner = PairwiseAligner()
    aligner.mode = "global"
    alignments = aligner.align(org_text, rel_text)
    best = alignments[0]
    best_score = best.score
    norm = best_score / (max(len(org_text), len(rel_text)) or 1)
    return {"needleman_wunsch_score": best_score,
            "needleman_wunsch_norm": norm,
            "alignment": best}


def _format_alignment_output(alignment, best_score: float, normalized_score: float):
    # Extract the two aligned sequences; Biopython's pairwise alignment object behaves like a 2-row alignment
    seq1 = alignment[0]
    seq2 = alignment[1]

    seq1_lines = _wrap_text(seq1)
    seq2_lines = _wrap_text(seq2)

    out = []
    out.append(f"Global alignment score: {best_score}")
    out.append(f"Normalized score (by length): {normalized_score}")
    out.append("")

    for s1, s2 in zip(seq1_lines, seq2_lines):
        out.append(f"Sequence 1: {s1}")
        align_line = "".join("|" if a == b else " " for a, b in zip(s1, s2))
        out.append(f"Alignment : {align_line}")
        out.append(f"Sequence 2: {s2}")
        out.append("")

    return "\n".join(out)


def _ensure_parent_dir(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _convert_cha_names(
    input_dir: str | Path,
    *,
    reliability_tag: str,
    reliability_dirname: str,
) -> dict[str, list[Path]]:
    """
    Recursively detect all reliability subdirectories under `input_dir` and create
    renamed copies of their .cha files by appending `reliability_tag` before '.cha'.

    Renamed files are written into a parallel 'renamed' subdirectory within each
    reliability directory.

    Returns
    -------
    dict[str, list[Path]]
        {
          "renamed": [list of new .cha paths],
          "originals": [list of corresponding original paths]
        }
    """
    input_dir = Path(input_dir).expanduser().resolve()
    renamed: list[Path] = []
    originals: list[Path] = []

    rel_dirs = [
        p for p in input_dir.rglob("*")
        if p.is_dir() and p.name == reliability_dirname
    ]
    if not rel_dirs:
        logger.info(
            "No '%s' subdirectories found under %s.",
            reliability_dirname,
            get_rel_path(input_dir),
        )
        return {"renamed": [], "originals": []}

    for rel_dir in rel_dirs:
        renamed_dir = rel_dir / "renamed"
        renamed_dir.mkdir(parents=True, exist_ok=True)

        for cha in rel_dir.rglob("*.cha"):
            try:
                if cha.stem.endswith(reliability_tag):
                    continue

                new_name = f"{cha.stem}{reliability_tag}.cha"
                new_path = renamed_dir / new_name

                if new_path.exists():
                    logger.warning(
                        "Renamed file already exists, skipping: %s",
                        get_rel_path(new_path),
                    )
                    continue

                new_path.write_bytes(cha.read_bytes())
                renamed.append(new_path)
                originals.append(cha)
                logger.info(
                    "Created renamed reliability copy: %s → %s",
                    cha.name,
                    new_name,
                )

            except Exception as e:
                logger.error(
                    "Failed to process reliability file %s: %s",
                    get_rel_path(cha),
                    e,
                )

    logger.info(
        "Reliability rename complete. %d file(s) copied from %d reliability dir(s).",
        len(renamed),
        len(rel_dirs),
    )
    return {"renamed": renamed, "originals": originals}


def _path_parts_for_metadata(path: Path, input_dir: Path | None = None) -> list[str]:
    """
    Return path parts scoped to the configured input directory when possible.
    """
    path = Path(path)
    scoped_path = path

    if input_dir is not None:
        try:
            scoped_path = path.resolve().relative_to(Path(input_dir).resolve())
        except ValueError:
            scoped_path = Path(path.name) if path.is_absolute() else path

    return [part for part in scoped_path.parts if part not in ("", ".")] or [path.name]


def _metadata_values_for(
    path: Path,
    metadata_fields,
    *,
    input_dir: Path | None = None,
) -> tuple:
    """Return ordered metadata field matches for a file path."""
    if not metadata_fields:
        return (path.stem,)

    parts = _path_parts_for_metadata(path, input_dir=input_dir)
    source = str(Path(*parts)) if parts else str(path)

    values = []
    for field in metadata_fields.values():
        if hasattr(field, "match_path_parts"):
            values.append(field.match_path_parts(parts, source=source))
        else:
            values.append(field.match(source))
    return tuple(values)


def _build_file_index(
    files: list[Path],
    metadata_fields,
    *,
    label: str,
    input_dir: Path | None = None,
) -> dict[tuple, Path]:
    """
    Build a metadata-value index for files, logging duplicate-key collisions.

    Parameters
    ----------
    files : list[Path]
        Files to index.
    metadata_fields : dict
        MetadataField objects used to extract transcript metadata.
    label : str
        Human-readable file class label for logging, e.g. 'original' or 'reliability'.

    Returns
    -------
    dict[tuple, Path]
        Mapping from metadata-value tuples to file paths.

    Notes
    -----
    If duplicate metadata-value tuples are detected, the first file is retained and later
    files with the same key are skipped.
    """
    index: dict[tuple, Path] = {}

    for path in files:
        try:
            key = _metadata_values_for(
                path,
                metadata_fields,
                input_dir=input_dir,
            )
            if key in index:
                logger.error(
                    "Duplicate %s metadata values detected for key %s: keeping %s, skipping %s",
                    label,
                    key,
                    get_rel_path(index[key]),
                    get_rel_path(path),
                )
                continue
            index[key] = path
        except Exception as e:
            logger.error(f"Failed to index {label} file {get_rel_path(path)}: {e}")

    logger.info("Indexed %d %s file(s).", len(index), label)
    return index


def _match_reliability_pairs(
    org_index: dict[tuple, Path],
    rel_index: dict[tuple, Path],
) -> list[tuple[tuple, Path, Path]]:
    """
    Match reliability files to original files by metadata-value tuple.

    Parameters
    ----------
    org_index : dict[tuple, Path]
        Original transcript index.
    rel_index : dict[tuple, Path]
        Reliability transcript index.

    Returns
    -------
    list[tuple[tuple, Path, Path]]
        Tuples of (metadata_values, original_file, reliability_file).

    Notes
    -----
    Reliability files without an original counterpart are logged and skipped.
    Originals without a reliability counterpart are allowed.
    """
    matched_pairs: list[tuple[tuple, Path, Path]] = []

    for key, rel_path in rel_index.items():
        org_path = org_index.get(key)
        if org_path is None:
            logger.error(
                "No matching original transcript found for reliability file %s with metadata values %s",
                get_rel_path(rel_path),
                key,
            )
            continue
        matched_pairs.append((key, org_path, rel_path))

    matched_n = len(matched_pairs)
    total_org = len(org_index)
    coverage = (matched_n / total_org * 100) if total_org else 0.0

    logger.info(
        "Matched %d reliability pairs (%d/%d originals; %.1f%% coverage).",
        matched_n,
        matched_n,
        total_org,
        coverage,
    )

    return matched_pairs


def _save_alignment(
    metadata_values: tuple,
    transc_rel_dir: Path,
    nw: dict,
) -> None:
    """Save a global alignment text file for manual inspection."""
    alignment_filename = (
        f"{'_'.join(str(x) for x in metadata_values)}_transcription_reliability_alignment.txt"
    )
    alignment_path = transc_rel_dir / ALIGNMENTS_SUBDIR / alignment_filename

    try:
        _ensure_parent_dir(alignment_path)
        alignment_str = _format_alignment_output(
            nw["alignment"],
            nw["needleman_wunsch_score"],
            nw["needleman_wunsch_norm"],
        )
        alignment_path.write_text(alignment_str, encoding="utf-8")
        logger.info(f"Saved alignment file: {get_rel_path(alignment_path)}")
    except Exception as e:
        logger.error(f"Failed to write alignment file {get_rel_path(alignment_path)}: {e}")


def _analyze_reliability_pairs(
    matched_pairs: list[tuple[tuple, Path, Path]],
    metadata_fields,
    transc_rel_dir: Path,
    exclude_participants: list[str],
    strip_clan: bool,
    prefer_correction: bool,
    lowercase: bool,
    input_dir: Path | None = None,
) -> list[dict]:
    """
    Compute transcription reliability metrics for already-matched transcript pairs.

    Parameters
    ----------
    matched_pairs : list[tuple[tuple, Path, Path]]
        Tuples of (metadata_values, original_file, reliability_file).
    metadata_fields : dict
        MetadataField objects used to populate metadata columns.
    transc_rel_dir : Path
        Output directory for reliability evaluation artifacts.
    exclude_participants : list[str]
        Participant codes to exclude.
    strip_clan : bool
        Whether to remove CLAN markup.
    prefer_correction : bool
        Whether to prefer corrected forms.
    lowercase : bool
        Whether to lowercase transcript text before comparison.

    Returns
    -------
    list[dict]
        Record dictionaries for the reliability evaluation dataframe.
    """
    records: list[dict] = []

    field_names = [field.name for field in metadata_fields.values()]

    for metadata_values, org_cha, rel_cha in tqdm(
        matched_pairs,
        desc="Analyzing reliability transcripts",
    ):
        try:
            org_chat_data = pylangacq.Reader.from_files([str(org_cha)], parallel=False)
            rel_chat_data = pylangacq.Reader.from_files([str(rel_cha)], parallel=False)

            org_text = process_utterances(
                org_chat_data,
                exclude_participants=exclude_participants,
                strip_clan=strip_clan,
                prefer_correction=prefer_correction,
                lowercase=lowercase,
            )
            rel_text = process_utterances(
                rel_chat_data,
                exclude_participants=exclude_participants,
                strip_clan=strip_clan,
                prefer_correction=prefer_correction,
                lowercase=lowercase,
            )

            simple = _compute_simple_stats(org_text, rel_text)
            lev = _levenshtein_metrics(org_text, rel_text)
            nw = _needleman_wunsch_global(org_text, rel_text)

            _save_alignment(metadata_values, transc_rel_dir, nw)
            metadata_values = _metadata_values_for(
                rel_cha,
                metadata_fields,
                input_dir=input_dir,
            )

            record = {
                **dict(zip(field_names, metadata_values)),
                "original_file": org_cha.name,
                "reliability_file": rel_cha.name,
                **simple,
                **lev,
                "needleman_wunsch_score": nw["needleman_wunsch_score"],
                "needleman_wunsch_norm": nw["needleman_wunsch_norm"],
            }
            records.append(record)

        except Exception as e:
            logger.error(
                "Failed to analyze pair %s vs %s: %s",
                get_rel_path(org_cha),
                get_rel_path(rel_cha),
                e,
            )

    return records


def _save_reliability_outputs(
    transc_rel_df: pd.DataFrame,
    transc_rel_dir: Path,
    test: bool = False,
):
    """
    Save reliability analysis outputs for the current run.

    Writes:
      - one Excel file
      - one summary report
    """
    results = []

    df_path = transc_rel_dir / "transcription_reliability_evaluation.xlsx"
    report_path = transc_rel_dir / "transcription_reliability_report.txt"

    try:
        transc_rel_df.to_excel(df_path, index=False)
        logger.info(f"Saved reliability analysis DataFrame to: {get_rel_path(df_path)}")
    except Exception as e:
        logger.error(f"Failed to write DataFrame to {get_rel_path(df_path)}: {e}")

    try:
        write_reliability_report(transc_rel_df, report_path)
        logger.info(f"Saved reliability report to: {get_rel_path(report_path)}")
    except Exception as e:
        logger.error(f"Failed to write reliability report to {get_rel_path(report_path)}: {e}")

    if test:
        results.append(transc_rel_df.copy())

    return results


def evaluate_transcription_reliability(
    metadata_fields,
    input_dir,
    output_dir,
    exclude_participants=None,
    strip_clan=True,
    prefer_correction=True,
    lowercase=True,
    reliability_tag: str = "_reliability",
    reliability_dirname: str = "reliability",
    test=False,
):
    """
    Analyze transcription reliability by comparing original and reliability CHAT files.

    Reliability files may be identified in two ways:
      1. by filename tag (e.g., '_reliability')
      2. by residing in a designated reliability directory (e.g., 'reliability'),
         in which case renamed copies are created with the reliability tag appended
         so they can be matched to originals by metadata fields.
    """
    exclude_participants = exclude_participants or []
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    transc_rel_dir = output_dir / "transcription_reliability_evaluation"
    transc_rel_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created directory: {get_rel_path(transc_rel_dir)}")

    converted = _convert_cha_names(
        input_dir,
        reliability_tag=reliability_tag,
        reliability_dirname=reliability_dirname,
    )
    original_reliability_files = set(converted["originals"])

    cha_files = [
        p for p in input_dir.rglob("*.cha")
        if p not in original_reliability_files
    ]
    logger.info(f"Found {len(cha_files)} .cha files in the input directory.")

    tag_pattern = re.compile(re.escape(reliability_tag), re.IGNORECASE)

    rel_chats = [p for p in cha_files if tag_pattern.search(p.stem)]
    org_chats = [p for p in cha_files if not tag_pattern.search(p.stem)]

    logger.info(
        "Detected %d original and %d reliability CHAT files.",
        len(org_chats),
        len(rel_chats),
    )

    org_index = _build_file_index(
        org_chats,
        metadata_fields,
        label="original",
        input_dir=input_dir,
    )
    rel_index = _build_file_index(
        rel_chats,
        metadata_fields,
        label="reliability",
        input_dir=input_dir,
    )

    matched_pairs = _match_reliability_pairs(org_index, rel_index)

    if not matched_pairs:
        logger.warning("No transcription reliability pairs matched.")
        return [] if test else None

    records = _analyze_reliability_pairs(
        matched_pairs=matched_pairs,
        metadata_fields=metadata_fields,
        transc_rel_dir=transc_rel_dir,
        exclude_participants=exclude_participants,
        strip_clan=strip_clan,
        prefer_correction=prefer_correction,
        lowercase=lowercase,
        input_dir=input_dir,
    )

    if not records:
        logger.warning("No transcription reliability records produced.")
        return [] if test else None

    transc_rel_df = pd.DataFrame.from_records(records)
    results = _save_reliability_outputs(
        transc_rel_df=transc_rel_df,
        transc_rel_dir=transc_rel_dir,
        test=test,
    )

    return results if test else None
