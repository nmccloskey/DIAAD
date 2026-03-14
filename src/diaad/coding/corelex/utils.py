import re
import numpy as np
import contractions
import pandas as pd
import num2words as n2w
from pathlib import Path
from scipy.stats import percentileofscore

from diaad.utils.logger import logger, _rel
from diaad.coding.utils import UNINTELLIGIBLE, resolve_stim_cols
from diaad.utils.auxiliary import find_matching_files, extract_transcript_data
from diaad.coding.corelex.supp import urls, scene_tokens
from diaad.coding.corelex.supp import urls, scene_tokens, lemma_dict


def generate_token_columns(present_narratives):
    """
    Build column names for surface-form lists per narrative × lemma.

    Parameters
    ----------
    present_narratives : iterable[str]
        Names of narratives appearing in the dataset.

    Returns
    -------
    list[str]
        Column names like ["San_eat", "Bro_throw", ...].
    """
    token_cols = [
        f"{scene[:3]}_{token}"
        for scene in present_narratives
        for token in scene_tokens.get(scene, [])
    ]
    return token_cols


def _col(df, candidates):
    """
    Return the first matching column from a list of possible variants.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose columns to inspect.
    candidates : list[str]
        Possible column name variants.

    Returns
    -------
    str | None
        The first existing column name, or None if none found.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def find_corelex_inputs(input_dir: str, output_dir: str) -> dict | None:
    """
    Locate valid CoreLex input files in priority order and load them.

    Search order:
      1. *unblind_utterance_data*.xlsx — preferred single summary file.
      2. *transcript_tables*.xlsx — fallback (may be multiple; concatenated).
      Optionally merges *speaking_times*.xlsx when available.

    Parameters
    ----------
    input_dir, output_dir : str | Path
        Directories to search for candidate files.

    Returns
    -------
    tuple[str, pd.DataFrame]
        ("unblind", utt_df) or ("transcripts", utt_df)
        Returns None if no usable file is found.
    """
    search_dirs = [Path(output_dir), Path(input_dir)]

    # 1) Preferred: unblind summary
    unblind_matches = []
    for d in search_dirs:
        unblind_matches += list(d.rglob("*unblind_utterance_data*.xlsx"))
    if unblind_matches:
        p = unblind_matches[0]
        df = read_excel_safely(p)
        if df is not None:
            logger.info(f"Using unblind utterance data: {_rel(p)}")
            return "unblind", df

    # 2) Fallback: transcript tables
    transcript_tables = find_matching_files(directories=[input_dir, output_dir],
                                                search_base="transcript_tables")
    if not transcript_tables:
        logger.error(
            "No CoreLex input files found (*unblind_utterance_data*.xlsx "
            "or *transcript_tables*.xlsx)."
        )
        return None, None

    utt_frames = [extract_transcript_data(tt) for tt in transcript_tables if tt]
    if not utt_frames:
        logger.error("Transcript tables found but none could be loaded.")
        return None, None

    utt_df = pd.concat(utt_frames, ignore_index=True, sort=False)
    logger.info(f"Loaded and concatenated {len(transcript_tables)} transcript table(s).")
    return "transcripts", utt_df


def prepare_corelex_inputs(input_dir, output_dir, exclude_participants, narrative_field):
    """
    Load and normalize utterance-level CoreLex inputs.

    Determines whether input is unblind or transcript-table mode,
    filters by valid narratives, excludes unwanted speakers, and
    normalizes column names for downstream analysis.

    Parameters
    ----------
    input_dir, output_dir : Path
        Source and destination directories.
    exclude_participants : set[str]
        Speaker IDs to exclude (e.g., {"INV"}).
    narrative_field : str
        Column/tier containing stimulus (entries would be BrokenWindow, RefusedUmbrella, etc.)
    Returns
    -------
    tuple[pd.DataFrame, set[str]]
        Normalized utterance DataFrame and the set of valid narratives.
        Returns (None, None) if loading fails.
    """
    try:
        mode, utt_df = find_corelex_inputs(input_dir, output_dir)
        if utt_df is None:
            return None, None
        
        stim_cols = resolve_stim_cols(narrative_field)

        if mode == "unblind":
            narr_col = _col(utt_df, stim_cols)
            utt_df = utt_df[utt_df[narr_col].isin(urls.keys())]
            cu_col = next((c for c in utt_df.columns if c.startswith("c2_cu")), None)
            wc_col = "word_count" if "word_count" in utt_df.columns else None
            filter_col = cu_col or wc_col
            if filter_col:
                utt_df = utt_df[~np.isnan(utt_df[filter_col])]
            else:
                logger.warning("No c2_cu or word_count column; continuing unfiltered.")
            present_narratives = set(utt_df[narr_col].dropna().unique())

        else:  # transcript table mode
            narr_col = _col(utt_df, stim_cols)
            utt_col = _col(utt_df, ["utterance", "text", "tokens"])
            time_col = _col(
                utt_df,
                [
                    "speaking_time",
                    "client_time",
                    "speech_time",
                    "time_s",
                    "time_sec",
                    "time_seconds",
                ],
            )

            if not all([narr_col, utt_col, time_col]):
                logger.error("Required columns missing in transcript table input.")
                return None, None

            utt_df = utt_df[utt_df[narr_col].isin(urls.keys())]
            if "speaker" in utt_df.columns and exclude_participants:
                utt_df = utt_df[~utt_df["speaker"].isin(exclude_participants)]

            present_narratives = set(utt_df[narr_col].dropna().unique())
            utt_df = utt_df.rename(columns={narr_col: "narrative"})
            if utt_col != "utterance":
                utt_df = utt_df.rename(columns={utt_col: "utterance"})
            if time_col and time_col != "speaking_time":
                utt_df = utt_df.rename(columns={time_col: "speaking_time"})

        return utt_df, present_narratives

    except Exception as e:
        logger.error(f"Failed to prepare CoreLex inputs: {e}")
        return None, None


def reformat(text: str) -> str:
    """
    Prepares a transcription text string for CoreLex analysis.

    - Expands contractions (keeps possessive 's / ’s).
    - Converts digits to words.
    - Preserves replacements like '[: dogs] [*]' → 'dogs'.
    - Removes other CHAT/CLAN annotations (repetitions, comments, gestures, events).
    - Removes tokens that START with punctuation (e.g., &=draws:a:cat), except standalone "'s"/"’s".
    """
    try:
        text = text.lower().strip()

        # 1) Handle specific pattern: "(he|it)'s got" → "he has got" / "it has got"
        text = re.sub(r"\b(he|it)'s got\b", r"\1 has got", text)

        # 2) Expand contractions while keeping possessive 's approximately
        tokens = text.split()
        expanded = []
        for tok in tokens:
            # If looks like possessive 's or ’s, keep as-is (don't expand to "is")
            if re.fullmatch(r"\w+'s", tok) or re.fullmatch(r"\w+’s", tok):
                expanded.append(tok)
            else:
                expanded.append(contractions.fix(tok))
        text = " ".join(expanded)

        # 3) Convert standalone digits to words
        text = re.sub(r"\b\d+\b", lambda m: n2w.num2words(int(m.group())), text)

        # 4) Preserve accepted clinician replacement: "[: dogs] [*]" → "dogs"
        text = re.sub(r'\[:\s*([^\]]+?)\s*\]\s*\[\*\]', r'\1', text)

        # 5) Remove ALL other square-bracketed content (e.g., [//], [?], [% ...], [& ...])
        text = re.sub(r'\[[^\]]+\]', ' ', text)

        # 6) Remove other common CLAN containers: <...> events, ((...)) comments, {...} paralinguistic
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\(\([^)]*\)\)', ' ', text)
        text = re.sub(r'\{[^}]+\}', ' ', text)

        # 7) Remove tokens that START with punctuation (gesture/dep. tiers), e.g. &=draws:a:cat, +/., =laughs
        #    Keep the standalone possessive token "'s"/"’s" if it appears.
        #    (?<!\S)  -> start of token (preceded by start or whitespace)
        #    (?!'s\b)(?!’s\b)  -> DO NOT match if the token is exactly 's/’s
        #    [^\w\s']\S*  -> a non-word, non-space, non-apostrophe first char, then the rest of the token
        text = re.sub(r"(?<!\S)(?!'s\b)(?!’s\b)[^\w\s']\S*", ' ', text)

        # 8) Remove non-word characters except apostrophes (keeps possessives like cinderella’s)
        text = re.sub(r"[^\w\s']", ' ', text)

        # 9) Token-level cleanup: drop CHAT placeholders like 'xxx', 'yyy', 'www'
        toks = [t for t in text.split() if t not in UNINTELLIGIBLE]

        # 10) Collapse whitespace and return
        return " ".join(toks).strip()

    except Exception as e:
        logger.error(f"An error occurred while reformatting: {e}")
        return ""


def id_core_words(scene_name: str, reformatted_text: str) -> dict:
    """
    Identifies and quantifies core words in a narrative sample.

    Args:
        scene_name (str): The narrative scene name.
        reformatted_text (str): Preprocessed transcript text.

    Returns:
        dict: {
            "num_tokens": int,
            "num_core_words": int,
            "num_cw_tokens": int,
            "lexicon_coverage": float,
            "token_sets": dict[str, set[str]]
        }
    """
    tokens = reformatted_text.split()
    token_sets = {}
    num_cw_tokens = 0

    for token in tokens:
        lemma = lemma_dict.get(token, token)

        if lemma in scene_tokens.get(scene_name, []):
            num_cw_tokens += 1
            if lemma in token_sets:
                token_sets[lemma].add(token)
            else:
                token_sets[lemma] = {token}

    if scene_name.lower() == "cinderella" and "'s" in tokens:
        token_sets["'s"] = {"'s"}
        num_cw_tokens += 1

    num_tokens = len(tokens)
    num_core_words = len(token_sets)
    total_lexicon_size = len(scene_tokens.get(scene_name, []))
    lexicon_coverage = num_core_words / total_lexicon_size if total_lexicon_size > 0 else 0.0

    return {
        "num_tokens": num_tokens,
        "num_core_words": num_core_words,
        "num_cw_tokens": num_cw_tokens,
        "lexicon_coverage": lexicon_coverage,
        "token_sets": token_sets
    }


def load_corelex_norms_online(stimulus_name: str, metric: str = "accuracy") -> pd.DataFrame:
    try:
        url = urls[stimulus_name][metric]
        return pd.read_csv(url)
    except KeyError:
        raise ValueError(f"Unknown stimulus '{stimulus_name}' or metric '{metric}'")
    except Exception as e:
        raise RuntimeError(f"Failed to load data from URL: {e}")


def preload_corelex_norms(present_narratives: set) -> dict:
    """
    Preloads accuracy and efficiency CoreLex norms for all narratives in current batch of samples.

    Args:
        present_narratives (set): Set of narratives present in the input batch.

    Returns:
        dict: Dictionary of dictionaries {scene_name: {accuracy: df, efficiency: df}}
    """
    norm_data = {}

    for scene in present_narratives:
        try:
            norm_data[scene] = {
                "accuracy": load_corelex_norms_online(scene, "accuracy"),
                "efficiency": load_corelex_norms_online(scene, "efficiency")
            }
            logger.info(f"Loaded CoreLex norms for: {scene}")
        except Exception as e:
            logger.warning(f"Failed to load norms for {scene}: {e}")
            norm_data[scene] = {"accuracy": None, "efficiency": None}

    return norm_data


def get_percentiles(score: float, norm_df: pd.DataFrame, column: str) -> dict:
    """
    Computes percentile rank of a score relative to both control and PWA distributions.

    Args:
        score (float): The participant's score.
        norm_df (pd.DataFrame): DataFrame with 'Aphasia' and score column.
        column (str): Name of the column containing scores (e.g., 'CoreLex Score', 'CoreLex/min').

    Returns:
        dict: {
            "control_percentile": float,
            "pwa_percentile": float
        }
    """
    control_scores = norm_df[norm_df['Aphasia'] == 0][column]
    pwa_scores = norm_df[norm_df['Aphasia'] == 1][column]

    return {
        "control_percentile": percentileofscore(control_scores, score, kind="weak"),
        "pwa_percentile": percentileofscore(pwa_scores, score, kind="weak")
    }


def read_excel_safely(path):
    try:
        return pd.read_excel(path)
    except Exception as e:
        logger.warning(f"Failed reading {_rel(path)}: {e}")
        return None
