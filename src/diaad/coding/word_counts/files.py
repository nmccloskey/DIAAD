import re
import random
import contractions
import pandas as pd
from tqdm import tqdm
import num2words as n2w
from pathlib import Path
from functools import lru_cache

from diaad.utils.logger import logger, _rel
from diaad.utils.auxiliary import calc_subset_size
from diaad.coding.utils import segment, assign_coders


@lru_cache(maxsize=1)
def get_word_checker():
    import nltk

    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')

    from nltk.corpus import words
    valid_words = set(words.words())
    return lambda word: word in valid_words


def count_words(text, d):
    """
    Prepares a transcription text string for counting words.
    
    Parameters:
        text (str): Input transcription text.
        d (function): A function or callable to check if a word exists in the dictionary.
        
    Returns:
        int: Count of valid words.
    """
    # Normalize text
    text = text.lower().strip()
    
    # Handle specific contractions and patterns
    text = re.sub(r"(?<=(he|it))'s got", ' has got', text)
    text = ' '.join([contractions.fix(w) for w in text.split()])
    text = text.replace(u'\xa0', '')
    text = re.sub(r'(^|\b)(u|e)+?(h|m|r)+?(\b|$)', '', text)
    text = re.sub(r'(^|\b|\b.)x+(\b|$)', '', text)
    
    # Remove annotations and special markers
    text = re.sub(r'\[.+?\]', '', text)
    text = re.sub(r'\*.+?\*', '', text)
    
    # Convert numbers to words
    text = re.sub(r'\d+', lambda x: n2w.num2words(int(x.group(0))), text)
    
    # Remove non-word characters and clean up spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\bcl\b', '', text)
    text = re.sub(r'\s{2,}', ' ', text).strip()
    
    # Tokenize and validate words
    tokens = [word for word in text.split() if d(word)]
    return len(tokens)


def _read_cu_file(file: Path) -> pd.DataFrame:
    """Read and shuffle CU coding file by sample_id."""
    cu_df = pd.read_excel(file)
    subdfs = [g for _, g in cu_df.groupby("sample_id")]
    random.shuffle(subdfs)
    shuffled = pd.concat(subdfs, ignore_index=True)
    logger.info(f"Read and shuffled {file.name}")
    return shuffled

def _prepare_wc_df(df: pd.DataFrame, d) -> pd.DataFrame:
    """Add coder and word_count columns; drop CU-specific ones."""
    df = df.copy()
    df["c1_id"] = ""
    c2_cu_col = next((col for col in df.columns if col.startswith("c2_cu")))
    df["word_count"] = df.apply(
        lambda r: count_words(r["utterance"], d) if not pd.isna(r.get(c2_cu_col)) else "NA",
        axis=1
    )
    df["wc_comment"] = ""
    
    drop_cols = [c for c in df if c.startswith(("c1_sv", "c1_rel", "c1_cu", "c1_comment", "c2_sv", "c2_rel", "c2_cu", "c2_comment", "agmt"))]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    return df

def _assign_wc_coders(df: pd.DataFrame, coders: list[str], frac: float):
    """Assign coders and build reliability subset."""
    assignments = assign_coders(coders)
    unique_ids = list(df["sample_id"].drop_duplicates())
    segments = segment(unique_ids, n=len(coders))

    rel_subsets = []
    for seg, assn in zip(segments, assignments):
        df.loc[df["sample_id"].isin(seg), "c1_id"] = assn[0]

        n_rel_samples = calc_subset_size(frac=frac, samples=seg)
        rel_samples = random.sample(seg, k=n_rel_samples)

        relsegdf = df[df["sample_id"].isin(rel_samples)].copy()
        relsegdf.rename(columns={"c1_id": "c2_id", "wc_comment": "wc_rel_com"}, inplace=True)
        relsegdf["c2_id"] = assn[1]
        rel_subsets.append(relsegdf)

    wc_rel_df = pd.concat(rel_subsets)
    logger.info(f"Reliability subset: {len(wc_rel_df)} utterances")
    return df, wc_rel_df

def _write_wc_outputs(wc_df, wc_rel_df, word_count_dir, labels):
    """Write word count and reliability files to disk."""
    lab_str = "_".join(labels) + "_" if labels else ""
    out_dir = Path(word_count_dir, *labels)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        f"{lab_str}word_counting.xlsx": wc_df,
        f"{lab_str}word_count_reliability.xlsx": wc_rel_df,
    }
    for fname, df in files.items():
        fpath = out_dir / fname
        try:
            df.to_excel(fpath, index=False)
            logger.info(f"Wrote {_rel(fpath)}")
        except Exception as e:
            logger.error(f"Failed to write {_rel(fpath)}: {e}")

def make_word_count_files(tiers, frac, coders, input_dir, output_dir):
    """
    Create word-count coding and reliability workbooks from CU utterance tables.

    Each input *cu_coding_by_utterance*.xlsx produces:
      - *_word_counting.xlsx* → all samples
      - *_word_count_reliability.xlsx* → subset (~frac) for reliability

    Steps:
      1. Locate CU coding files in `input_dir` and `output_dir`.
      2. Read each file, shuffle samples, drop CU-specific columns.
      3. Compute `word_count` per utterance using `count_words`.
      4. Assign coders and sample reliability subsets.
      5. Write outputs under `{output_dir}/word_counts/<labels>/`.

    Parameters
    ----------
    tiers : dict[str, Tier]
    frac : float
    coders : list[str]
    input_dir, output_dir : Path or str
    """
    word_count_dir = Path(output_dir) / "word_counts"
    word_count_dir.mkdir(parents=True, exist_ok=True)
    d = get_word_checker()
    cu_files = list(Path(input_dir).rglob("*cu_coding_by_utterance*.xlsx")) + \
               list(Path(output_dir).rglob("*cu_coding_by_utterance*.xlsx"))

    for file in tqdm(cu_files, desc="Generating word count files"):
        try:
            wc_df = _read_cu_file(file)
            labels = [t.match(file.name, return_none=True) for t in tiers.values() if t.match(file.name, return_none=True)]
            wc_df = _prepare_wc_df(wc_df, d)
            wc_df, wc_rel_df = _assign_wc_coders(wc_df, coders, frac)
            _write_wc_outputs(wc_df, wc_rel_df, word_count_dir, labels)
        except Exception as e:
            logger.error(f"Failed processing {_rel(file)}: {e}")
