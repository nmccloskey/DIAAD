import re
import spacy
import random
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from rascal.utterances.make_coding_files import segment, assign_coders
from rascal.transcription.transcription_reliability_analysis import _clean_clan_for_reliability


POWERS_cols = [
    "id", "turn_type", "speech_units", "content_words", "num_nouns", "filled_pauses", "collab_repair", "POWERS_comment"
]
coder_cols = [f"c{n}_{col}" for n in ["1", "2"] for col in POWERS_cols]

COMM_cols = [
    "communication", "topic", "subject", "dialogue", "conversation"
]

CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}

GENERIC_TERMS = {"stuff", "thing", "things", "something", "anything", "everything", "nothing"}

# count speech units after cleaning
def compute_speech_units(utt):
    cleaned = _clean_clan_for_reliability(utt)
    tokens = cleaned.split()
    su = sum(tok.lower() not in {"xx","xxx","yy","yyy"} for tok in tokens)
    return su

FILLER_PATTERN = re.compile(
    r"(?<!\w)(?:&-?)?(?:um+|uh+|erm+|er+|eh+)(?!\w)",
    re.IGNORECASE
)

# Count filled pauses Without cleaning
def count_fillers(utt: str) -> int:
    return len(FILLER_PATTERN.findall(utt))

# --- NLP model singleton (your version, trimmed to essentials here) ---
class NLPmodel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._nlp_models = {}
            cls._instance.load_nlp()
        return cls._instance

    def load_nlp(self, model_name="en_core_web_trf"):
        if model_name not in self._nlp_models:
            self._nlp_models[model_name] = spacy.load(model_name)

    def get_nlp(self, model_name="en_core_web_trf"):
        if model_name not in self._nlp_models:
            self.load_nlp(model_name)
        return self._nlp_models[model_name]

# ---------- Rule helpers ----------
def is_generic(token) -> bool:
    return token.text.lower() in GENERIC_TERMS

def is_aux_or_modal(token) -> bool:
    """
    True for auxiliaries and modals you want to EXCLUDE.
    - SpaCy marks helping verbs as AUX (be/have/do/will/shall, etc.).
    - Modals have PTB tag 'MD'.
    """
    if token.pos_ != "AUX":
        return False
    # If it's AUX, always exclude for your rule set
    return True  # (covers modals + non-modal auxiliaries)

def is_ly_adverb(token) -> bool:
    # Only count adverbs that end with -ly
    return token.pos_ == "ADV" and token.text.lower().endswith("ly")

def is_numeral(token) -> bool:
    # Count numerals; SpaCy may set pos_==NUM, tag_==CD, and/or like_num==True
    return token.pos_ == "NUM" or token.tag_ == "CD" or token.like_num

def is_main_verb(token) -> bool:
    # Count ONLY main verbs (VERB); exclude AUX (handled separately)
    return token.pos_ == "VERB"

def is_noun_or_propn(token) -> bool:
    return token.pos_ in {"NOUN", "PROPN"}

def is_adjective(token) -> bool:
    return token.pos_ == "ADJ"

def is_content_token(token) -> bool:
    """
    Master predicate implementing your rules:
    - Include: NOUN, PROPN, VERB (main only), ADJ, ADV(-ly only), NUM
    - Exclude: AUX (including modals), generic terms
    """
    if is_generic(token):
        return False
    if is_aux_or_modal(token):
        return False

    if is_noun_or_propn(token):
        return True
    if is_main_verb(token):
        return True
    if is_adjective(token):
        return True
    if is_ly_adverb(token):
        return True
    if is_numeral(token):
        return True

    return False

# ---------- Core counting function ----------
def count_content_words_from_doc(doc, count_type="all"):
    """
    Count content words from a spaCy Doc object.
    """
    total = total_nouns = 0
    for tok in doc:
        if is_content_token(tok):
            total += 1
            if tok.pos_ in ("NOUN", "PROPN"):
                total_nouns += 1
    return total if count_type == "all" else total_nouns

minimal_turns = ["I know", "I don't know", "I see", "alright", "oh dear", "okay", "mm"]

minimal_turns = [
    r"\bi know\b",
    r"\bi don't know\b",
    r"\bi see\b",
    r"\balright\b",
    r"\boh dear\b",
    r"\bokay\b",
    r"\bmm+\b",           # catches "mm", "mmm"
    r"\byeah\b",
    r"\bno\b",
    r"\bmaybe\b",
    # combos
    r"\balright(,?\s*i see)?(,?\s*i don't know)?",
    r"\bi don't know(,?\s*maybe)?",
]

def label_turn(utterance: str, count_content_words: int) -> str:
    """
    Label turns:
      - "MT": minimal turn (from minimal_turns list)
      - "ST": substantial turn (has content words)
      - "T" : subminimal turn (no content words, not minimal)
    """
    utt = utterance.strip().lower()
    for pat in minimal_turns:
        if re.match(pat, utt, flags=re.IGNORECASE):
            return "MT"
    if count_content_words > 0:
        return "ST"
    return "T"

def run_automation(df, coder_num):
    """
    Apply automated linguistic measures to a POWERS coding dataframe.

    Loads a spaCy transformer pipeline (en_core_web_trf) and applies:
      - Speech unit counts
      - Filled pause counts
      - Content word counts (all and nouns)
      - Turn type labeling

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least an "utterance" column.
    coder_num : str or int
        Coder identifier (e.g., "1", "2", "3"). Used to prefix new columns.

    Returns
    -------
    pandas.DataFrame
        Input dataframe with added columns:
        - c{coder_num}_speech_units
        - c{coder_num}_filled_pauses
        - c{coder_num}_content_words
        - c{coder_num}_num_nouns
        - c{coder_num}_turn_type
    """

    try:
        NLP = NLPmodel()
        nlp = NLP.get_nlp("en_core_web_trf")
    except Exception as e:
        logging.error(f"Failed to load NLP model - automation not available: {e}")
        return df
    
    try:
        df[f"c{coder_num}_speech_units"] = df["utterance"].apply(compute_speech_units)
        df[f"c{coder_num}_filled_pauses"] = df["utterance"].apply(count_fillers)

        content_counts, noun_counts, turn_types = [], [], []
        utterances = df["utterance"].fillna("").map(_clean_clan_for_reliability)

        total_its = len(utterances)
        for doc, utt in tqdm(zip(nlp.pipe(utterances, batch_size=100, n_process=2), utterances), total=total_its, desc="Applying automation to utterances"):
            count_content_words = count_content_words_from_doc(doc, "all")
            content_counts.append(count_content_words)
            noun_counts.append(count_content_words_from_doc(doc, "noun"))
            turn_types.append(label_turn(utt, count_content_words))
        df[f"c{coder_num}_content_words"] = content_counts
        df[f"c{coder_num}_num_nouns"] = noun_counts
        df[f"c{coder_num}_turn_type"] = turn_types
        return df
    
    except Exception as e:
        logging.error(f"Failed to apply automation: {e}")
        return df

def make_POWERS_coding_files(tiers, frac, coders, input_dir, output_dir, exclude_participants, automate_POWERS=True):
    """
    Generate POWERS coding and reliability files from utterance-level transcripts.

    Steps:
      1. Load transcript "Utterances.xlsx" files from input_dir/output_dir.
      2. Assign two coders per sample and shuffle sample order.
      3. Write a POWERS coding file with initialized coder columns.
      4. Select a fraction of samples for a reliability coder, producing a
         POWERS reliability coding file.
      5. Optionally run automated speech measures via run_automation.

    Parameters
    ----------
    tiers : dict
        Mapping of tier patterns to regex objects for file labeling.
    frac : float
        Proportion of samples to assign for reliability coding (0-1).
    coders : list of str
        List of coder IDs. If fewer than 3 provided, defaults to ['1','2','3'].
    input_dir : str or Path
        Directory containing input Utterances.xlsx files.
    output_dir : str or Path
        Base directory for POWERS_Coding output.
    exclude_participants : list
        Speakers to exclude (filled with "NA").
    automate_POWERS : bool, optional
        If True, apply run_automation() to coder 1 columns.

    Returns
    -------
    None
        Writes Excel files to output_dir/POWERS_Coding.
    """

    if len(coders) < 3:
        logging.warning(f"Coders entered: {coders} do not meet minimum of 3. Using default 1, 2, 3.")
        coders = ['1', '2', '3']

    output_dir = Path(output_dir)
    POWERS_coding_dir = output_dir / "POWERS_Coding"
    logging.info(f"Writing POWERS coding files to {POWERS_coding_dir}")
    utterance_files = list(Path(input_dir).rglob("*Utterances*.xlsx")) + list(Path(output_dir).rglob("*Utterances*.xlsx"))

    for file in tqdm(utterance_files, desc="Generating POWERS coding files"):
        logging.info(f"Processing file: {file}")
        labels = [t.match(file.name, return_None=True) for t in tiers.values()]
        labels = [l for l in labels if l is not None]

        assignments = assign_coders(coders)

        try:
            uttdf = pd.read_excel(str(file))
            logging.info(f"Successfully read file: {file}")
        except Exception as e:
            logging.error(f"Failed to read file {file}: {e}")
            continue

        # Shuffle samples
        subdfs = []
        for _, subdf in uttdf.groupby(by="sample_id"):
            subdfs.append(subdf)
        random.shuffle(subdfs)
        shuffled_utt_df = pd.concat(subdfs, ignore_index=True)

        PCdf = shuffled_utt_df.drop(columns=[
            col for col in ['file'] + [t for t in tiers if t not in COMM_cols] if col in shuffled_utt_df.columns
            ]).copy()
        
        PCdf["c1_id"] = pd.Series(dtype="object")
        PCdf["c2_id"] = pd.Series(dtype="object")

        for col in coder_cols:
            PCdf[col] = np.where(PCdf["speaker"].isin(exclude_participants), "NA", np.nan)
        
        if automate_POWERS:
            PCdf = run_automation(PCdf, "1")

        unique_sample_ids = list(PCdf['sample_id'].drop_duplicates(keep='first'))
        segments = segment(unique_sample_ids, n=len(coders))
        rel_subsets = []

        for seg, ass in zip(segments, assignments):
            PCdf.loc[PCdf['sample_id'].isin(seg), 'c1_id'] = ass[0]
            PCdf.loc[PCdf['sample_id'].isin(seg), 'c2_id'] = ass[1]

            rel_samples = random.sample(seg, k=max(1, round(len(seg) * frac)))
            relsegdf = PCdf[PCdf['sample_id'].isin(rel_samples)].copy()

            rel_subsets.append(relsegdf)

        reldf = pd.concat(rel_subsets)

        rel_drop_cols = [col for col in coder_cols if col.startswith("c2")]
        reldf.drop(columns=rel_drop_cols, inplace=True, errors='ignore')
        
        rename_map = {col:col.replace("1", "3") for col in coder_cols if col.startswith("c1")}
        reldf.rename(columns=rename_map, inplace=True)
        
        logging.info(f"Selected {len(set(reldf['sample_id']))} samples for reliability from {len(set(PCdf['sample_id']))} total samples.")

        lab_str = '_'.join(labels) + '_' if labels else ''

        pc_filename = Path(POWERS_coding_dir, *labels, f"{lab_str}POWERS_Coding.xlsx")
        rel_filename = Path(POWERS_coding_dir, *labels, f"{lab_str}POWERS_ReliabilityCoding.xlsx")

        try:
            pc_filename.parent.mkdir(parents=True, exist_ok=True)
            PCdf.to_excel(pc_filename, index=False)
            logging.info(f"Successfully wrote POWERS coding file: {pc_filename}")
        except Exception as e:
            logging.error(f"Failed to write POWERS coding file {pc_filename}: {e}")

        try:
            rel_filename.parent.mkdir(parents=True, exist_ok=True)
            reldf.to_excel(rel_filename, index=False)
            logging.info(f"Successfully wrote POWERS reliability coding file: {rel_filename}")
        except Exception as e:
            logging.error(f"Failed to write POWERS reliability coding file {rel_filename}: {e}")


def reselect_POWERS_reliability(input_dir, output_dir, frac, exclude_participants, automate_POWERS):
    """
    Reselect new reliability subsets from existing POWERS coding files.

    Finds POWERS_Coding and POWERS_ReliabilityCoding files, determines
    which samples are already covered by reliability coders, and selects
    new samples from the remaining pool. Optionally applies automation.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing original POWERS_Coding and POWERS_ReliabilityCoding files.
    output_dir : str or Path
        Directory for saving POWERS_ReselectedReliability outputs.
    frac : float
        Fraction of samples per file to assign to reliability (0-1).
    exclude_participants : list
        Speakers to exclude (filled with "NA").
    automate_POWERS : bool
        If True, apply run_automation() to coder 3 columns.

    Returns
    -------
    None
        Writes new Excel reliability files to output_dir/POWERS_ReselectedReliability.
    """

    output_dir = Path(output_dir)
    
    POWERS_Reselected_Reliability_dir = output_dir / "POWERS_ReselectedReliability"
    try:
        POWERS_Reselected_Reliability_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {POWERS_Reselected_Reliability_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory {POWERS_Reselected_Reliability_dir}: {e}")
        return

    coding_files = [f for f in Path(input_dir).rglob('*POWERS_Coding*.xlsx')]
    rel_files = [f for f in Path(input_dir).rglob('*POWERS_ReliabilityCoding*.xlsx')]

    # Match original coding and reliability files.
    for cod in tqdm(coding_files, desc="Analyzing POWERS reliability coding..."):
        try:
            covered_sample_ids = set()
            PCcod = pd.read_excel(cod)
            logging.info(f"Processing coding file: {cod}")
        except Exception as e:
            logging.error(f"Failed to read file {cod}: {e}")
            continue
        for rel in rel_files:
            if rel.name.replace("POWERS_Coding", "POWERS_ReliabilityCoding") == cod.name:
                try:
                    PCrel = pd.read_excel(rel)
                    logging.info(f"Processing reliability file: {rel}")
                except Exception as e:
                    logging.error(f"Failed to read file {rel}: {e}")
                    continue
                
            covered_sample_ids.update(set(PCrel["sample_id"].dropna()))
        
        if covered_sample_ids:
            all_samples = set(PCcod["sample_id"].dropna())
            available_samples = list(all_samples - covered_sample_ids)

            if len(available_samples) == 0:
                logging.warning(f"No available samples to reselect for {cod.name}. Skipping.")
                continue
            
            num_to_select = max(1, round(len(all_samples) * float(frac)))
            if len(available_samples) < num_to_select:
                logging.warning(
                    f"Not enough unused samples in {cod.name}. "
                    f"Selecting {len(available_samples)} instead of target {num_to_select}."
                )
                num_to_select = len(available_samples)
            
            reselected_rel_samples = set(random.sample(available_samples, k=num_to_select))
            new_rel_df = PCcod[PCcod['sample_id'].isin(reselected_rel_samples)].copy()

            for col in coder_cols:
                new_rel_df[col] = np.where(new_rel_df["speaker"].isin(exclude_participants), "NA", np.nan)

            rel_drop_cols = [col for col in coder_cols if col.startswith("c2")]
            new_rel_df.drop(columns=rel_drop_cols, inplace=True, errors='ignore')
            
            rename_map = {col:col.replace("1", "3") for col in coder_cols if col.startswith("c1")}
            new_rel_df.rename(columns=rename_map, inplace=True)
            
            logging.info(f"Reselected {len(set(new_rel_df['sample_id']))} samples for reliability from {len(set(PCcod['sample_id']))} total samples.")

            if automate_POWERS:
                new_rel_df = run_automation(new_rel_df, "3")

            try:
                new_rel_filename = cod.name.replace("POWERS_Coding", "POWERS_Reselected_ReliabilityCoding")
                new_rel_filepath = POWERS_Reselected_Reliability_dir / new_rel_filename
                new_rel_df.to_excel(new_rel_filepath, index=False)
                logging.info(f"Successfully wrote reselected POWERS reliability coding file: {new_rel_filepath}")
            except Exception as e:
                logging.error(f"Failed to write reselected POWERS reliability coding file {new_rel_filepath}: {e}")
