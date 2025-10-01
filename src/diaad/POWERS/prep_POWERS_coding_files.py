import os
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


def make_POWERS_coding_files(tiers, frac, coders, input_dir, output_dir,exclude_participants, automate_POWERS=True):
    """
    Generate POWERS coding and reliability files from utterance-level transcript tables.

    This function takes transcript-derived "Utterances.xlsx" files, assigns coders to
    samples, and produces two types of outputs per file: (1) a POWERS coding file with
    two primary coder columns initialized, and (2) a reliability coding file where a
    fraction of samples are randomly selected and assigned to a third coder. Files are
    partitioned and labeled according to tier matches, shuffled to randomize sample order,
    and written to a structured `POWERS_Coding` subdirectory under the output directory.
    Excluded participants receive "NA" values in coder columns.
    """

    if len(coders) < 3:
        logging.warning(f"Coders entered: {coders} do not meet minimum of 3. Using default 1, 2, 3.")
        coders = ['1', '2', '3']

    POWERS_coding_dir = os.path.join(output_dir, 'POWERS_Coding')
    logging.info(f"Writing POWERS coding files to {POWERS_coding_dir}")
    utterance_files = list(Path(input_dir).rglob("*Utterances*.xlsx")) + list(Path(output_dir).rglob("*Utterances*.xlsx"))

    if automate_POWERS:
        try:
            NLP = NLPmodel()
            nlp = NLP.get_nlp("en_core_web_trf")
        except Exception as e:
            logging.error(f"Failed to load NLP model - automation not available: {e}")
            return

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

        coder_cols = [f"c{n}_{col}" for n in ["1", "2"] for col in POWERS_cols]
        for col in coder_cols:
            PCdf[col] = np.where(PCdf["speaker"].isin(exclude_participants), "NA", np.nan)
        
        if automate_POWERS:
            PCdf["c1_speech_units"] = PCdf["utterance"].apply(compute_speech_units)
            PCdf["c1_filled_pauses"] = PCdf["utterance"].apply(count_fillers)

            content_counts, noun_counts, turn_types = [], [], []
            utterances = PCdf["utterance"].fillna("").map(_clean_clan_for_reliability)
            for doc, utt in zip(nlp.pipe(utterances, batch_size=100, n_process=2), utterances):
                count_content_words = count_content_words_from_doc(doc, "all")
                content_counts.append(count_content_words)
                noun_counts.append(count_content_words_from_doc(doc, "noun"))
                turn_types.append(label_turn(utt, count_content_words))
            PCdf["c1_content_words"] = content_counts
            PCdf["c1_num_nouns"] = noun_counts
            PCdf["c1_turn_type"] = turn_types

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

        pc_filename = os.path.join(POWERS_coding_dir, *labels, lab_str + 'POWERS_Coding.xlsx')
        rel_filename = os.path.join(POWERS_coding_dir, *labels, lab_str + 'POWERS_ReliabilityCoding.xlsx')

        try:
            os.makedirs(os.path.dirname(pc_filename), exist_ok=True)
            PCdf.to_excel(pc_filename, index=False)
            logging.info(f"Successfully wrote POWERS coding file: {pc_filename}")
        except Exception as e:
            logging.error(f"Failed to write POWERS coding file {pc_filename}: {e}")

        try:
            os.makedirs(os.path.dirname(rel_filename), exist_ok=True)
            reldf.to_excel(rel_filename, index=False)
            logging.info(f"Successfully wrote POWERS reliability coding file: {rel_filename}")
        except Exception as e:
            logging.error(f"Failed to write POWERS reliability coding file {rel_filename}: {e}")

def reselect_POWERS_reliability(input_dir, output_dir, frac):

    output_dir = Path(output_dir)
    
    POWERS_Reselected_Reliability_dir = output_dir / "POWERS_ReselectedReliability"
    try:
        os.makedirs(POWERS_Reselected_Reliability_dir, exist_ok=True)
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
            rel_samples = random.sample(available_samples, k=max(1, round(len(PCcod) * frac)))
            new_rel_df = PCcod[PCcod['sample_id'].isin(rel_samples)].copy()

            try:
                new_rel_filename = cod.name.replace("POWERS_Coding", "POWERS_Reselected_ReliabilityCoding")
                new_rel_filepath = POWERS_Reselected_Reliability_dir / new_rel_filename
                os.makedirs(new_rel_filepath, exist_ok=True)
                new_rel_df.to_excel(new_rel_filepath, index=False)
                logging.info(f"Successfully wrote reselected POWERS reliability coding file: {new_rel_filepath}")
            except Exception as e:
                logging.error(f"Failed to write reselected POWERS reliability coding file {new_rel_filepath}: {e}")
