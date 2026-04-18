import re
import contractions

from tqdm import tqdm
import pandas as pd

from psair.core.logger import logger
from psair.nlp import NLPModel

from diaad.transcripts.transcription_reliability_evaluation import process_utterances


CONTENT_UPOS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
GENERIC_TERMS = {"stuff", "thing", "things", "something", "anything", "everything", "nothing"}
UNINTELLIGIBLES = {"xx", "xxx", "yy", "yyy"}


# count speech units after cleaning
def compute_speech_units(utt):
    cleaned = process_utterances(utt)
    tokens = cleaned.split()
    su = sum(tok.lower() not in UNINTELLIGIBLES for tok in tokens)
    return su

FILLER_PATTERN = re.compile(
    r"(?<!\w)(?:&-?)?(?:um+|uh+|erm+|er+|eh+)(?!\w)",
    re.IGNORECASE
)

# Count filled pauses Without cleaning
def count_fillers(utt: str) -> int:
    return len(FILLER_PATTERN.findall(utt))

# Expand contractions
def expand_contractions(utt: str) -> str:
    return contractions.fix(utt)

# Modified processing
def expand_and_process_utterances(utt: str) -> str:
    codeless_utt = " ".join([t for t in utt.split() if t and not t.startswith("&")])
    expanded_utt = expand_contractions(codeless_utt)
    modified_utt = expanded_utt.replace("-", "_")
    return process_utterances(modified_utt)

# --- NLP model ---
def get_powers_nlp(model_name: str = "en_core_web_sm"):
    return NLPModel().get_nlp(model_name=model_name)

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

def is_unintelligble(token) -> bool:
    """Exclude unintelligible speech"""
    return token.text.lower() in UNINTELLIGIBLES

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

def is_chat_code(token) -> bool:
    return token.text.startswith("&")

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
    if is_unintelligble(token):
        return False
    if is_chat_code(token):
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

def check_main_verb(tagged_utt: str, total_cw: int) -> tuple[str, int]:
    """
    Treat 'be' form as a main verb in the absence of any other VERB tag.

    Parameters
    ----------
    tagged_utt : str
        Utterance tagged with POS markers (e.g., "_VERB_CW").
    total_cw : int
        Current count of content words.

    Returns
    -------
    tuple[str, int]
        Possibly updated tagged utterance and content word count.
    """
    # Only apply if spaCy found no main verbs
    if "_VERB" not in tagged_utt:
        # Match standalone forms of 'be' (case-insensitive)
        m = re.search(r"\b(?:be|am|are|is|was|were|been|being)\b", tagged_utt, flags=re.IGNORECASE)
        if m:
            tagged_utt += "_BE_FORM_MAIN"
            total_cw += 1
    return tagged_utt, total_cw

# ---------- Core counting function ----------
def count_content_words_from_doc(doc):
    """
    Tally content words & nouns from a spaCy Doc object.
    Also tag tokens for manual review.
    """
    total_cw = total_nouns = 0
    tagged_utt = ""
    for tok in doc:
        tagged_utt += f"{tok}"
        if is_content_token(tok):
            total_cw += 1
            tagged_utt += f"_{tok.pos_}_CW"
            if tok.pos_ in ("NOUN", "PROPN"):
                total_nouns += 1
                tagged_utt += "_N"
        tagged_utt += " "
    tagged_utt, total_cw = check_main_verb(tagged_utt, total_cw)
    return total_cw, total_nouns, tagged_utt


def run_automation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Populate automated POWERS fields on an unprefixed coding dataframe.

    Adds speech-unit counts, filled-pause counts, content-word counts,
    noun counts, and a tagged utterance column when possible. If the NLP
    model fails to load or processing fails, the input dataframe is
    returned unchanged.
    """
    try:
        nlp = get_powers_nlp()
    except Exception as e:
        logger.error(f"Failed to load NLP model; POWERS automation unavailable: {e}")
        return df

    if "utterance" not in df.columns:
        logger.error("POWERS automation requires an 'utterance' column.")
        return df

    try:
        utterances = df["utterance"].fillna("").map(expand_and_process_utterances)
        df["speech_units"] = df["utterance"].apply(compute_speech_units)
        df["filled_pauses"] = df["utterance"].apply(count_fillers)

        content_counts, noun_counts, tagged_utts = _automate_content_measures(
            utterances=utterances,
            nlp=nlp,
        )

        df["content_words"] = content_counts
        df["num_nouns"] = noun_counts

        if "tagged_utterance" in df.columns:
            df["tagged_utterance"] = tagged_utts
        else:
            utt_idx = df.columns.get_loc("utterance")
            df.insert(utt_idx + 1, "tagged_utterance", tagged_utts)

        return df

    except Exception as e:
        logger.error(f"Failed to apply POWERS automation: {e}")
        return df


def _automate_content_measures(utterances: pd.Series, nlp) -> tuple[list[int], list[int], list[str]]:
    """
    Run spaCy over utterances and return content-word, noun, and tag outputs.
    """
    content_counts: list[int] = []
    noun_counts: list[int] = []
    tagged_utts: list[str] = []

    total_its = len(utterances)
    for doc in tqdm(
        nlp.pipe(utterances, batch_size=100, n_process=2),
        total=total_its,
        desc="Applying automation to utterances",
    ):
        num_content_words, num_nouns, tagged_utt = count_content_words_from_doc(doc)
        content_counts.append(num_content_words)
        noun_counts.append(num_nouns)
        tagged_utts.append(tagged_utt)

    return content_counts, noun_counts, tagged_utts
