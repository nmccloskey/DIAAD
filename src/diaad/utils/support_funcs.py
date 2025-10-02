import re
import logging
import pandas as pd
from pathlib import Path

def as_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()

def read_df(file_path):
    try:
        df = pd.read_excel(str(file_path))
        logging.info(f"Successfully read file: {file_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to read file {file_path}: {e}")
        return None

def parse_stratify_fields(values: list[str] | None) -> list[str]:
    """
    Accepts:
      --stratify site test
      --stratify site,test
      --stratify "site, test"
      --stratify site --stratify test
    """
    if not values:
        return []
    items: list[str] = []
    for v in values:
        parts = re.split(r"[,\s]+", v.strip())
        parts = [x for x in parts if x]
        items.extend(parts)
    # preserve order but dedupe
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def run_analyze_digital_convo_turns(input_dir, output_dir):
    from diaad.convo_turns.digital_convo_turns_analyzer import analyze_digital_convo_turns
    analyze_digital_convo_turns(input_dir=input_dir, output_dir=output_dir)

def find_utt_files(input_dir, output_dir):
    logging.info("Searching for *Utterances*.xlsx files")
    utterance_files = list(Path(input_dir).rglob("*Utterances*.xlsx")) + \
        list(Path(output_dir).rglob("*Utterances*.xlsx"))
    logging.info(f"Found {len(utterance_files)} utterance file(s)")
    return utterance_files

def find_powers_coding_files(input_dir, output_dir):
    logging.info("Searching for *POWERS_Coding*.xlsx files")
    pc_files = list(Path(input_dir).rglob("*POWERS_Coding*.xlsx")) + \
        list(Path(output_dir).rglob("*POWERS_Coding*.xlsx"))
    logging.info(f"Found {len(pc_files)} POWERS Coding file(s)")
    return pc_files

def run_make_POWERS_coding_files(tiers, frac, coders, input_dir, output_dir, exclude_participants, automate_POWERS=True):
    from diaad.POWERS.prep_POWERS_coding_files import make_POWERS_coding_files
    make_POWERS_coding_files(
        tiers=tiers,
        frac=frac,
        coders=coders,
        input_dir=input_dir,
        output_dir=output_dir,
        exclude_participants=exclude_participants,
        automate_POWERS=automate_POWERS
    )

def run_analyze_POWERS_coding(input_dir, output_dir, just_c2_POWERS=False):
    from diaad.POWERS.analyze_POWERS_coding import analyze_POWERS_coding
    analyze_POWERS_coding(input_dir=input_dir, output_dir=output_dir, reliability=False, just_c2_POWERS=just_c2_POWERS)

def run_evaluate_POWERS_reliability(input_dir, output_dir):
    from diaad.POWERS.analyze_POWERS_coding import match_reliability_files, analyze_POWERS_coding
    match_reliability_files(input_dir=input_dir, output_dir=output_dir)
    analyze_POWERS_coding(input_dir=input_dir, output_dir=output_dir, reliability=True, just_c2_POWERS=False)

def run_reselect_POWERS_reliability_coding(input_dir, output_dir, frac, exclude_participants, automate_POWERS):
    from diaad.POWERS.prep_POWERS_coding_files import reselect_POWERS_reliability
    reselect_POWERS_reliability(
        input_dir=input_dir,
        output_dir=output_dir,
        frac=frac,
        exclude_participants=exclude_participants,
        automate_POWERS=automate_POWERS)
