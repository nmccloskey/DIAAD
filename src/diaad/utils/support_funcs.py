import re
import logging
import pandas as pd
from pathlib import Path

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
