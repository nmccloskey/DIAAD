#!/usr/bin/env python3
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from rascal.main import load_config, run_read_tiers, run_read_cha_files, run_prepare_utterance_dfs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_analyze_digital_convo_turns(input_dir, output_dir):
    from .convo_turns.digital_convo_turns_analyzer import analyze_digital_convo_turns
    analyze_digital_convo_turns(input_dir=input_dir, output_dir=output_dir)

def check_for_utt_files(input_dir, output_dir):
    logging.info("Checking for *Utterances*.xlsx files")
    utterance_files = list(Path(input_dir).rglob("*Utterances*.xlsx")) + list(Path(output_dir).rglob("*Utterances*.xlsx"))
    logging.info(f"Found {len(utterance_files)} utterance file(s)")
    return bool(utterance_files)

def run_make_POWERS_coding_files(tiers, frac, coders, input_dir, output_dir, exclude_participants, automate_POWERS=True):
    from .POWERS.prep_POWERS_coding_files import make_POWERS_coding_files
    make_POWERS_coding_files(
        tiers=tiers,
        frac=frac,
        coders=coders,
        input_dir=input_dir,
        output_dir=output_dir,
        exclude_participants=exclude_participants,
        automate_POWERS=automate_POWERS
    )

def run_analyze_POWERS_coding(input_dir, output_dir):
    from .POWERS.analyze_POWERS_coding import analyze_POWERS_coding
    analyze_POWERS_coding(input_dir=input_dir, output_dir=output_dir, reliability=False)

def run_evaluate_POWERS_reliability(input_dir, output_dir):
    from .POWERS.analyze_POWERS_coding import match_reliability_files, analyze_POWERS_coding
    match_reliability_files(input_dir=input_dir, output_dir=output_dir)
    analyze_POWERS_coding(input_dir=input_dir, output_dir=output_dir, reliability=True)

def run_reselect_POWERS_reliability_coding(input_dir, output_dir):
    ...


def main(args):
    """Main function to process input arguments and execute appropriate steps."""
    config = load_config(args.config)
    input_dir = config.get('input_dir', 'diaad_data/input')
    output_dir = config.get('output_dir', 'diaad_data/output')
    
    frac = config.get('reliability_fraction', 0.2)
    coders = config.get('coders', [])
    exclude_participants = config.get('exclude_participants', []) or []
    automate_POWERS = config.get('automate', True)

    input_dir = os.path.abspath(os.path.expanduser(input_dir))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    os.makedirs(input_dir, exist_ok=True)
    tiers = run_read_tiers(config.get('tiers', {}))

    # --- Timestamped output folder ---
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    if args.command == "turns":
        output_dir = os.path.join(output_dir, f"diaad_turns_output_{timestamp}")
    elif args.command == "powers":
        output_dir = os.path.join(output_dir, f"diaad_powers_{args.action}_output_{timestamp}")
    else:
        output_dir = os.path.join(output_dir, f"diaad_output_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # --- Dispatch ---
    if args.command == "turns":
        run_analyze_digital_convo_turns(input_dir, output_dir)

    elif args.command == "powers":
        if args.action == "make":
            utt_files = check_for_utt_files(input_dir, output_dir)
            if not utt_files:
                chats = run_read_cha_files(input_dir)
                run_prepare_utterance_dfs(tiers, chats, output_dir)
            run_make_POWERS_coding_files(
                tiers, frac, coders, input_dir, output_dir, exclude_participants, automate_POWERS
            )
        elif args.action == "analyze":
            run_analyze_POWERS_coding(input_dir, output_dir)
        elif args.action == "evaluate":
            run_evaluate_POWERS_reliability(input_dir, output_dir)
        elif args.action == "reselect":
            run_reselect_POWERS_reliability_coding(input_dir, output_dir)

    else:
        logging.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIAAD CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # turns
    turns_parser = subparsers.add_parser("turns", help="Analyze digital conversation turns")
    # powers
    powers_parser = subparsers.add_parser("powers", help="POWERS coding workflow")
    powers_parser.add_argument("action", choices=["make", "analyze", "evaluate", "reselect"], help="POWERS step")
    # global options
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")

    args = parser.parse_args()
    main(args)
