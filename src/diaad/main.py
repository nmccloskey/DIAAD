#!/usr/bin/env python3
import argparse
import logging
from datetime import datetime
from diaad.utils.auxiliary import parse_stratify_fields, find_utt_files
from diaad.run_wrappers import (
    run_analyze_digital_convo_turns,
    run_make_POWERS_coding_files,
    run_analyze_POWERS_coding,
    run_evaluate_POWERS_reliability,
    run_reselect_POWERS_reliability_coding
)
from diaad.powers.automation_validation import select_validation_samples, validate_automation
from rascal.utils.auxiliary import as_path, load_config
from rascal.run_wrappers import run_read_tiers, run_read_cha_files, run_make_transcript_tables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------
# Unified parser builder
# ---------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the DIAAD argument parser (shared by CLI and direct run)."""
    parser = argparse.ArgumentParser(description="DIAAD CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- turns ---
    subparsers.add_parser("turns", help="Analyze digital conversation turns")

    # --- powers ---
    powers_parser = subparsers.add_parser("powers", help="POWERS coding workflow")
    powers_parser.add_argument(
        "action",
        choices=["make", "analyze", "evaluate", "reselect", "select", "validate"],
        help="POWERS step"
    )

    # --- powers: select ---
    powers_parser.add_argument(
        "--stratify",
        action="append",
        help="Fields to stratify by. Accepts repeated flags or comma/space-delimited list "
             "(e.g., --stratify site,test)."
    )
    powers_parser.add_argument(
        "--strata",
        type=int,
        default=5,
        help="Number of samples to draw per stratum (default: 5)."
    )
    powers_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for selection (default: 42)."
    )

    # --- powers: validate ---
    powers_parser.add_argument(
        "--selection",
        type=str,
        default=None,
        help="Optional path to a selection .xlsx to restrict validation (output of 'powers select')."
    )
    powers_parser.add_argument(
        "--numbers",
        type=str,
        default=None,
        help="Optional selection of stratum numbers to include in validation."
    )

    # --- global options ---
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)."
    )
    return parser


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------
def main(args):
    """Process input arguments and execute appropriate DIAAD operations."""
    config = load_config(args.config)
    input_dir = as_path(config.get('input_dir', 'diaad_data/input'))
    output_dir = as_path(config.get('output_dir', 'diaad_data/output'))
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)

    frac = config.get('reliability_fraction', 0.2)
    coders = config.get('coders', []) or []
    exclude_participants = config.get('exclude_participants', []) or []
    automate_POWERS = config.get('automate_POWERS', True)
    just_c2_POWERS = config.get('just_c2_POWERS', False)

    tiers = run_read_tiers(config.get('tiers', {})) or {}

    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    if args.command == "turns":
        out_root = output_dir / f"diaad_turns_output_{timestamp}"
    elif args.command == "powers":
        out_root = output_dir / f"diaad_powers_{args.action}_output_{timestamp}"
    else:
        out_root = output_dir / f"diaad_output_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Dispatch ---
    if args.command == "turns":
        run_analyze_digital_convo_turns(input_dir, out_root)

    elif args.command == "powers":
        if args.action == "make":
            utt_files = find_utt_files(input_dir, out_root)
            if not utt_files:
                chats = run_read_cha_files(input_dir)
                run_make_transcript_tables(tiers, chats, out_root)
            run_make_POWERS_coding_files(
                tiers, frac, coders, input_dir, out_root, exclude_participants, automate_POWERS
            )

        elif args.action == "analyze":
            run_analyze_POWERS_coding(input_dir, out_root, just_c2_POWERS)

        elif args.action == "evaluate":
            run_evaluate_POWERS_reliability(input_dir, out_root)

        elif args.action == "reselect":
            run_reselect_POWERS_reliability_coding(
                input_dir, out_root, frac, exclude_participants, automate_POWERS
            )

        elif args.action == "select":
            stratify_fields = parse_stratify_fields(args.stratify)
            select_validation_samples(
                input_dir=input_dir,
                output_dir=out_root,
                stratify=stratify_fields,
                strata=args.strata,
                seed=args.seed
            )

        elif args.action == "validate":
            selection_table = args.selection if args.selection else None
            stratum_numbers = parse_stratify_fields(args.numbers)
            validate_automation(
                input_dir=input_dir,
                output_dir=out_root,
                selection_table=selection_table,
                stratum_numbers=stratum_numbers
            )
            run_analyze_POWERS_coding(out_root, out_root, exclude_participants=exclude_participants)

        else:
            logging.error(f"Unknown powers action: {args.action}")
    else:
        logging.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)
