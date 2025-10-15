#!/usr/bin/env python3
"""
DIAAD Command-Line Interface
----------------------------
Entry point for the Digital Interface for Aphasiological Analysis of Discourse (DIAAD).
Provides subcommands for analyzing digital conversation turns and managing POWERS workflows.
"""

import argparse
from .main import main as main_core


def main():
    parser = argparse.ArgumentParser(description="DIAAD CLI.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- turns ---
    turns_parser = subparsers.add_parser(
        "turns",
        help="Analyze digital conversation turns"
    )

    # --- powers ---
    powers_parser = subparsers.add_parser(
        "powers",
        help="POWERS coding workflow"
    )
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
        help="Optional path to a selection .xlsx file (output of 'powers select')."
    )
    powers_parser.add_argument(
        "--numbers",
        type=str,
        default=None,
        help="Optional list of stratum numbers to include in validation."
    )

    # --- global options ---
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)."
    )

    args = parser.parse_args()
    main_core(args)


if __name__ == "__main__":
    main()
