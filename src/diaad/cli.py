#!/usr/bin/env python3
import argparse
from .main import main as main_core

def main():
    parser = argparse.ArgumentParser(description="DIAAD CLI")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # turns
    turns_parser = subparsers.add_parser("turns", help="Analyze digital conversation turns")

    # powers
    powers_parser = subparsers.add_parser("powers", help="POWERS coding workflow")
    powers_parser.add_argument("action", choices=["make", "analyze", "reselect"], help="POWERS step")

    # global options
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")

    args = parser.parse_args()
    main_core(args)

if __name__ == "__main__":
    main()
