from __future__ import annotations

from diaad.cli.parser import build_arg_parser


def test_parser_leaves_config_unset_when_omitted() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(["powers", "evaluate"])

    assert args.config is None


def test_parser_accepts_batch_config_overrides() -> None:
    parser = build_arg_parser()

    args = parser.parse_args(
        [
            "powers",
            "evaluate",
            "--config",
            "config",
            "--input-dir",
            "data/input/siteA",
            "--output-dir",
            "data/output/siteA",
            "--set",
            "powers_coding_filename=siteA_powers.xlsx",
            "--set",
            "powers_reliability_filename=siteA_powers_rel.xlsx",
            "--dry-run-config",
        ]
    )

    assert args.command == ["powers", "evaluate"]
    assert args.config == "config"
    assert args.input_dir == "data/input/siteA"
    assert args.output_dir == "data/output/siteA"
    assert args.set_values == [
        "powers_coding_filename=siteA_powers.xlsx",
        "powers_reliability_filename=siteA_powers_rel.xlsx",
    ]
    assert args.dry_run_config is True
