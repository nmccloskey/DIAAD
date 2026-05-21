from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from psair.core.logger import get_rel_path, logger

from diaad.core.config import AdvancedConfig
from diaad.metadata.discovery import find_one_matching_file, require_one_file
from diaad.metadata.blinding import blind_analysis_dataframe, write_blind_codebook
from diaad.metadata.utils import normalize_to_list, present_cols, validate_columns


def _read_xlsx(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {get_rel_path(path)}")
    return pd.read_excel(path)


def _choose_first_path(
    matches,
    *,
    resource_name: str,
    required: bool = False,
    directories=None,
) -> Path | None:
    paths = [Path(p) for p in normalize_to_list(matches)]

    if not paths and not required:
        return None

    return require_one_file(
        paths,
        label=f"{resource_name} file",
        directories=directories,
    )


def _find_blind_codebook(input_dir: str | Path) -> Path | None:
    matches = [
        Path(p)
        for p in sorted(Path(input_dir).rglob("*.xlsx"))
        if "blind_codebook" in p.stem.lower() and not p.name.startswith("~$")
    ]
    if not matches:
        return None
    return _choose_first_path(
        matches,
        resource_name="blind codebook",
        required=False,
        directories=input_dir,
    )


def _find_target_xlsx(
    input_dir: str | Path,
    *,
    exclude_paths: list[str | Path] | None = None,
) -> Path:
    excluded = {Path(p).resolve() for p in normalize_to_list(exclude_paths)}
    target_matches = [
        Path(p)
        for p in sorted(Path(input_dir).rglob("*.xlsx"))
        if p.resolve() not in excluded
        and "blind_codebook" not in p.stem.lower()
        and not p.name.startswith("~$")
    ]

    return _choose_first_path(
        target_matches,
        resource_name="non-blind-codebook xlsx",
        required=True,
        directories=input_dir,
    )


def _config_for_present_analysis_cols(
    df: pd.DataFrame,
    config: AdvancedConfig,
    requested_cols: list[str],
    *,
    source_name: str,
) -> AdvancedConfig:
    requested_cols = list(dict.fromkeys(requested_cols))
    available_cols = present_cols(df, requested_cols)
    missing_cols = [col for col in requested_cols if col not in available_cols]

    if missing_cols:
        logger.warning(
            "%s column(s) not found in target xlsx and will be skipped by the "
            "general blinding command: %s",
            source_name,
            missing_cols,
        )

    if not available_cols:
        raise ValueError(
            f"None of the {source_name} columns were present in the target xlsx. "
            f"Requested columns: {requested_cols}"
        )

    logger.info("Applying blinding to column(s): %s", available_cols)
    return replace(
        config,
        blind_columns=available_cols,
    )


def _codebook_target_cols(codebook_df: pd.DataFrame) -> list[str]:
    validate_columns(codebook_df, ["column"], df_name="blind codebook")
    return codebook_df["column"].dropna().astype(str).drop_duplicates().tolist()


def _find_named_codebook(input_dir: str | Path, codebook_filename: str) -> Path:
    filename = str(codebook_filename or "").strip()
    return find_one_matching_file(
        directories=input_dir,
        filename=filename,
        label="configured blind codebook file",
    )


def encode_blinding(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    blinding_config: AdvancedConfig,
    seed: int = 99,
) -> tuple[Path, Path, Path | None]:
    """
    Blind one xlsx file from input_dir, optionally reusing a blind codebook.

    The command discovers:
      1. the first ``*blind_codebook*.xlsx`` file, if present;
      2. the first non-codebook ``.xlsx`` file to blind.

    Output is written under ``<output_dir>/blinding``.
    """
    input_dir = Path(input_dir)
    out_dir = Path(output_dir) / "blinding"
    out_dir.mkdir(parents=True, exist_ok=True)

    codebook_path = (
        _find_named_codebook(input_dir, blinding_config.codebook_filename)
        if blinding_config.codebook_filename
        else _find_blind_codebook(input_dir)
    )
    target_path = _find_target_xlsx(
        input_dir,
        exclude_paths=[codebook_path] if codebook_path is not None else None,
    )

    logger.info("Blinding target xlsx: %s", get_rel_path(target_path))
    target_df = _read_xlsx(target_path)

    existing_codebook = None
    if codebook_path is None:
        logger.info(
            "No blind codebook found in input directory; generating a new blind codebook."
        )
        command_config = _config_for_present_analysis_cols(
            target_df,
            blinding_config,
            blinding_config.get_blind_cols("analysis") or [],
            source_name="configured blind_columns",
        )
    else:
        logger.info("Using blind codebook: %s", get_rel_path(codebook_path))
        existing_codebook = _read_xlsx(codebook_path)
        command_config = _config_for_present_analysis_cols(
            target_df,
            blinding_config,
            _codebook_target_cols(existing_codebook),
            source_name="blind codebook",
        )

    try:
        blinded_df, diagnostics_df, codebook_df = blind_analysis_dataframe(
            target_df,
            command_config,
            existing_codebook=existing_codebook,
            discover_existing_codebook=existing_codebook is not None,
            seed=seed,
        )
    except Exception as e:
        logger.error("Failed to blind %s: %s", get_rel_path(target_path), e)
        raise

    blinded_path = out_dir / f"{target_path.stem}_blinded.xlsx"
    diagnostics_path = out_dir / f"{target_path.stem}_blinding_diagnostics.xlsx"
    output_codebook_path = out_dir / "blind_codebook.xlsx"

    blinded_df.to_excel(blinded_path, index=False)
    diagnostics_df.to_excel(diagnostics_path, index=False)
    write_blind_codebook(codebook_df, output_codebook_path)

    logger.info("Blinded xlsx written to %s", get_rel_path(blinded_path))
    logger.info("Blinding diagnostics written to %s", get_rel_path(diagnostics_path))

    return blinded_path, output_codebook_path, diagnostics_path
