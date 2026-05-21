from __future__ import annotations

from pathlib import Path

import pandas as pd

from psair.core.logger import get_rel_path, logger

from diaad.core.config import AdvancedConfig
from diaad.metadata.discovery import find_one_matching_file, require_one_file
from diaad.metadata.unblinding import unblind_dataframe, validate_decode_codebook
from diaad.metadata.utils import normalize_to_list


def _read_xlsx(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {get_rel_path(path)}")
    return pd.read_excel(path)


def _choose_first_path(matches, *, resource_name: str, directories=None) -> Path:
    paths = [Path(p) for p in normalize_to_list(matches)]

    return require_one_file(
        paths,
        label=f"{resource_name} file",
        directories=directories,
    )


def _find_blind_codebook(input_dir: str | Path) -> Path:
    matches = [
        Path(p)
        for p in sorted(Path(input_dir).rglob("*.xlsx"))
        if "blind_codebook" in p.stem.lower() and not p.name.startswith("~$")
    ]
    return _choose_first_path(
        matches,
        resource_name="blind codebook",
        directories=input_dir,
    )


def _find_named_codebook(input_dir: str | Path, codebook_filename: str) -> Path:
    filename = str(codebook_filename or "").strip()
    return find_one_matching_file(
        directories=input_dir,
        filename=filename,
        label="configured blind codebook file",
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
        directories=input_dir,
    )


def decode_blinding(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    blinding_config: AdvancedConfig,
) -> Path:
    """
    Decode one blinded xlsx file from input_dir using a blind codebook.
    """
    input_dir = Path(input_dir)
    out_dir = Path(output_dir) / "blinding"
    out_dir.mkdir(parents=True, exist_ok=True)

    codebook_path = (
        _find_named_codebook(input_dir, blinding_config.codebook_filename)
        if blinding_config.codebook_filename
        else _find_blind_codebook(input_dir)
    )
    target_path = _find_target_xlsx(input_dir, exclude_paths=[codebook_path])

    logger.info("Using blind codebook: %s", get_rel_path(codebook_path))
    logger.info("Decoding target xlsx: %s", get_rel_path(target_path))

    codebook_df = _read_xlsx(codebook_path)
    target_df = _read_xlsx(target_path)

    try:
        validate_decode_codebook(codebook_df)
        decoded_df = unblind_dataframe(
            target_df,
            codebook_df,
            suffix=blinding_config.blinded_suffix,
            strict=False,
        )
    except Exception as e:
        logger.error("Failed to decode %s: %s", get_rel_path(target_path), e)
        raise

    decoded_path = out_dir / f"{target_path.stem}_decoded.xlsx"
    decoded_df.to_excel(decoded_path, index=False)

    logger.info("Decoded xlsx written to %s", get_rel_path(decoded_path))
    return decoded_path
