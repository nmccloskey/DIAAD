from __future__ import annotations

from pathlib import Path

import pandas as pd

from psair.core.logger import get_rel_path, logger

from diaad.core.config import AdvancedConfig
from diaad.metadata.unblinding import unblind_dataframe, validate_decode_codebook
from diaad.metadata.utils import normalize_to_list


def _read_xlsx(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {get_rel_path(path)}")
    return pd.read_excel(path)


def _choose_first_path(matches, *, resource_name: str) -> Path:
    paths = [Path(p) for p in normalize_to_list(matches)]

    if not paths:
        raise FileNotFoundError(
            f"No {resource_name} file found. Please provide one in the input directory."
        )

    if len(paths) > 1:
        logger.warning(
            "Multiple %s files found; using %s. All matches: %s",
            resource_name,
            get_rel_path(paths[0]),
            [get_rel_path(p) for p in paths],
        )

    return paths[0]


def _find_blind_codebook(input_dir: str | Path) -> Path:
    matches = [
        Path(p)
        for p in sorted(Path(input_dir).rglob("*.xlsx"))
        if "blind_codebook" in p.stem.lower() and not p.name.startswith("~$")
    ]
    return _choose_first_path(
        matches,
        resource_name="blind codebook",
    )


def _find_named_codebook(input_dir: str | Path, codebook_filename: str) -> Path:
    filename = str(codebook_filename or "").strip()
    candidate = Path(filename).expanduser()

    if candidate.is_absolute():
        matches = [candidate] if candidate.exists() else []
    else:
        matches = list(Path(input_dir).rglob(filename))

    if not matches:
        raise FileNotFoundError(
            "Configured blind codebook file was not found. Check the "
            f"codebook_filename setting in advanced.yaml: {filename!r}"
        )

    return _choose_first_path(
        matches,
        resource_name="configured blind codebook",
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
