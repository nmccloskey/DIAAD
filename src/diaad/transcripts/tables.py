from __future__ import annotations
import pandas as pd
from pathlib import Path
from diaad.utils.logger import logger, _rel


def extract_transcript_data(
    transcript_table_path: str | Path,
    kind: str = "joined",
) -> pd.DataFrame:
    """
    Load data from a transcript table Excel file.

    Parameters
    ----------
    transcript_table_path : str or Path
        Path to an Excel file produced by `tabularize_transcripts`.
    kind : {'utterance', 'sample', 'joined'}, default='joined'
        Which dataset to return:
          - 'utterance': utterance-level data
          - 'sample': sample-level metadata
          - 'joined': merged table of both (inner join on 'sample_id')

    Returns
    -------
    pandas.DataFrame
        The requested DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the `type` argument is invalid.
    """
    path = Path(transcript_table_path).expanduser().resolve()
    if not path.exists():
        logger.error(f"Transcript table not found: {_rel(path)}")
        raise FileNotFoundError(f"Transcript table not found: {path}")

    if kind not in {"sample", "utterance", "joined"}:
        raise ValueError(
            f"Invalid kind '{kind}'. Must be 'sample', 'utterance', or 'joined'."
        )

    try:
        with pd.ExcelFile(path, engine="openpyxl") as xls:
            sheet_names = {s.lower() for s in xls.sheet_names}
            sample_df = (
                pd.read_excel(xls, sheet_name="samples")
                if "samples" in sheet_names else None
            )
            utt_df = (
                pd.read_excel(xls, sheet_name="utterances")
                if "utterances" in sheet_names else None
            )

        if kind == "sample":
            if sample_df is None:
                raise ValueError("Sample sheet not found in transcript table.")
            logger.info(f"Loaded sample data from {_rel(path)}")
            return sample_df

        if kind == "utterance":
            if utt_df is None:
                raise ValueError("Utterance sheet not found in transcript table.")
            logger.info(f"Loaded utterance data from {_rel(path)}")
            return utt_df

        if sample_df is None or utt_df is None:
            raise ValueError("Both sheets required for joined kind are missing.")

        joined = sample_df.merge(utt_df, on="sample_id", how="inner")
        logger.info(f"Loaded joined transcript data from {_rel(path)}")
        return joined

    except Exception as e:
        logger.error(f"Failed to read {_rel(path)}: {e}")
        raise
