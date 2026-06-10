from __future__ import annotations

import math

import pandas as pd
import pytest

from diaad.coding.templates.subset import (
    SAMPLE_SUBSET_FILENAME,
    make_sample_subset_file,
)
from diaad.coding.templates.utils import TEMPLATE_SUBDIR
from diaad.metadata.discovery import MultipleFilesFoundError


def _write_input_workbook(path, samples_df):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        samples_df.to_excel(writer, sheet_name="samples", index=False)


def _read_output(path):
    return (
        pd.read_excel(path, sheet_name="samples"),
        pd.read_excel(path, sheet_name="subset"),
    )


def test_make_sample_subset_file_writes_plain_subset(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source = input_dir / "source.xlsx"
    _write_input_workbook(
        source,
        pd.DataFrame({"sample_id": ["S1", "S2", "S2", "S3", "S4", "S5"]}),
    )

    outpath = make_sample_subset_file(
        input_dir=input_dir,
        output_dir=output_dir,
        frac=0.4,
        seed=13,
    )

    assert outpath == output_dir / TEMPLATE_SUBDIR / SAMPLE_SUBSET_FILENAME
    samples_df, subset_df = _read_output(outpath)

    assert list(samples_df.columns) == ["sample_id", "selected", "excluded"]
    assert len(samples_df) == 5
    assert samples_df["excluded"].tolist() == [0, 0, 0, 0, 0]
    assert samples_df["selected"].sum() == math.ceil(0.4 * 5)
    assert subset_df.equals(samples_df.loc[samples_df["selected"] == 1].reset_index(drop=True))


def test_make_sample_subset_file_writes_re_subset_with_exclusions(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source = input_dir / "coding" / "source.xlsx"
    _write_input_workbook(
        source,
        pd.DataFrame(
            {
                "expanded_sample_id": ["S1", "S1", "S2", "S3", "S4", "S5"],
                "exclude": [1, 0, 0, 1, 0, 0],
            }
        ),
    )

    outpath = make_sample_subset_file(
        input_dir=input_dir,
        output_dir=output_dir,
        frac=0.4,
        sample_id_field="expanded_sample_id",
        seed=13,
    )
    samples_df, subset_df = _read_output(outpath)

    excluded = dict(zip(samples_df["expanded_sample_id"], samples_df["excluded"]))
    assert excluded == {"S1": 1, "S2": 0, "S3": 1, "S4": 0, "S5": 0}
    assert samples_df["selected"].sum() == math.ceil(0.4 * 5)
    assert not subset_df["excluded"].any()
    assert set(subset_df["expanded_sample_id"]) <= {"S2", "S4", "S5"}


def test_make_sample_subset_file_errors_on_multiple_xlsx(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_input_workbook(input_dir / "one.xlsx", pd.DataFrame({"sample_id": ["S1"]}))
    _write_input_workbook(input_dir / "nested" / "two.xlsx", pd.DataFrame({"sample_id": ["S2"]}))

    with pytest.raises(MultipleFilesFoundError, match="multiple sample subset input workbook files"):
        make_sample_subset_file(
            input_dir=input_dir,
            output_dir=output_dir,
            frac=0.2,
            seed=13,
        )


def test_make_sample_subset_file_requires_samples_sheet_and_sample_id(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    source = input_dir / "source.xlsx"
    source.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(source, engine="openpyxl") as writer:
        pd.DataFrame({"other": ["S1"]}).to_excel(
            writer,
            sheet_name="not_samples",
            index=False,
        )

    with pytest.raises(ValueError, match="must contain a 'samples' sheet"):
        make_sample_subset_file(
            input_dir=input_dir,
            output_dir=output_dir,
            frac=0.2,
            seed=13,
        )

    _write_input_workbook(source, pd.DataFrame({"other": ["S1"]}))
    with pytest.raises(Exception, match="sample_id"):
        make_sample_subset_file(
            input_dir=input_dir,
            output_dir=output_dir,
            frac=0.2,
            seed=13,
        )


def test_make_sample_subset_file_rejects_non_binary_exclude(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_input_workbook(
        input_dir / "source.xlsx",
        pd.DataFrame({"sample_id": ["S1", "S2"], "exclude": [0, 2]}),
    )

    with pytest.raises(ValueError, match="only 0/1 values"):
        make_sample_subset_file(
            input_dir=input_dir,
            output_dir=output_dir,
            frac=0.5,
            seed=13,
        )
