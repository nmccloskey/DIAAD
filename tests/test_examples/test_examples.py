from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from diaad.examples import get_example_io_docs_path, iter_example_io_markdown_files
from diaad.examples import generate as generate_module
from diaad.examples.generate import generate_example_files
from diaad.examples.render_docs import render_example_docs
from psair.examples import long_path


def _exists(path):
    return long_path(path).exists()


def _assert_no_scratch_artifacts(path):
    assert not any(item.name.startswith("_dx_") for item in path.rglob("*"))


def test_generate_synthetic_project(tmp_path):
    project_dir = generate_example_files(tmp_path / "synthetic_project")

    _assert_no_scratch_artifacts(project_dir)
    assert (project_dir / "config" / "project.yaml").exists()
    assert (project_dir / "config" / "advanced.yaml").exists()
    assert not (project_dir / "config" / "advanced_project.yaml").exists()
    assert len(list((project_dir / "input" / "chat").glob("*.cha"))) == 3
    assert len(list((project_dir / "input" / "chat" / "reliability").glob("*.cha"))) == 2
    assert (
        project_dir / "input" / "transcript_tables" / "transcript_tables.xlsx"
    ).exists()
    assert (
        project_dir
        / "input"
        / "transcription_reliability_selection"
        / "transcription_reliability_samples.xlsx"
    ).exists()

    workbook = (
        project_dir
        / "expected_outputs"
        / "transcripts_module"
        / "transcripts_tabularize"
        / "transcript_tables.xlsx"
    )
    assert _exists(workbook)

    with pd.ExcelFile(long_path(workbook), engine="openpyxl") as xls:
        assert {"samples", "utterances"} <= set(xls.sheet_names)

    assert _exists(
        project_dir
        / "expected_outputs"
        / "transcripts_module"
        / "transcripts_select"
        / "transcription_reliability_samples.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "transcripts_module"
        / "transcripts_evaluate"
        / "transcription_reliability_evaluation.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "transcripts_module"
        / "transcripts_reselect"
        / "reselected_transcription_reliability"
        / "reselected_transcription_reliability_samples.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "blinding_module"
        / "blinding_encode"
        / "powers_coding_blinded.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "blinding_module"
        / "blinding_encode"
        / "blind_codebook.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "blinding_module"
        / "blinding_decode"
        / "cu_coding_decoded.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "templates_module"
        / "templates_utterances"
        / "utterance_coding_template.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "templates_module"
        / "templates_samples"
        / "sample_coding_template.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "templates_module"
        / "templates_times"
        / "speaking_times.xlsx"
    )
    assert _exists(
        project_dir
        / "input"
        / "sample_subset"
        / "sample_subset_input.xlsx"
    )
    assert _exists(
        project_dir
        / "input"
        / "sample_resubset"
        / "sample_resubset_input.xlsx"
    )
    subset_workbook = (
        project_dir
        / "expected_outputs"
        / "templates_module"
        / "templates_subset"
        / "sample_subset.xlsx"
    )
    resubset_workbook = (
        project_dir
        / "expected_outputs"
        / "templates_module"
        / "templates_resubset"
        / "sample_subset.xlsx"
    )
    assert _exists(subset_workbook)
    assert _exists(resubset_workbook)
    with pd.ExcelFile(long_path(subset_workbook), engine="openpyxl") as xls:
        assert {"samples", "subset"} <= set(xls.sheet_names)
    resubset_samples = pd.read_excel(long_path(resubset_workbook), sheet_name="samples")
    resubset_subset = pd.read_excel(long_path(resubset_workbook), sheet_name="subset")
    assert {"selected", "excluded"} <= set(resubset_samples.columns)
    assert not resubset_subset["excluded"].any()
    assert _exists(
        project_dir
        / "expected_outputs"
        / "cus_module"
        / "cus_files"
        / "cu_coding.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "cus_module"
        / "cus_evaluate"
        / "cu_reliability_coding_by_sample.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "cus_module"
        / "cus_reselect"
        / "reselected_cu_reliability_coding.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "cus_module"
        / "cus_analyze"
        / "cu_coding_by_sample_long.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "cus_module"
        / "cus_rates"
        / "cu_coding_rates.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "words_module"
        / "words_files"
        / "word_counting.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "words_module"
        / "words_evaluate"
        / "word_count_reliability_results.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "words_module"
        / "words_reselect"
        / "reselected_word_count_reliability.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "words_module"
        / "words_analyze"
        / "word_counting_by_sample.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "words_module"
        / "words_rates"
        / "word_counting_rates.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "powers_module"
        / "powers_files"
        / "powers_coding.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "powers_module"
        / "powers_files"
        / "powers_blind_codebook.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "powers_module"
        / "powers_evaluate"
        / "powers_reliability_results.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "powers_module"
        / "powers_reselect"
        / "reselected_powers_reliability_coding.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "powers_module"
        / "powers_analyze"
        / "powers_analysis.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "powers_module"
        / "powers_rates"
        / "powers_coding_rates.xlsx"
    )
    assert _exists(
        project_dir
        / "input"
        / "target_vocab"
        / "resources"
        / "picnic_target_vocab.json"
    )
    assert _exists(
        project_dir
        / "input"
        / "target_vocab"
        / "unblind_utterance_data.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "vocab_module"
        / "vocab_file"
        / "target_vocabulary_resource_template.json"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "vocab_module"
        / "vocab_check"
        / "target_vocab_resource_check.txt"
    )
    vocab_workbook = (
        project_dir
        / "expected_outputs"
        / "vocab_module"
        / "vocab_analyze"
        / "target_vocab_data_260101_0000.xlsx"
    )
    assert _exists(vocab_workbook)
    with pd.ExcelFile(long_path(vocab_workbook), engine="openpyxl") as xls:
        assert {"summary", "details"} <= set(xls.sheet_names)
    assert _exists(
        project_dir
        / "expected_outputs"
        / "vocab_module"
        / "vocab_rates"
        / "target_vocab_rates.xlsx"
    )
    assert _exists(
        project_dir
        / "input"
        / "conversation_turns"
        / "conversation_turns_template.xlsx"
    )
    assert _exists(
        project_dir
        / "input"
        / "conversation_turns"
        / "conversation_turns_reliability_template.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "turns_module"
        / "turns_files"
        / "conversation_turns_template.xlsx"
    )
    turns_reliability = (
        project_dir
        / "expected_outputs"
        / "turns_module"
        / "turns_evaluate"
        / "conversation_turns_reliability_results.xlsx"
    )
    assert _exists(turns_reliability)
    with pd.ExcelFile(long_path(turns_reliability), engine="openpyxl") as xls:
        assert {"counts", "sequences", "samples"} <= set(xls.sheet_names)
    assert _exists(
        project_dir
        / "expected_outputs"
        / "turns_module"
        / "turns_reselect"
        / "reselected_conversation_turns_reliability_template.xlsx"
    )
    assert _exists(
        project_dir
        / "expected_outputs"
        / "turns_module"
        / "turns_analyze"
        / "conversation_turns_template_analysis.xlsx"
    )


def test_command_example_capability_union_deduplicates_shared_inputs():
    assert generate_module._required_capabilities_for_commands(
        ["cus analyze", "cus evaluate"]
    ) == ("cu_coding_workbooks",)


def test_generate_transcripts_tabularize_command_example(tmp_path):
    package_dir = generate_example_files(
        tmp_path / "command_examples",
        commands=["transcripts tabularize"],
    )

    assert package_dir.name == "example_files_transcripts_tabularize"
    _assert_no_scratch_artifacts(package_dir)
    assert (package_dir / "README.md").exists()
    assert (package_dir / "example_config" / "project.yaml").exists()
    assert (package_dir / "example_config" / "advanced.yaml").exists()
    assert len(list((package_dir / "example_input" / "chat").glob("*.cha"))) == 3
    assert (
        package_dir
        / "example_output"
        / "transcript_tables"
        / "transcript_tables.xlsx"
    ).exists()
    assert (package_dir / "example_logs" / "diaad_example.log").exists()
    assert not (package_dir / "expected_outputs").exists()


def test_generate_combined_cu_command_example_reuses_shared_inputs(tmp_path):
    package_dir = generate_example_files(
        tmp_path / "command_examples",
        commands=["cus analyze", "cus evaluate"],
    )

    assert package_dir.name == "example_files_cus_analyze_cus_evaluate"
    _assert_no_scratch_artifacts(package_dir)
    assert (
        package_dir / "example_input" / "cu_coding" / "cu_coding.xlsx"
    ).exists()
    assert (
        package_dir / "example_input" / "cu_coding" / "cu_reliability_coding.xlsx"
    ).exists()
    assert not (package_dir / "example_input" / "transcript_tables").exists()
    assert (
        package_dir
        / "example_output"
        / "cu_coding_analysis"
        / "cu_coding_by_sample.xlsx"
    ).exists()
    assert (
        package_dir
        / "example_output"
        / "cu_reliability"
        / "cu_reliability_coding_by_sample.xlsx"
    ).exists()
    assert not (package_dir / "expected_outputs").exists()


def test_generate_command_example_rejects_valid_but_unsupported_command(tmp_path):
    with pytest.raises(ValueError, match="not yet available"):
        generate_example_files(tmp_path / "command_examples", commands=["words evaluate"])


def test_render_example_docs():
    paths = render_example_docs()

    assert any(path.name == "01_overview.md" for path in paths)
    assert any(path.name == "tabularize.md" for path in paths)
    assert any(path.name == "select.md" for path in paths)
    assert any(path.name == "evaluate.md" for path in paths)
    assert any(path.name == "reselect.md" for path in paths)
    assert any(path.name == "encode.md" and path.parent.name == "blinding" for path in paths)
    assert any(path.name == "decode.md" and path.parent.name == "blinding" for path in paths)
    assert any(path.name == "utterances.md" for path in paths)
    assert any(path.name == "samples.md" for path in paths)
    assert any(path.name == "times.md" for path in paths)
    assert any(path.name == "subset.md" and path.parent.name == "templates" for path in paths)
    assert any(path.name == "files.md" and path.parent.name == "cus" for path in paths)
    assert any(path.name == "analyze.md" and path.parent.name == "cus" for path in paths)
    assert any(path.name == "rates.md" and path.parent.name == "cus" for path in paths)
    assert any(path.name == "files.md" and path.parent.name == "words" for path in paths)
    assert any(path.name == "analyze.md" and path.parent.name == "words" for path in paths)
    assert any(path.name == "rates.md" and path.parent.name == "words" for path in paths)
    assert any(path.name == "files.md" and path.parent.name == "powers" for path in paths)
    assert any(path.name == "analyze.md" and path.parent.name == "powers" for path in paths)
    assert any(path.name == "rates.md" and path.parent.name == "powers" for path in paths)
    assert any(path.name == "file.md" and path.parent.name == "vocab" for path in paths)
    assert any(path.name == "check.md" and path.parent.name == "vocab" for path in paths)
    assert any(path.name == "analyze.md" and path.parent.name == "vocab" for path in paths)
    assert any(path.name == "rates.md" and path.parent.name == "vocab" for path in paths)
    assert any(path.name == "files.md" and path.parent.name == "turns" for path in paths)
    assert any(path.name == "evaluate.md" and path.parent.name == "turns" for path in paths)
    assert any(path.name == "reselect.md" and path.parent.name == "turns" for path in paths)
    assert any(path.name == "analyze.md" and path.parent.name == "turns" for path in paths)
    assert (get_example_io_docs_path() / "transcripts" / "tabularize.md").exists()


def test_generate_synthetic_project_uses_custom_transcript_table_filename(
    monkeypatch,
    tmp_path,
):
    specs = generate_module._read_specs()
    custom = "site_transcript_tables.xlsx"
    specs["advanced_config"] = deepcopy(specs["advanced_config"])
    specs["advanced_config"]["transcript_table_filename"] = custom
    specs["advanced_config"]["metadata_source"] = custom

    monkeypatch.setattr(generate_module, "_read_specs", lambda: deepcopy(specs))

    project_dir = generate_module.generate_example_files(
        tmp_path / "custom_transcript_table_project"
    )

    assert (project_dir / "input" / "transcript_tables" / custom).exists()
    assert not (
        project_dir / "input" / "transcript_tables" / "transcript_tables.xlsx"
    ).exists()
    assert (
        project_dir
        / "expected_outputs"
        / "transcripts_module"
        / "transcripts_tabularize"
        / custom
    ).exists()
    assert (
        project_dir
        / "expected_outputs"
        / "templates_module"
        / "templates_utterances"
        / "utterance_coding_template.xlsx"
    ).exists()
    assert (get_example_io_docs_path() / "blinding" / "encode.md").exists()
    assert (get_example_io_docs_path() / "templates" / "utterances.md").exists()
    assert (get_example_io_docs_path() / "templates" / "subset.md").exists()
    assert (get_example_io_docs_path() / "cus" / "files.md").exists()
    assert (get_example_io_docs_path() / "words" / "files.md").exists()
    assert (get_example_io_docs_path() / "powers" / "files.md").exists()
    assert (get_example_io_docs_path() / "vocab" / "analyze.md").exists()
    assert (get_example_io_docs_path() / "turns" / "analyze.md").exists()
    assert any(path.name == "tabularize.md" for path in iter_example_io_markdown_files())
