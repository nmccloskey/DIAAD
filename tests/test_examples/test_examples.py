from __future__ import annotations

import pandas as pd

from diaad.examples import get_example_io_docs_path, iter_example_io_markdown_files
from diaad.examples.generate import _long_path, generate_example_files
from diaad.examples.render_docs import render_example_docs


def _exists(path):
    return _long_path(path).exists()


def test_generate_synthetic_project(tmp_path):
    project_dir = generate_example_files(tmp_path / "synthetic_project")

    assert (project_dir / "config" / "project.yaml").exists()
    assert (project_dir / "config" / "advanced.yaml").exists()
    assert not (project_dir / "config" / "advanced_project.yaml").exists()
    assert len(list((project_dir / "input" / "chat").glob("*.cha"))) == 3
    assert len(list((project_dir / "input" / "chat" / "reliability").glob("*.cha"))) == 2
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
        / "transcript_table.xlsx"
    )
    assert _exists(workbook)

    with pd.ExcelFile(_long_path(workbook), engine="openpyxl") as xls:
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
    with pd.ExcelFile(_long_path(vocab_workbook), engine="openpyxl") as xls:
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
    with pd.ExcelFile(_long_path(turns_reliability), engine="openpyxl") as xls:
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


def test_render_example_docs():
    paths = render_example_docs()

    assert any(path.name == "01_overview.md" for path in paths)
    assert any(path.name == "tabularize.md" for path in paths)
    assert any(path.name == "select.md" for path in paths)
    assert any(path.name == "evaluate.md" for path in paths)
    assert any(path.name == "reselect.md" for path in paths)
    assert any(path.name == "utterances.md" for path in paths)
    assert any(path.name == "samples.md" for path in paths)
    assert any(path.name == "times.md" for path in paths)
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
    assert (get_example_io_docs_path() / "templates" / "utterances.md").exists()
    assert (get_example_io_docs_path() / "cus" / "files.md").exists()
    assert (get_example_io_docs_path() / "words" / "files.md").exists()
    assert (get_example_io_docs_path() / "powers" / "files.md").exists()
    assert (get_example_io_docs_path() / "vocab" / "analyze.md").exists()
    assert (get_example_io_docs_path() / "turns" / "analyze.md").exists()
    assert any(path.name == "tabularize.md" for path in iter_example_io_markdown_files())
