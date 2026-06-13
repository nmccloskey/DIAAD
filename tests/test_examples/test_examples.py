from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest
import yaml

from diaad.examples import get_example_io_docs_path, iter_example_io_markdown_files
from diaad.examples import generate as generate_module
from diaad.examples.generate import generate_example_files
from diaad.examples.render_docs import render_example_docs
from psair.examples import long_path


def _exists(path):
    return long_path(path).exists()


def _assert_no_scratch_artifacts(path):
    assert not any(item.name.startswith("_dx_") for item in path.rglob("*"))


def _read_front_matter(path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---")
    _opening, front_matter, _body = text.split("---", 2)
    return yaml.safe_load(front_matter)


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


def test_command_example_capability_union_uses_prior_outputs():
    assert generate_module._required_capabilities_for_commands(
        ["transcripts tabularize", "templates utterances"]
    ) == ("synthetic_chat_files",)
    assert generate_module._required_capabilities_for_commands(
        ["cus analyze", "cus rates"]
    ) == ("cu_coding_workbooks", "cu_coding_analysis", "speaking_time_workbook")


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


def test_generate_combined_command_example_uses_prior_outputs(tmp_path):
    transcript_package = generate_example_files(
        tmp_path / "command_examples",
        commands=["transcripts tabularize", "templates utterances"],
    )

    assert not (transcript_package / "example_input" / "transcript_tables").exists()
    assert _exists(
        transcript_package
        / "example_output"
        / "transcript_tables"
        / "transcript_tables.xlsx"
    )
    assert _exists(
        transcript_package
        / "example_output"
        / "coding_templates"
        / "utterance_coding_template.xlsx"
    )


def test_generate_command_example_rejects_valid_but_unsupported_command(tmp_path):
    with pytest.raises(ValueError, match="not yet available"):
        generate_example_files(tmp_path / "command_examples", commands=["examples"])


def test_generate_command_example_rejects_unknown_command(tmp_path):
    with pytest.raises(ValueError, match="Unknown DIAAD command"):
        generate_example_files(tmp_path / "command_examples", commands=["not a command"])


def test_generate_templates_subset_must_be_individual(tmp_path):
    with pytest.raises(ValueError, match="templates subset"):
        generate_example_files(
            tmp_path / "command_examples",
            commands=["templates subset", "templates times"],
        )


@pytest.mark.parametrize(
    ("command", "expected_rel_path"),
    [
        (
            "transcripts chats",
            "example_output/chat_files/P1_picnic_pre_cha_source_input_chat_P1_picnic_pre_0.cha",
        ),
        (
            "transcripts select",
            "example_output/transcription_reliability_selection/transcription_reliability_samples.xlsx",
        ),
        (
            "transcripts reselect",
            "example_output/reselected_transcription_reliability/reselected_transcription_reliability_samples.xlsx",
        ),
        (
            "transcripts evaluate",
            "example_output/transcription_reliability_evaluation/transcription_reliability_evaluation.xlsx",
        ),
        (
            "templates utterances",
            "example_output/coding_templates/utterance_coding_template.xlsx",
        ),
        (
            "templates samples",
            "example_output/coding_templates/sample_coding_template.xlsx",
        ),
        ("templates times", "example_output/coding_templates/speaking_times.xlsx"),
        ("templates subset", "example_output/coding_templates/sample_subset.xlsx"),
        ("cus files", "example_output/cu_coding/cu_coding.xlsx"),
        (
            "cus reselect",
            "example_output/reselected_cu_coding_reliability/reselected_cu_reliability_coding.xlsx",
        ),
        ("cus rates", "example_output/cu_coding_analysis/cu_coding_rates.xlsx"),
        ("words files", "example_output/word_counts/word_counting.xlsx"),
        (
            "words reselect",
            "example_output/reselected_word_count_reliability/reselected_word_count_reliability.xlsx",
        ),
        (
            "words evaluate",
            "example_output/word_count_reliability/word_count_reliability_results.xlsx",
        ),
        (
            "words analyze",
            "example_output/word_count_analysis/word_counting_by_sample.xlsx",
        ),
        (
            "words rates",
            "example_output/word_count_analysis/word_counting_rates.xlsx",
        ),
    ],
)
def test_generate_pass_4_1_command_examples(tmp_path, command, expected_rel_path):
    package_dir = generate_example_files(
        tmp_path / "command_examples",
        commands=[command],
    )

    assert package_dir.name == f"example_files_{command.replace(' ', '_')}"
    _assert_no_scratch_artifacts(package_dir)
    assert (package_dir / "README.md").exists()
    assert (package_dir / "example_config" / "project.yaml").exists()
    assert _exists(package_dir / expected_rel_path)
    assert not (package_dir / "expected_outputs").exists()


@pytest.mark.parametrize(
    ("command", "expected_rel_path"),
    [
        ("blinding encode", "example_output/blinding/powers_coding_blinded.xlsx"),
        (
            "blinding decode",
            "example_output/blinding/powers_coding_blinded_decoded.xlsx",
        ),
        ("powers files", "example_output/powers_coding/powers_coding.xlsx"),
        (
            "powers evaluate",
            "example_output/powers_reliability/powers_reliability_results.xlsx",
        ),
        (
            "powers reselect",
            "example_output/reselected_powers_reliability/reselected_powers_reliability_coding.xlsx",
        ),
        ("powers analyze", "example_output/powers_coding_analysis/powers_analysis.xlsx"),
        ("powers rates", "example_output/powers_coding_analysis/powers_coding_rates.xlsx"),
        (
            "vocab file",
            "example_output/target_vocab/target_vocabulary_resource_template.json",
        ),
        (
            "vocab check",
            "example_output/target_vocab/target_vocab_resource_check.txt",
        ),
        (
            "vocab analyze",
            "example_output/target_vocab/target_vocab_data_260101_0000.xlsx",
        ),
        ("vocab rates", "example_output/target_vocab/target_vocab_rates.xlsx"),
        (
            "turns files",
            "example_output/coding_templates/conversation_turns_template.xlsx",
        ),
        (
            "turns evaluate",
            "example_output/turns_reliability/conversation_turns_reliability_results.xlsx",
        ),
        (
            "turns reselect",
            "example_output/reselected_turns_reliability/reselected_conversation_turns_reliability_template.xlsx",
        ),
        ("turns analyze", "example_output/conversation_turns_template_analysis.xlsx"),
    ],
)
def test_generate_pass_4_2_command_examples(tmp_path, command, expected_rel_path):
    package_dir = generate_example_files(
        tmp_path / "command_examples",
        commands=[command],
    )

    assert package_dir.name == f"example_files_{command.replace(' ', '_')}"
    _assert_no_scratch_artifacts(package_dir)
    assert (package_dir / "README.md").exists()
    assert (package_dir / "example_config" / "project.yaml").exists()
    assert _exists(package_dir / expected_rel_path)
    assert not (package_dir / "expected_outputs").exists()
    if command.startswith("vocab "):
        assert "example_input/target_vocab/resources" in (
            package_dir / "example_config" / "advanced.yaml"
        ).read_text()


def test_all_non_examples_commands_have_example_plans():
    unsupported = sorted(generate_module.VALID_COMMANDS - {"examples"} - set(generate_module.EXAMPLE_COMMAND_PLANS))

    assert unsupported == []


def test_example_command_identity_helpers():
    assert generate_module.canonical_command_to_command_id("cus files") == "cus.files"
    assert generate_module.command_id_to_slug("cus.files") == "cus_files"
    assert generate_module.command_id_parts("transcripts.tabularize") == (
        "transcripts",
        "tabularize",
    )
    assert generate_module.rendered_doc_path_for_command("blinding decode") == (
        "blinding",
        "decode.md",
    )
    assert generate_module.EXAMPLE_COMMAND_PLANS["cus files"].command_id == "cus.files"


def test_render_example_docs():
    paths = render_example_docs()

    assert any(path.name == "01_overview.md" for path in paths)
    assert any(path.name == "tabularize.md" for path in paths)
    assert any(path.name == "chats.md" for path in paths)
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


def test_rendered_example_docs_have_composable_front_matter():
    render_example_docs()
    docs_root = get_example_io_docs_path()

    overview = _read_front_matter(docs_root / "01_overview.md")
    assert overview["object_type"] == "workflow"
    assert overview["object_types"] == ["workflow", "command"]
    assert overview["workflow_id"] == "full_example_dataset"
    assert overview["command_id"] == "examples"
    assert overview["canonical_command"] == "examples"
    assert overview["command_subtype"] == "omnibus"
    assert overview["view"] == "example_io"

    for command, plan in generate_module.EXAMPLE_COMMAND_PLANS.items():
        doc_path = docs_root.joinpath(
            *generate_module.rendered_doc_path_for_command(command)
        )
        assert doc_path.exists(), command
        metadata = _read_front_matter(doc_path)
        module_id, _action_id = generate_module.command_id_parts(plan.command_id)
        assert metadata["object_type"] == "command"
        assert metadata["object_types"] == ["command"]
        assert metadata["command_id"] == plan.command_id
        assert metadata["canonical_command"] == command
        assert metadata["module_id"] == module_id
        assert metadata["view"] == "example_io"
        assert metadata["slot"] == "examples"
        assert metadata["title"]


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
