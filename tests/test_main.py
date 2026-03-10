import types
from pathlib import Path

import pytest

import diaad.main as main_module


@pytest.fixture
def mock_utils(monkeypatch, tmp_path):
    """
    Patch filesystem/config/logger helpers so main() can be tested without
    touching the real project environment.
    """
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "random_seed": 99,
        "reliability_fraction": 0.2,
        "shuffle_samples": True,
        "coders": [1, 2, 3],
        "cu_paradigms": ["SAE", "AAE"],
        "exclude_participants": ["INV"],
        "strip_clan": True,
        "prefer_correction": True,
        "lowercase": True,
        "automate_POWERS": True,
        "just_c2_POWERS": True,
        "tiers": {
            "site": {"order": 1, "values": ["AC", "BU", "TU"]},
            "test": {"order": 2, "values": ["Pre", "Post", "Maint"]},
            "study_id": {"order": 3, "regex": r"(AC|BU|TU)\d+"},
            "narrative": {
                "order": 4,
                "values": [
                    "CATGrandpa",
                    "BrokenWindow",
                    "RefusedUmbrella",
                    "CatRescue",
                    "BirthdayScene",
                ],
            },
        },
    }

    def fake_project_path(pathlike):
        return Path(pathlike)

    monkeypatch.setattr(main_module, "project_path", fake_project_path)
    monkeypatch.setattr(main_module, "load_config", lambda _: config)
    monkeypatch.setattr(main_module, "set_root", lambda _: None)
    monkeypatch.setattr(main_module, "get_root", lambda: tmp_path)
    monkeypatch.setattr(main_module, "initialize_logger", lambda *a, **k: None)
    monkeypatch.setattr(main_module, "terminate_logger", lambda **k: None)

    class DummyLogger:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    monkeypatch.setattr(main_module, "logger", DummyLogger())

    return {
        "tmp_path": tmp_path,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "config": config,
    }


@pytest.fixture
def mock_run_functions(monkeypatch):
    """
    Patch all run_* wrappers and record calls in a shared dictionary.
    """
    calls = {}

    def make_stub(name, return_value=None):
        def _stub(*args, **kwargs):
            calls[name] = {"args": args, "kwargs": kwargs}
            return return_value
        return _stub

    monkeypatch.setattr(
        main_module,
        "run_read_tiers",
        make_stub("run_read_tiers", return_value=("tiers_obj", "tm_obj")),
    )
    monkeypatch.setattr(
        main_module,
        "run_read_cha_files",
        make_stub("run_read_cha_files", return_value="chats_obj"),
    )
    monkeypatch.setattr(
        main_module,
        "run_select_transcription_reliability_samples",
        make_stub("run_select_transcription_reliability_samples"),
    )
    monkeypatch.setattr(
        main_module,
        "run_reselect_transcription_reliability_samples",
        make_stub("run_reselect_transcription_reliability_samples"),
    )
    monkeypatch.setattr(
        main_module,
        "run_evaluate_transcription_reliability",
        make_stub("run_evaluate_transcription_reliability"),
    )
    monkeypatch.setattr(
        main_module,
        "run_tabularize_transcripts",
        make_stub("run_tabularize_transcripts"),
    )
    monkeypatch.setattr(
        main_module,
        "run_make_cu_coding_files",
        make_stub("run_make_cu_coding_files"),
    )
    monkeypatch.setattr(
        main_module,
        "run_evaluate_cu_reliability",
        make_stub("run_evaluate_cu_reliability"),
    )
    monkeypatch.setattr(
        main_module,
        "run_analyze_cu_coding",
        make_stub("run_analyze_cu_coding"),
    )
    monkeypatch.setattr(
        main_module,
        "run_reselect_cu_reliability",
        make_stub("run_reselect_cu_reliability"),
    )
    monkeypatch.setattr(
        main_module,
        "run_make_word_count_files",
        make_stub("run_make_word_count_files"),
    )
    monkeypatch.setattr(
        main_module,
        "run_evaluate_word_count_reliability",
        make_stub("run_evaluate_word_count_reliability"),
    )
    monkeypatch.setattr(
        main_module,
        "run_reselect_wc_reliability",
        make_stub("run_reselect_wc_reliability"),
    )
    monkeypatch.setattr(
        main_module,
        "run_summarize_cus",
        make_stub("run_summarize_cus"),
    )
    monkeypatch.setattr(
        main_module,
        "run_run_corelex",
        make_stub("run_run_corelex"),
    )
    monkeypatch.setattr(
        main_module,
        "run_analyze_digital_convo_turns",
        make_stub("run_analyze_digital_convo_turns"),
    )

    return calls


@pytest.fixture
def base_args():
    """
    Helper factory for argparse-style namespaces expected by main().
    """
    def _make(command, config=None):
        return types.SimpleNamespace(command=command, config=config)
    return _make


def test_main_executes_transcripts_tabularize(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["transcripts", "tabularize"])
    main_module.main(args)

    assert "run_read_tiers" in mock_run_functions
    assert "run_read_cha_files" in mock_run_functions
    assert "run_tabularize_transcripts" in mock_run_functions


def test_main_executes_transcripts_select(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["transcripts", "select"])
    main_module.main(args)

    assert "run_read_tiers" in mock_run_functions
    assert "run_read_cha_files" in mock_run_functions
    assert "run_select_transcription_reliability_samples" in mock_run_functions


def test_main_executes_transcripts_evaluate(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["transcripts", "evaluate"])
    main_module.main(args)

    assert "run_read_tiers" in mock_run_functions
    assert "run_evaluate_transcription_reliability" in mock_run_functions


def test_main_executes_transcripts_reselect(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["transcripts", "reselect"])
    main_module.main(args)

    assert "run_read_tiers" in mock_run_functions
    assert "run_reselect_transcription_reliability_samples" in mock_run_functions


def test_main_executes_cus_make(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["cus", "make"])
    main_module.main(args)

    assert "run_read_tiers" in mock_run_functions
    assert "run_make_cu_coding_files" in mock_run_functions


def test_main_executes_cus_evaluate(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["cus", "evaluate"])
    main_module.main(args)

    assert "run_evaluate_cu_reliability" in mock_run_functions


def test_main_executes_cus_analyze(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["cus", "analyze"])
    main_module.main(args)

    assert "run_analyze_cu_coding" in mock_run_functions


def test_main_executes_cus_reselect(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["cus", "reselect"])
    main_module.main(args)

    assert "run_reselect_cu_reliability" in mock_run_functions


def test_main_executes_cus_summarize(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["cus", "summarize"])
    main_module.main(args)

    assert "run_summarize_cus" in mock_run_functions


def test_main_executes_words_make(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["words", "make"])
    main_module.main(args)

    assert "run_make_word_count_files" in mock_run_functions


def test_main_executes_words_evaluate(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["words", "evaluate"])
    main_module.main(args)

    assert "run_evaluate_word_count_reliability" in mock_run_functions


def test_main_executes_words_reselect(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["words", "reselect"])
    main_module.main(args)

    assert "run_reselect_wc_reliability" in mock_run_functions


def test_main_executes_corelex_analyze(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["corelex", "analyze"])
    main_module.main(args)

    assert "run_run_corelex" in mock_run_functions


def test_main_executes_turns_analyze(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["turns", "analyze"])
    main_module.main(args)

    assert "run_analyze_digital_convo_turns" in mock_run_functions


def test_main_auto_creates_transcript_tables_when_required(
    monkeypatch, mock_utils, mock_run_functions, base_args
):
    """
    For commands that require transcript tables (currently cus make, corelex analyze),
    main() should auto-create transcript tables if none are found.
    """
    monkeypatch.setattr(main_module, "find_files", lambda **kwargs: [])

    args = base_args(["cus", "make"])
    main_module.main(args)

    assert "run_read_cha_files" in mock_run_functions
    assert "run_tabularize_transcripts" in mock_run_functions
    assert "run_make_cu_coding_files" in mock_run_functions


def test_main_does_not_auto_create_transcript_tables_for_transcripts_tabularize(
    monkeypatch, mock_utils, mock_run_functions, base_args
):
    """
    transcripts tabularize should dispatch directly and not go through the
    auto-create transcript-table prerequisite branch.
    """
    monkeypatch.setattr(main_module, "find_files", lambda **kwargs: [])

    args = base_args(["transcripts", "tabularize"])
    main_module.main(args)

    assert "run_tabularize_transcripts" in mock_run_functions


def test_main_executes_multiple_commands(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["transcripts", "tabularize,", "cus", "make"])
    main_module.main(args)

    assert "run_tabularize_transcripts" in mock_run_functions
    assert "run_make_cu_coding_files" in mock_run_functions


def test_main_skips_unrecognized_command_and_continues(
    mock_utils, mock_run_functions, base_args
):
    args = base_args(["nonsense", "oops,", "words", "make"])
    main_module.main(args)

    assert "run_make_word_count_files" in mock_run_functions
