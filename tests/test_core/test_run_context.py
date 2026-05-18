from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

import diaad.core.run_context as run_context_module


class FakeMetadataManager:
    def __init__(self, config):
        self.config = config
        self.metadata_fields = {"group": "parsed"}


class FakeConfigManager:
    def __init__(self, config_dir, config_overrides=None):
        self.config_dir = config_dir
        self.config_overrides = dict(config_overrides or {})
        self.input_dir = "input"
        self.output_dir = "output"
        self.random_seed = 13
        self.reliability_fraction = 0.25
        self.shuffle_samples = True
        self.num_coders = 2
        self.num_bins = 4
        self.cu_paradigms = ["sv"]
        self.stimulus_field = "narrative"
        self.exclude_participants = ["INV"]
        self.strip_clan = True
        self.prefer_correction = False
        self.lowercase = True
        self.automate_powers = True
        self.powers_coding_file = "powers_coding.xlsx"
        self.powers_reliability_file = "powers_reliability_coding.xlsx"
        self.sample_id_field = "expanded_sample_id"
        self.utterance_id_field = "expanded_utterance_id"
        self.metadata_fields_config = {"tiers": {"group": "regex"}}
        self.advanced = SimpleNamespace(
            reliability_tag="_rel",
            reliability_dirname="reliability",
            cu_samples_file="cu_samples.xlsx",
            speaking_time_file="speaking_times.xlsx",
            speaking_time_field="speaking_time",
            word_count_file="word_counts.xlsx",
            word_count_field="word_count",
            wc_samples_file="wc_samples.xlsx",
            target_vocabulary_resource_path="",
            powers_coding_file="powers_coding.xlsx",
            powers_reliability_file="powers_reliability_coding.xlsx",
            sample_id_field="expanded_sample_id",
            utterance_id_field="expanded_utterance_id",
        )
        self.blinding = self.advanced
        self.override_diff = {}

    def to_dict(self):
        return {"project": {"input_dir": self.input_dir}, "advanced": {}}


def test_run_context_resolves_paths_and_builds_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(run_context_module, "ConfigManager", FakeConfigManager)
    monkeypatch.setattr(run_context_module, "MetadataManager", FakeMetadataManager)

    ctx = run_context_module.RunContext(
        config_dir=tmp_path / "config",
        project_root=tmp_path,
        start_time=datetime(2026, 4, 25, 12, 30),
    )

    assert ctx.input_dir == tmp_path / "input"
    assert ctx.base_output_dir == tmp_path / "output"
    assert ctx.out_dir.exists()
    assert ctx.metadata_fields == {"group": "parsed"}
    assert ctx.timestamp == "260425_1230"


def test_run_context_ensure_transcript_tables_generates_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(run_context_module, "ConfigManager", FakeConfigManager)
    monkeypatch.setattr(run_context_module, "MetadataManager", FakeMetadataManager)
    monkeypatch.setattr(run_context_module, "find_matching_files", lambda **kwargs: [])

    calls = {}

    def fake_tabularize_transcripts(**kwargs):
        calls.update(kwargs)

    import diaad.transcripts.transcript_tables as transcript_tables

    monkeypatch.setattr(transcript_tables, "tabularize_transcripts", fake_tabularize_transcripts)
    chats = {"file.cha": object()}

    ctx = run_context_module.RunContext(
        config_dir=tmp_path / "config",
        project_root=tmp_path,
        start_time=datetime(2026, 4, 25, 12, 30),
    )
    monkeypatch.setattr(ctx, "load_chats", lambda force=False: chats)

    ctx.ensure_transcript_tables()

    assert calls["metadata_fields"] == {"group": "parsed"}
    assert calls["chats"] is chats
    assert calls["shuffle"] is True
    assert calls["random_seed"] == 13


def test_run_context_kwargs_tabularize_requires_chats(monkeypatch, tmp_path):
    monkeypatch.setattr(run_context_module, "ConfigManager", FakeConfigManager)
    monkeypatch.setattr(run_context_module, "MetadataManager", FakeMetadataManager)

    ctx = run_context_module.RunContext(
        config_dir=tmp_path / "config",
        project_root=tmp_path,
        start_time=datetime(2026, 4, 25, 12, 30),
    )

    with pytest.raises(RuntimeError, match="CHAT files have not been loaded"):
        ctx.kwargs_tabularize_transcripts()


def test_run_context_threads_transcript_identifier_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(run_context_module, "ConfigManager", FakeConfigManager)
    monkeypatch.setattr(run_context_module, "MetadataManager", FakeMetadataManager)

    ctx = run_context_module.RunContext(
        config_dir=tmp_path / "config",
        project_root=tmp_path,
        start_time=datetime(2026, 4, 25, 12, 30),
    )
    ctx.chats = {"sample.cha": object()}

    tabularize_kwargs = ctx.kwargs_tabularize_transcripts()
    detabularize_kwargs = ctx.kwargs_detabularize_transcripts()

    assert tabularize_kwargs["sample_id_field"] == "expanded_sample_id"
    assert tabularize_kwargs["utterance_id_field"] == "expanded_utterance_id"
    assert detabularize_kwargs["sample_id_field"] == "expanded_sample_id"


def test_run_context_termination_kwargs(monkeypatch, tmp_path):
    monkeypatch.setattr(run_context_module, "ConfigManager", FakeConfigManager)
    monkeypatch.setattr(run_context_module, "MetadataManager", FakeMetadataManager)

    ctx = run_context_module.RunContext(
        config_dir=tmp_path / "config",
        project_root=tmp_path,
        start_time=datetime(2026, 4, 25, 12, 30),
    )
    payload = ctx.termination_kwargs()

    assert payload["input_dir"] == tmp_path / "input"
    assert payload["output_dir"] == ctx.out_dir
    assert payload["program_name"] == "DIAAD"


def test_run_context_can_resolve_paths_without_creating_output(monkeypatch, tmp_path):
    monkeypatch.setattr(run_context_module, "ConfigManager", FakeConfigManager)
    monkeypatch.setattr(run_context_module, "MetadataManager", FakeMetadataManager)

    ctx = run_context_module.RunContext(
        config_dir=tmp_path / "config",
        project_root=tmp_path,
        start_time=datetime(2026, 4, 25, 12, 30),
        create_output_dir=False,
    )

    assert ctx.out_dir == tmp_path / "output" / "diaad_260425_1230"
    assert not ctx.out_dir.exists()
    assert ctx.run_paths()["run_output_dir"] == str(ctx.out_dir)


def test_run_context_threads_powers_identifier_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(run_context_module, "ConfigManager", FakeConfigManager)
    monkeypatch.setattr(run_context_module, "MetadataManager", FakeMetadataManager)

    ctx = run_context_module.RunContext(
        config_dir=tmp_path / "config",
        project_root=tmp_path,
        start_time=datetime(2026, 4, 25, 12, 30),
    )

    assert ctx.kwargs_make_powers_coding_files()["sample_id_field"] == "expanded_sample_id"
    assert ctx.kwargs_make_powers_coding_files()["utterance_id_field"] == "expanded_utterance_id"
    assert ctx.kwargs_analyze_powers_coding()["sample_id_field"] == "expanded_sample_id"
    assert ctx.kwargs_powers_rates()["sample_id_field"] == "expanded_sample_id"
    assert ctx.kwargs_evaluate_powers_reliability()["sample_id_field"] == "expanded_sample_id"
    assert ctx.kwargs_evaluate_powers_reliability()["utterance_id_field"] == "expanded_utterance_id"
    assert ctx.kwargs_reselect_powers_reliability()["sample_id_field"] == "expanded_sample_id"


def test_run_context_threads_template_identifier_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(run_context_module, "ConfigManager", FakeConfigManager)
    monkeypatch.setattr(run_context_module, "MetadataManager", FakeMetadataManager)

    ctx = run_context_module.RunContext(
        config_dir=tmp_path / "config",
        project_root=tmp_path,
        start_time=datetime(2026, 4, 25, 12, 30),
    )

    utterance_kwargs = ctx.kwargs_make_utterance_templates()
    sample_kwargs = ctx.kwargs_make_sample_templates()
    time_kwargs = ctx.kwargs_make_speaking_time_templates()

    assert utterance_kwargs["sample_id_field"] == "expanded_sample_id"
    assert utterance_kwargs["utterance_id_field"] == "expanded_utterance_id"
    assert sample_kwargs["sample_id_field"] == "expanded_sample_id"
    assert time_kwargs["sample_id_field"] == "expanded_sample_id"
