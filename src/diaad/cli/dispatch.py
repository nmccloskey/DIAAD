from __future__ import annotations

from collections.abc import Callable, Iterable

from diaad.core.run_wrappers import (
    run_decode_blinding,
    run_encode_blinding,
    run_analyze_cu_coding,
    run_analyze_word_counts,
    run_analyze_digital_convo_turns,
    run_analyze_powers_coding,
    run_calculate_powers_rates,
    run_calculate_target_vocab_rates,
    run_evaluate_digital_convo_turns,
    run_make_digital_convo_turn_files,
    run_check_target_vocab_resources,
    run_make_target_vocab_file,
    run_target_vocab,
    run_evaluate_cu_reliability,
    run_evaluate_powers_reliability,
    run_evaluate_transcription_reliability,
    run_evaluate_word_count_reliability,
    run_make_cu_coding_files,
    run_detabularize_transcripts,
    run_make_sample_templates,
    run_make_sample_subset,
    run_make_speaking_time_templates,
    run_make_utterance_templates,
    run_make_powers_coding_files,
    run_make_word_count_files,
    run_reselect_cu_rel,
    run_reselect_digital_convo_turns,
    run_reselect_powers_reliability_coding,
    run_reselect_transcription_reliability_samples,
    run_reselect_wc_rel,
    run_calculate_cu_rates,
    run_calculate_word_count_rates,
    run_select_transcription_reliability_samples,
    run_tabularize_transcripts,
)


CommandDispatch = dict[str, Callable[[], object]]


# ------------------------------------------------------------------
# Command requirement helpers
# ------------------------------------------------------------------

CHAT_REQUIRED_COMMANDS = {
    "transcripts tabularize",
    "transcripts select",
}

TRANSCRIPT_TABLE_REQUIRED_COMMANDS = {
    "cus files",
    "vocab analyze",
    "powers files",
    "transcripts chats",
    "templates utterances",
    "templates samples",
    "templates times",
    "turns files",
}

def commands_require_chats(commands: Iterable[str]) -> bool:
    """Return True if any command requires CHAT files."""
    return bool(set(commands) & CHAT_REQUIRED_COMMANDS)

def commands_require_transcript_tables(commands: Iterable[str]) -> bool:
    """Return True if any command requires transcript tables."""
    return bool(set(commands) & TRANSCRIPT_TABLE_REQUIRED_COMMANDS)

def prepare_dispatch_prerequisites(ctx, commands: Iterable[str]) -> None:
    """
    Load prerequisite runtime state before dispatching commands.
    """
    commands = list(commands)

    if commands_require_chats(commands):
        ctx.load_chats()

    if (
        "transcripts tabularize" not in commands
        and commands_require_transcript_tables(commands)
    ):
        ctx.ensure_transcript_tables()

# ------------------------------------------------------------------
# Dispatch construction
# ------------------------------------------------------------------

def build_dispatch(ctx) -> CommandDispatch:
    """
    Build the DIAAD command dispatch dictionary for a single run.

    Each dispatched callable is a zero-argument lambda that closes over
    the provided RunContext.
    """
    return {
        # --------------------------------------------------------------
        # Blinding
        # --------------------------------------------------------------
        "blinding encode": lambda: run_encode_blinding(ctx),
        "blinding decode": lambda: run_decode_blinding(ctx),

        # --------------------------------------------------------------
        # Transcription
        # --------------------------------------------------------------
        "transcripts tabularize": lambda: run_tabularize_transcripts(ctx),
        "transcripts chats": lambda: run_detabularize_transcripts(ctx),
        "transcripts select": lambda: run_select_transcription_reliability_samples(ctx),
        "transcripts reselect": lambda: run_reselect_transcription_reliability_samples(ctx),
        "transcripts evaluate": lambda: run_evaluate_transcription_reliability(ctx),

        # --------------------------------------------------------------
        # Complete Utterance coding
        # --------------------------------------------------------------
        "cus files": lambda: run_make_cu_coding_files(ctx),
        "cus reselect": lambda: run_reselect_cu_rel(ctx),
        "cus evaluate": lambda: run_evaluate_cu_reliability(ctx),
        "cus analyze": lambda: run_analyze_cu_coding(ctx),
        "cus rates": lambda: run_calculate_cu_rates(ctx),

        # --------------------------------------------------------------
        # Manual word counting
        # --------------------------------------------------------------
        "words files": lambda: run_make_word_count_files(ctx),
        "words reselect": lambda: run_reselect_wc_rel(ctx),
        "words evaluate": lambda: run_evaluate_word_count_reliability(ctx),
        "words analyze": lambda: run_analyze_word_counts(ctx),
        "words rates": lambda: run_calculate_word_count_rates(ctx),

        # --------------------------------------------------------------
        # Target Vocabulary Coverage
        # --------------------------------------------------------------
        "vocab file": lambda: run_make_target_vocab_file(ctx),
        "vocab check": lambda: run_check_target_vocab_resources(ctx),
        "vocab analyze": lambda: run_target_vocab(ctx),
        "vocab rates": lambda: run_calculate_target_vocab_rates(ctx),

        # --------------------------------------------------------------
        # Digital Conversation Turns
        # --------------------------------------------------------------
        "turns files": lambda: run_make_digital_convo_turn_files(ctx),
        "turns evaluate": lambda: run_evaluate_digital_convo_turns(ctx),
        "turns reselect": lambda: run_reselect_digital_convo_turns(ctx),
        "turns analyze": lambda: run_analyze_digital_convo_turns(ctx),

        # --------------------------------------------------------------
        # Generic coding templates
        # --------------------------------------------------------------
        "templates utterances": lambda: run_make_utterance_templates(ctx),
        "templates samples": lambda: run_make_sample_templates(ctx),
        "templates times": lambda: run_make_speaking_time_templates(ctx),
        "templates subset": lambda: run_make_sample_subset(ctx),

        # --------------------------------------------------------------
        # POWERS coding workflow
        # --------------------------------------------------------------
        "powers files": lambda: run_make_powers_coding_files(ctx),
        "powers analyze": lambda: run_analyze_powers_coding(ctx),
        "powers rates": lambda: run_calculate_powers_rates(ctx),
        "powers evaluate": lambda: run_evaluate_powers_reliability(ctx),
        "powers reselect": lambda: run_reselect_powers_reliability_coding(ctx),
    }
