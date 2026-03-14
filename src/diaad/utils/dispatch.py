from __future__ import annotations

from collections.abc import Callable, Iterable

from diaad.run_wrappers import (
    run_analyze_cu_coding,
    run_analyze_digital_convo_turns,
    run_analyze_powers_coding,
    run_corelex,
    run_evaluate_cu_reliability,
    run_evaluate_powers_reliability,
    run_evaluate_transcription_reliability,
    run_evaluate_word_count_reliability,
    run_make_cu_coding_files,
    run_make_powers_coding_files,
    run_make_word_count_files,
    run_reselect_cu_rel,
    run_reselect_powers_reliability_coding,
    run_reselect_transcription_reliability_samples,
    run_reselect_wc_rel,
    run_select_for_validation,
    run_select_transcription_reliability_samples,
    run_tabularize_transcripts,
    run_validate_automation,
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
    "cus make",
    "corelex analyze",
    "powers make",
}


def commands_require_chats(commands: Iterable[str]) -> bool:
    """Return True if any command requires CHAT files."""
    return bool(set(commands) & CHAT_REQUIRED_COMMANDS)


def commands_require_transcript_tables(commands: Iterable[str]) -> bool:
    """Return True if any command requires transcript tables."""
    return bool(set(commands) & TRANSCRIPT_TABLE_REQUIRED_COMMANDS)

def prepare_dispatch_prerequisites(ctx, commands):
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
        # Transcription
        # --------------------------------------------------------------
        "transcripts tabularize": lambda: run_tabularize_transcripts(ctx),
        "transcripts select": lambda: run_select_transcription_reliability_samples(ctx),
        "transcripts reselect": lambda: run_reselect_transcription_reliability_samples(ctx),
        "transcripts evaluate": lambda: run_evaluate_transcription_reliability(ctx),

        # --------------------------------------------------------------
        # Complete Utterance coding
        # --------------------------------------------------------------
        "cus make": lambda: run_make_cu_coding_files(ctx),
        "cus reselect": lambda: run_reselect_cu_rel(ctx),
        "cus evaluate": lambda: run_evaluate_cu_reliability(ctx),
        "cus analyze": lambda: run_analyze_cu_coding(ctx),

        # --------------------------------------------------------------
        # Manual word counting
        # --------------------------------------------------------------
        "words make": lambda: run_make_word_count_files(ctx),
        "words reselect": lambda: run_reselect_wc_rel(ctx),
        "words evaluate": lambda: run_evaluate_word_count_reliability(ctx),

        # --------------------------------------------------------------
        # CoreLex
        # --------------------------------------------------------------
        "corelex analyze": lambda: run_corelex(ctx),

        # --------------------------------------------------------------
        # Digital Conversation Turns
        # --------------------------------------------------------------
        "turns analyze": lambda: run_analyze_digital_convo_turns(ctx),

        # --------------------------------------------------------------
        # POWERS coding workflow
        # --------------------------------------------------------------
        "powers make": lambda: run_make_powers_coding_files(ctx),
        "powers analyze": lambda: run_analyze_powers_coding(ctx),
        "powers evaluate": lambda: run_evaluate_powers_reliability(ctx),
        "powers reselect": lambda: run_reselect_powers_reliability_coding(ctx),

        # --------------------------------------------------------------
        # POWERS automation validation
        # --------------------------------------------------------------
        "powers select": lambda: run_select_for_validation(ctx),
        "powers validate": lambda: run_validate_automation(ctx),
    }
