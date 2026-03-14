from __future__ import annotations


# ------------------------------------------------------------------
# Transcription
# ------------------------------------------------------------------

def run_tabularize_transcripts(ctx):
    """Convert CHAT transcripts into tabular transcript files."""
    from diaad.transcripts.transcript_tables import tabularize_transcripts

    return tabularize_transcripts(**ctx.kwargs_tabularize_transcripts())


def run_select_transcription_reliability_samples(ctx):
    """Select transcript samples for transcription reliability coding."""
    from diaad.transcripts.transcription_reliability_selection import (
        select_transcription_reliability_samples,
    )

    return select_transcription_reliability_samples(
        **ctx.kwargs_select_transcription_reliability_samples()
    )


def run_reselect_transcription_reliability_samples(ctx):
    """Reselect transcript samples for transcription reliability coding."""
    from diaad.transcripts.transcription_reliability_selection import (
        reselect_transcription_reliability_samples,
    )

    return reselect_transcription_reliability_samples(
        **ctx.kwargs_reselect_transcription_reliability_samples()
    )


def run_evaluate_transcription_reliability(ctx):
    """Evaluate transcription reliability results."""
    from diaad.transcripts.transcription_reliability_evaluation import (
        evaluate_transcription_reliability,
    )

    return evaluate_transcription_reliability(
        **ctx.kwargs_evaluate_transcription_reliability()
    )


# ------------------------------------------------------------------
# Complete Utterance coding
# ------------------------------------------------------------------

def run_make_cu_coding_files(ctx):
    """Create CU coding and reliability files."""
    from diaad.coding.compl_utts.files import make_cu_coding_files

    return make_cu_coding_files(**ctx.kwargs_make_cu_coding_files())


def run_reselect_cu_rel(ctx):
    """Reselect CU reliability samples."""
    from diaad.coding.compl_utts.rel_reselection import reselect_cu_rel

    return reselect_cu_rel(**ctx.kwargs_reselect_cu_rel())


def run_evaluate_cu_reliability(ctx):
    """Evaluate CU reliability."""
    from diaad.coding.compl_utts.rel_evaluation import evaluate_cu_reliability

    return evaluate_cu_reliability(**ctx.kwargs_cu_analysis())


def run_analyze_cu_coding(ctx):
    """Analyze finalized CU coding."""
    from diaad.coding.compl_utts.analysis import analyze_cu_coding

    return analyze_cu_coding(**ctx.kwargs_cu_analysis())


# ------------------------------------------------------------------
# Manual word counting
# ------------------------------------------------------------------

def run_make_word_count_files(ctx):
    """Create manual word-count coding and reliability files."""
    from diaad.coding.word_counts.files import make_word_count_files

    return make_word_count_files(**ctx.kwargs_make_word_count_files())


def run_reselect_wc_rel(ctx):
    """Reselect word-count reliability samples."""
    from diaad.coding.word_counts.rel_reselection import reselect_wc_rel

    return reselect_wc_rel(**ctx.kwargs_reselect_wc_rel())


def run_evaluate_word_count_reliability(ctx):
    """Evaluate word-count reliability."""
    from diaad.coding.word_counts.rel_evaluation import (
        evaluate_word_count_reliability,
    )

    return evaluate_word_count_reliability(**ctx.kwargs_tiered_io())


# ------------------------------------------------------------------
# CoreLex
# ------------------------------------------------------------------

def run_corelex(ctx):
    """Run CoreLex analysis."""
    from diaad.coding.corelex.corelex import run_corelex

    return run_corelex(**ctx.kwargs_corelex())


# ------------------------------------------------------------------
# Digital Conversation Turns
# ------------------------------------------------------------------

def run_analyze_digital_convo_turns(ctx):
    """Analyze digital conversation turns."""
    from diaad.coding.convo_turns.digital_convo_turns import (
        analyze_digital_convo_turns,
    )

    return analyze_digital_convo_turns(**ctx.kwargs_digital_convo_turns())


# ------------------------------------------------------------------
# POWERS coding workflow
# ------------------------------------------------------------------

def run_make_powers_coding_files(ctx):
    """Create POWERS coding and reliability files."""
    from diaad.coding.powers.files import make_powers_coding_files

    return make_powers_coding_files(**ctx.kwargs_make_powers_coding_files())


def run_analyze_powers_coding(ctx):
    """Analyze POWERS coding results."""
    from diaad.coding.powers.analysis import analyze_powers_coding

    return analyze_powers_coding(**ctx.kwargs_analyze_powers_coding())


def run_evaluate_powers_reliability(ctx):
    """Match reliability files and analyze POWERS reliability."""
    from diaad.coding.powers.analysis import (
        analyze_powers_coding,
        match_reliability_files,
    )

    match_reliability_files(**ctx.kwargs_io())

    return analyze_powers_coding(
        **ctx.kwargs_analyze_powers_coding(
            reliability=True,
            just_c2_powers=False,
        )
    )


def run_reselect_powers_reliability_coding(ctx):
    """Reselect POWERS reliability coding files."""
    from diaad.coding.powers.files import reselect_powers_reliability

    return reselect_powers_reliability(**ctx.kwargs_reselect_powers_reliability())


# ------------------------------------------------------------------
# POWERS automation validation
# ------------------------------------------------------------------

def run_select_for_validation(ctx):
    """Select samples for POWERS automation validation."""
    from diaad.coding.powers.validation import (
        parse_stratify_fields,
        select_validation_samples,
    )

    kwargs = ctx.kwargs_select_for_validation()
    kwargs["stratify"] = parse_stratify_fields(kwargs.pop("stratify_by"))
    kwargs["strata"] = kwargs.pop("num_strata")

    return select_validation_samples(**kwargs)


def run_validate_automation(ctx):
    """Validate POWERS automation and analyze resulting coding."""
    from diaad.coding.powers.validation import (
        parse_stratify_fields,
        validate_automation,
    )

    kwargs = ctx.kwargs_validate_automation()
    kwargs["selection_table"] = kwargs["selection_table"] or None
    kwargs["stratum_numbers"] = parse_stratify_fields(kwargs["stratum_numbers"])

    validate_automation(
        input_dir=kwargs["input_dir"],
        output_dir=kwargs["output_dir"],
        selection_table=kwargs["selection_table"],
        stratum_numbers=kwargs["stratum_numbers"],
    )

    return run_analyze_powers_coding(ctx)
