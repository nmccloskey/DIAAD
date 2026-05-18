from __future__ import annotations


# ------------------------------------------------------------------
# Blinding
# ------------------------------------------------------------------

def run_encode_blinding(ctx):
    """Blind a general xlsx file using or generating a blind codebook."""
    from diaad.blinding.encode import encode_blinding

    return encode_blinding(**ctx.kwargs_encode_blinding())


def run_decode_blinding(ctx):
    """Decode a general xlsx file using a blind codebook."""
    from diaad.blinding.decode import decode_blinding

    return decode_blinding(**ctx.kwargs_decode_blinding())


# ------------------------------------------------------------------
# Transcription
# ------------------------------------------------------------------

def run_tabularize_transcripts(ctx):
    """Convert CHAT transcripts into tabular transcript files."""
    from diaad.transcripts.transcript_tables import tabularize_transcripts

    return tabularize_transcripts(**ctx.kwargs_tabularize_transcripts())


def run_detabularize_transcripts(ctx):
    """Convert transcript tables into CHAT files."""
    from diaad.transcripts.detabularization import detabularize_transcripts

    return detabularize_transcripts(**ctx.kwargs_detabularize_transcripts())


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

    return evaluate_cu_reliability(**ctx.kwargs_evaluate_cu_reliability())


def run_analyze_cu_coding(ctx):
    """Analyze finalized CU coding."""
    from diaad.coding.compl_utts.analysis import analyze_cu_coding

    return analyze_cu_coding(**ctx.kwargs_cu_analysis())

def run_calculate_cu_rates(ctx):
    """ Calculate CU rates from speaking times."""
    from diaad.coding.compl_utts.rates import calculate_cu_rates

    return calculate_cu_rates(**ctx.kwargs_cu_rates())


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

    return evaluate_word_count_reliability(
        **ctx.kwargs_evaluate_word_count_reliability()
    )

def run_analyze_word_counts(ctx):
    """Analyze word counts."""
    from diaad.coding.word_counts.analysis import analyze_word_counts

    return analyze_word_counts(**ctx.kwargs_analyze_word_counts())

def run_calculate_word_count_rates(ctx):
    """ Calculate CU rates from speaking times."""
    from diaad.coding.word_counts.rates import calculate_word_count_rates

    return calculate_word_count_rates(**ctx.kwargs_wc_rates())


# ------------------------------------------------------------------
# Target Vocabulary Coverage
# ------------------------------------------------------------------

def run_target_vocab(ctx):
    """Run target vocab coverage analysis."""
    from diaad.coding.target_vocab.analysis import run_target_vocab

    return run_target_vocab(**ctx.kwargs_target_vocab())

def run_check_target_vocab_resources(ctx):
    """Validate and summarize target vocabulary resources."""
    from diaad.coding.target_vocab.files import check_target_vocab_resources

    return check_target_vocab_resources(**ctx.kwargs_target_vocab_check())


def run_make_target_vocab_file(ctx):
    """Create a blank target vocabulary resource template."""
    from diaad.coding.target_vocab.files import make_target_vocab_file

    return make_target_vocab_file(**ctx.kwargs_target_vocab_file())


def run_calculate_target_vocab_rates(ctx):
    """Calculate target vocabulary per-minute rates from analysis output."""
    from diaad.coding.target_vocab.rates import calculate_target_vocab_rates

    return calculate_target_vocab_rates(**ctx.kwargs_target_vocab_rates())


# ------------------------------------------------------------------
# Digital Conversation Turns
# ------------------------------------------------------------------

def run_make_digital_convo_turn_files(ctx):
    """Create digital conversation turn coding templates."""
    from diaad.coding.convo_turns.files import make_digital_convo_turn_files

    return make_digital_convo_turn_files(**ctx.kwargs_make_digital_convo_turn_files())

def run_evaluate_digital_convo_turns(ctx):
    """Evaluate digital conversation turn reliability."""
    from diaad.coding.convo_turns.rel_evaluation import evaluate_digital_convo_turns_reliability

    return evaluate_digital_convo_turns_reliability(**ctx.kwargs_digital_convo_turns_reliability())


def run_reselect_digital_convo_turns(ctx):
    """Reselect digital conversation turn reliability samples."""
    from diaad.coding.convo_turns.rel_reselection import reselect_digital_convo_turns_rel

    return reselect_digital_convo_turns_rel(**ctx.kwargs_reselect_digital_convo_turns())


def run_analyze_digital_convo_turns(ctx):
    """Analyze digital conversation turns."""
    from diaad.coding.convo_turns.analysis import (
        analyze_digital_convo_turns,
    )

    return analyze_digital_convo_turns(**ctx.kwargs_digital_convo_turns())


# ------------------------------------------------------------------
# Generic coding templates
# ------------------------------------------------------------------

def run_make_utterance_templates(ctx):
    """Create utterance coding templates and reliability subsets."""
    from diaad.coding.templates.utterances import make_utterance_template_files

    return make_utterance_template_files(**ctx.kwargs_make_utterance_templates())

def run_make_speaking_time_templates(ctx):
    """Create speaking-time templates keyed by sample_id."""
    from diaad.coding.templates.times import make_speaking_time_template_files

    return make_speaking_time_template_files(**ctx.kwargs_make_speaking_time_templates())


def run_make_sample_templates(ctx):
    """Create sample coding templates and reliability subsets."""
    from diaad.coding.templates.samples import make_sample_template_files

    return make_sample_template_files(**ctx.kwargs_make_sample_templates())


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

def run_calculate_powers_rates(ctx):
    """Calculate POWERS per-minute rates from speaking times."""
    from diaad.coding.powers.rates import calculate_powers_rates

    return calculate_powers_rates(**ctx.kwargs_powers_rates())


def run_evaluate_powers_reliability(ctx):
    """Evaluate POWERS reliability."""
    from diaad.coding.powers.rel_evaluation import evaluate_powers_reliability

    return evaluate_powers_reliability(**ctx.kwargs_evaluate_powers_reliability())


def run_reselect_powers_reliability_coding(ctx):
    """Reselect POWERS reliability coding files."""
    from diaad.coding.powers.rel_reselection import reselect_powers_rel

    return reselect_powers_rel(**ctx.kwargs_reselect_powers_reliability())
