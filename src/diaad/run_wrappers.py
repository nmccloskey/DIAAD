
# ------------------------------------------------------------------
# Transcription
# ------------------------------------------------------------------

def run_tabularize_transcripts(tiers, chats, output_dir, shuffle_samples, random_seed):
    """Convert CHAT transcripts into tabular transcript files."""
    from diaad.transcripts.transcript_tables import tabularize_transcripts

    return tabularize_transcripts(
        tiers=tiers,
        chats=chats,
        output_dir=output_dir,
        shuffle=shuffle_samples,
        random_seed=random_seed,
    )


def run_select_transcription_reliability_samples(tiers, chats, frac, output_dir):
    """Select transcript samples for transcription reliability coding."""
    from diaad.transcripts.transcription_reliability_selection import (
        select_transcription_reliability_samples,
    )

    return select_transcription_reliability_samples(
        tiers=tiers,
        chats=chats,
        frac=frac,
        output_dir=output_dir,
    )


def run_reselect_transcription_reliability_samples(input_dir, output_dir, frac):
    """Reselect transcript samples for transcription reliability coding."""
    from diaad.transcripts.transcription_reliability_selection import (
        reselect_transcription_reliability_samples,
    )

    return reselect_transcription_reliability_samples(
        input_dir=input_dir,
        output_dir=output_dir,
        frac=frac,
    )


def run_evaluate_transcription_reliability(
    tiers,
    input_dir,
    output_dir,
    exclude_participants,
    strip_clan,
    prefer_correction,
    lowercase,
):
    """Evaluate transcription reliability results."""
    from diaad.transcripts.transcription_reliability_evaluation import (
        evaluate_transcription_reliability,
    )

    return evaluate_transcription_reliability(
        tiers=tiers,
        input_dir=input_dir,
        output_dir=output_dir,
        exclude_participants=exclude_participants,
        strip_clan=strip_clan,
        prefer_correction=prefer_correction,
        lowercase=lowercase,
    )


# ------------------------------------------------------------------
# Complete Utterance coding
# ------------------------------------------------------------------

def run_make_cu_coding_files(
    tiers,
    frac,
    coders,
    input_dir,
    output_dir,
    cu_paradigms,
    exclude_participants,
    narrative_field,
):
    """Create CU coding and reliability files."""
    from diaad.coding.compl_utts.files import make_cu_coding_files

    return make_cu_coding_files(
        tiers=tiers,
        frac=frac,
        coders=coders,
        input_dir=input_dir,
        output_dir=output_dir,
        cu_paradigms=cu_paradigms,
        exclude_participants=exclude_participants,
        narrative_field=narrative_field,
    )


def run_reselect_cu_rel(tiers, input_dir, output_dir, frac, random_seed):
    """Reselect CU reliability samples."""
    from diaad.coding.compl_utts.rel_reselection import reselect_cu_rel

    return reselect_cu_rel(
        tiers=tiers,
        input_dir=input_dir,
        output_dir=output_dir,
        frac=frac,
        random_seed=random_seed
    )


def run_evaluate_cu_reliability(tiers, input_dir, output_dir, cu_paradigms):
    """Evaluate CU reliability."""
    from diaad.coding.compl_utts.rel_evaluation import evaluate_cu_reliability

    return evaluate_cu_reliability(
        tiers=tiers,
        input_dir=input_dir,
        output_dir=output_dir,
        cu_paradigms=cu_paradigms,
    )


def run_analyze_cu_coding(tiers, input_dir, output_dir, cu_paradigms):
    """Analyze finalized CU coding."""
    from diaad.coding.compl_utts.analysis import analyze_cu_coding

    return analyze_cu_coding(
        tiers=tiers,
        input_dir=input_dir,
        output_dir=output_dir,
        cu_paradigms=cu_paradigms,
    )


# ------------------------------------------------------------------
# Manual word counting
# ------------------------------------------------------------------

def run_make_word_count_files(tiers, frac, coders, input_dir, output_dir):
    """Create manual word-count coding and reliability files."""
    from diaad.coding.word_counts.files import make_word_count_files

    return make_word_count_files(
        tiers=tiers,
        frac=frac,
        coders=coders,
        input_dir=input_dir,
        output_dir=output_dir,
    )


def run_reselect_wc_rel(tiers, input_dir, output_dir, frac, random_seed):
    """Reselect word-count reliability samples."""
    from diaad.coding.word_counts.rel_reselection import reselect_wc_rel

    return reselect_wc_rel(
        tiers=tiers,
        input_dir=input_dir,
        output_dir=output_dir,
        frac=frac,
        random_seed=random_seed
    )


def run_evaluate_word_count_reliability(tiers, input_dir, output_dir):
    """Evaluate word-count reliability."""
    from diaad.coding.word_counts.rel_evaluation import (
        evaluate_word_count_reliability,
    )

    return evaluate_word_count_reliability(
        tiers=tiers,
        input_dir=input_dir,
        output_dir=output_dir,
    )


# ------------------------------------------------------------------
# CoreLex
# ------------------------------------------------------------------

def run_corelex(tiers, input_dir, output_dir, exclude_participants, narrative_field):
    """Run CoreLex analysis."""
    from diaad.coding.corelex.corelex import run_corelex

    return run_corelex(
        tiers=tiers,
        input_dir=input_dir,
        output_dir=output_dir,
        exclude_participants=exclude_participants,
        narrative_field=narrative_field,
    )


# ------------------------------------------------------------------
# Digital Conversation Turns
# ------------------------------------------------------------------

def run_analyze_digital_convo_turns(input_dir, output_dir):
    """Analyze digital conversation turns."""
    from diaad.coding.convo_turns.digital_convo_turns import (
        analyze_digital_convo_turns,
    )

    return analyze_digital_convo_turns(
        input_dir=input_dir,
        output_dir=output_dir,
    )


# ------------------------------------------------------------------
# POWERS coding workflow
# ------------------------------------------------------------------

def run_make_powers_coding_files(
    tiers,
    frac,
    coders,
    input_dir,
    output_dir,
    exclude_participants,
    automate_powers=True,
):
    """Create POWERS coding and reliability files."""
    from diaad.coding.powers.files import make_powers_coding_files

    return make_powers_coding_files(
        tiers=tiers,
        frac=frac,
        coders=coders,
        input_dir=input_dir,
        output_dir=output_dir,
        exclude_participants=exclude_participants,
        automate_powers=automate_powers,
    )


def run_analyze_powers_coding(
    input_dir,
    output_dir,
    reliability=False,
    just_c2_powers=False,
    exclude_participants=None,
):
    """Analyze POWERS coding results."""
    from diaad.coding.powers.analysis import analyze_powers_coding

    if exclude_participants is None:
        exclude_participants = []

    return analyze_powers_coding(
        input_dir=input_dir,
        output_dir=output_dir,
        reliability=reliability,
        just_c2_powers=just_c2_powers,
        exclude_participants=exclude_participants,
    )


def run_evaluate_powers_reliability(input_dir, output_dir):
    """Match reliability files and analyze POWERS reliability."""
    from diaad.coding.powers.analysis import (
        analyze_powers_coding,
        match_reliability_files,
    )

    match_reliability_files(input_dir=input_dir, output_dir=output_dir)

    return analyze_powers_coding(
        input_dir=input_dir,
        output_dir=output_dir,
        reliability=True,
        just_c2_powers=False,
    )


def run_reselect_powers_reliability_coding(
    input_dir,
    output_dir,
    frac,
    exclude_participants,
    automate_powers,
):
    """Reselect POWERS reliability coding files."""
    from diaad.coding.powers.files import reselect_powers_reliability

    return reselect_powers_reliability(
        input_dir=input_dir,
        output_dir=output_dir,
        frac=frac,
        exclude_participants=exclude_participants,
        automate_powers=automate_powers,
    )


# ------------------------------------------------------------------
# POWERS automation validation
# ------------------------------------------------------------------

def run_select_for_validation(stratify_by, input_dir, output_dir, num_strata, random_seed):
    """Select samples for POWERS automation validation."""
    from diaad.coding.powers.validation import (
        parse_stratify_fields,
        select_validation_samples,
    )

    stratify_fields = parse_stratify_fields(stratify_by)

    return select_validation_samples(
        input_dir=input_dir,
        output_dir=output_dir,
        stratify=stratify_fields,
        strata=num_strata,
        random_seed=random_seed,
    )


def run_validate_automation(
    selection_table,
    stratum_numbers,
    input_dir,
    output_dir,
    exclude_participants,
):
    """Validate POWERS automation and analyze resulting coding."""
    from diaad.coding.powers.validation import (
        parse_stratify_fields,
        validate_automation,
    )

    parsed_selection_table = selection_table or None
    parsed_stratum_numbers = parse_stratify_fields(stratum_numbers)

    validate_automation(
        input_dir=input_dir,
        output_dir=output_dir,
        selection_table=parsed_selection_table,
        stratum_numbers=parsed_stratum_numbers,
    )

    return run_analyze_powers_coding(
        input_dir=input_dir,
        output_dir=output_dir,
        exclude_participants=exclude_participants,
    )
