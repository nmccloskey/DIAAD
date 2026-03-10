from diaad.utils.logger import logger


# ------------------------------------------------------------------
# Config and initial inputs
# ------------------------------------------------------------------

def run_read_tiers(config):
    from diaad.utils.tiers import TierManager
    from diaad.utils.diaad_tier_adapter import adapt_tiers_for_diaad

    TM = TierManager(config)

    adapted_tiers = adapt_tiers_for_diaad(TM)

    if adapted_tiers:
        logger.info("Successfully parsed and adapted tiers for DIAAD.")
    else:
        logger.warning("Adapted tiers are empty or malformed.")

    return adapted_tiers, TM

def run_read_cha_files(input_dir, shuffle=False):
    from diaad.utils.cha_files import read_cha_files
    return read_cha_files(input_dir=input_dir, shuffle=shuffle)

# ------------------------------------------------------------------
# Transcription
# ------------------------------------------------------------------

def run_tabularize_transcripts(tiers, chats, output_dir, shuffle_samples, seed):
    from diaad.transcripts.transcript_tables import tabularize_transcripts
    return tabularize_transcripts(tiers=tiers, chats=chats, output_dir=output_dir, shuffle=shuffle_samples, random_seed=seed)

def run_select_transcription_reliability_samples(tiers, chats, frac, output_dir):
    from diaad.transcripts.transcription_reliability_selection import select_transcription_reliability_samples
    select_transcription_reliability_samples(tiers=tiers, chats=chats, frac=frac, output_dir=output_dir)

def run_reselect_transcription_reliability_samples(input_dir, output_dir, frac):
    from diaad.transcripts.transcription_reliability_selection import reselect_transcription_reliability_samples
    reselect_transcription_reliability_samples(input_dir, output_dir, frac)

def run_evaluate_transcription_reliability(tiers, input_dir, output_dir, exclude_participants, strip_clan, prefer_correction, lowercase):
    from diaad.transcripts.transcription_reliability_evaluation import evaluate_transcription_reliability
    evaluate_transcription_reliability(tiers=tiers, input_dir=input_dir, output_dir=output_dir, exclude_participants=exclude_participants, strip_clan=strip_clan, prefer_correction=prefer_correction, lowercase=lowercase)

# ------------------------------------------------------------------
# Complete Utterance coding
# ------------------------------------------------------------------

def run_make_cu_coding_files(tiers, frac, coders, input_dir, output_dir, cu_paradigms, exclude_participants):
    from diaad.coding.coding_files import make_cu_coding_files
    make_cu_coding_files(tiers=tiers, frac=frac, coders=coders, input_dir=input_dir, output_dir=output_dir, cu_paradigms=cu_paradigms, exclude_participants=exclude_participants)

def run_reselect_cu_reliability(tiers, input_dir, output_dir, rel_type, frac):
    from diaad.coding.coding_files import reselect_cu_wc_reliability
    reselect_cu_wc_reliability(tiers, input_dir, output_dir, rel_type, frac)

def run_evaluate_cu_reliability(tiers, input_dir, output_dir, cu_paradigms):
    from diaad.coding.cu_analysis import evaluate_cu_reliability
    evaluate_cu_reliability(tiers=tiers, input_dir=input_dir, output_dir=output_dir, cu_paradigms=cu_paradigms)

def run_analyze_cu_coding(tiers, input_dir, output_dir, cu_paradigms):
    from diaad.coding.cu_analysis import analyze_cu_coding
    analyze_cu_coding(tiers=tiers, input_dir=input_dir, output_dir=output_dir, cu_paradigms=cu_paradigms)

def run_summarize_cus(tiers, input_dir, output_dir, seed, TM):
    from diaad.coding.cu_summarization import summarize_cus
    summarize_cus(tiers=tiers, input_dir=input_dir, output_dir=output_dir, seed=seed, TM=TM)

# ------------------------------------------------------------------
# Manual word counting
# ------------------------------------------------------------------

def run_make_word_count_files(tiers, frac, coders, input_dir, output_dir):
    from diaad.coding.coding_files import make_word_count_files
    make_word_count_files(tiers=tiers, frac=frac, coders=coders, input_dir=input_dir, output_dir=output_dir)

def run_reselect_wc_reliability(tiers, input_dir, output_dir, rel_type, frac):
    from diaad.coding.coding_files import reselect_cu_wc_reliability
    reselect_cu_wc_reliability(tiers, input_dir, output_dir, rel_type, frac)

def run_evaluate_word_count_reliability(tiers, input_dir, output_dir):
    from diaad.coding.word_count_reliability_evaluation import evaluate_word_count_reliability
    evaluate_word_count_reliability(tiers=tiers, input_dir=input_dir, output_dir=output_dir)

# ------------------------------------------------------------------
# CoreLex - convenience layer
# ------------------------------------------------------------------

def run_run_corelex(tiers, input_dir, output_dir, exclude_participants):
    from diaad.coding.corelex import run_corelex
    run_corelex(tiers=tiers, input_dir=input_dir, output_dir=output_dir, exclude_participants=exclude_participants)

# ------------------------------------------------------------------
# Digital conversation turns
# ------------------------------------------------------------------

def run_analyze_digital_convo_turns(input_dir, output_dir):
    from diaad.coding.convo_turns.digital_convo_turns import analyze_digital_convo_turns
    analyze_digital_convo_turns(input_dir=input_dir, output_dir=output_dir)
