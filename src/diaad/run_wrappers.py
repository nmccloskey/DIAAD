def run_analyze_digital_convo_turns(input_dir, output_dir):
    from diaad.convo_turns.digital_convo_turns_analysis import analyze_digital_convo_turns
    analyze_digital_convo_turns(input_dir=input_dir, output_dir=output_dir)

def run_make_POWERS_coding_files(tiers, frac, coders, input_dir, output_dir, exclude_participants, automate_POWERS=True):
    from diaad.POWERS.POWERS_coding_files import make_POWERS_coding_files
    make_POWERS_coding_files(
        tiers=tiers,
        frac=frac,
        coders=coders,
        input_dir=input_dir,
        output_dir=output_dir,
        exclude_participants=exclude_participants,
        automate_POWERS=automate_POWERS
    )

def run_analyze_POWERS_coding(input_dir, output_dir, reliability=False, just_c2_POWERS=False, exclude_participants=[]):
    from diaad.POWERS.POWERS_coding_analysis import analyze_POWERS_coding
    analyze_POWERS_coding(
        input_dir=input_dir,
        output_dir=output_dir,
        reliability=reliability,
        just_c2_POWERS=just_c2_POWERS, exclude_participants=exclude_participants)

def run_evaluate_POWERS_reliability(input_dir, output_dir):
    from diaad.POWERS.POWERS_coding_analysis import match_reliability_files, analyze_POWERS_coding
    match_reliability_files(input_dir=input_dir, output_dir=output_dir)
    analyze_POWERS_coding(input_dir=input_dir, output_dir=output_dir, reliability=True, just_c2_POWERS=False)

def run_reselect_POWERS_reliability_coding(input_dir, output_dir, frac, exclude_participants, automate_POWERS):
    from diaad.POWERS.POWERS_coding_files import reselect_POWERS_reliability
    reselect_POWERS_reliability(
        input_dir=input_dir,
        output_dir=output_dir,
        frac=frac,
        exclude_participants=exclude_participants,
        automate_POWERS=automate_POWERS)
