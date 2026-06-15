import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.stats import entropy
from collections import Counter, defaultdict
from diaad.metadata.discovery import find_one_matching_file, find_transcript_table
from diaad.transcripts.transcript_tables import extract_transcript_data
from psair.core.logger import logger, get_rel_path


def extract_turn_counts(turn_string):
    """
    Parse a coded turns string into per-speaker total turns, ignoring dot markers.

    Parameters
    ----------
    turn_string : str
        String with coded turns, digits represent speakers and optional dots
        represent markers. Example: "0.1..12.0".

    Returns
    -------
    collections.Counter
        Counts keyed by speaker ID (as str).

    Examples
    --------
    > extract_turn_counts("0.1..12.0")
    Counter({'0': 2, '1': 3, '2': 1})
    """
    matches = re.findall(r'(\d)', str(turn_string))
    return Counter(matches)

def extract_turn_stats(turn_string):
    """
    Parse a coded turns string into per-speaker totals and dot-mark counts.

    Parameters
    ----------
    turn_string : str
        String of coded turns, digits are speakers, dots are markers.

    Returns
    -------
    tuple of Counter
        (turn_counts, mark1_counts, mark2_counts), each a Counter keyed by speaker.

    Examples
    --------
    > extract_turn_stats("0.1..12.0")
    (Counter({'0': 2, '1': 3, '2': 1}),
     Counter({'1': 1}),
     Counter({'1': 1}))
    """
    matches = re.findall(r'(\d)(\.{1,2})?', str(turn_string))
    turn_counts = Counter()
    mark1_counts = Counter()
    mark2_counts = Counter()
    for speaker, dots in matches:
        turn_counts[speaker] += 1
        if dots == '.':
            mark1_counts[speaker] += 1
        elif dots == '..':
            mark2_counts[speaker] += 1
    return turn_counts, mark1_counts, mark2_counts


def mean_absolute_change(series):
    """Return mean absolute change between consecutive elements of a numeric series."""
    return np.mean(np.abs(np.diff(series)))

def _excluded_speaker_token(exclude_speakers=None) -> str:
    speakers = [str(speaker) for speaker in (exclude_speakers or []) if str(speaker)]
    return speakers[0] if speakers else "0"


def _normalize_speaker_token(speaker, exclude_speakers=None) -> str:
    mapped, _reason = _map_speaker_label(speaker, exclude_speakers)
    return mapped


def _map_speaker_label(speaker, exclude_speakers=None) -> tuple[str, str]:
    token = str(speaker).strip()
    excluded_token = _excluded_speaker_token(exclude_speakers)
    excluded = {str(item).strip() for item in (exclude_speakers or [])}
    if token == "0" and exclude_speakers:
        return excluded_token, "dct_zero_to_excluded_speaker"
    if token in excluded:
        if token == excluded_token:
            return excluded_token, "excluded_speaker_anchor"
        return excluded_token, "excluded_speaker_pooled"
    return token, "identity"


def _speaker_mapping_from_events(events):
    columns = ["source", "original_speaker", "mapped_speaker", "mapping_reason"]
    if events.empty or "original_speaker" not in events.columns:
        return pd.DataFrame(columns=columns)

    mapping = (
        events[columns]
        .drop_duplicates()
        .sort_values(["source", "original_speaker", "mapped_speaker"], kind="mergesort")
        .reset_index(drop=True)
    )
    return mapping


def _log_speaker_mapping(mapping):
    if mapping.empty:
        logger.info("No speaker labels were available to map for DCT analysis.")
        return

    remapped = mapping[mapping["original_speaker"] != mapping["mapped_speaker"]]
    if remapped.empty:
        logger.info(
            "DCT speaker label mapping retained %d speaker label(s) unchanged.",
            len(mapping),
        )
        return

    pairs = ", ".join(
        f"{row.original_speaker}->{row.mapped_speaker}"
        for row in remapped.itertuples(index=False)
    )
    logger.info("DCT speaker label mapping applied: %s.", pairs)


def clinician_to_participant_ratio(group, disinterest_speaker: str = "0"):
    """
    Compute the clinician-to-participant turn ratio within a group.

    Parameters
    ----------
    group : pandas.DataFrame
        Must contain 'speaker' and 'turns' columns. The configured
        disinterest speaker is treated as the clinician/non-client category.

    Returns
    -------
    float
        Ratio of clinician turns to participant turns, or NaN if denominator is zero.
    """
    speaker_turns = group.groupby('speaker')['turns'].sum()
    clinician_turns = speaker_turns.get(disinterest_speaker, 0)
    participant_turns = speaker_turns.drop(disinterest_speaker, errors='ignore').sum()
    return clinician_turns / participant_turns if participant_turns > 0 else np.nan

def compute_speaker_level(df):
    """
    Aggregate turns and markers at the speaker level.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain at least ['group', 'speaker', 'turns'] and optionally
        ['mark1','mark2','session','bin'].

    Returns
    -------
    pandas.DataFrame
        One row per speaker per group with totals, proportions, and optional
        session/bin counts.
    """
    agg_dict = {
        'turns': 'sum',
        'mark1': 'sum' if 'mark1' in df.columns else 'sum',
        'mark2': 'sum' if 'mark2' in df.columns else 'sum',
    }

    if 'session' in df.columns:
        agg_dict['session'] = pd.Series.nunique
    if 'bin' in df.columns:
        agg_dict['bin'] = 'count'

    speaker_level = (
        df.groupby(['group', 'speaker'], as_index=False)
        .agg(agg_dict)
        .rename(columns={
            'turns': 'total_turns',
            'mark1': 'mark1',
            'mark2': 'mark2',
            'session': 'unique_sessions' if 'session' in df.columns else None,
            'bin': 'bins_appeared_in' if 'bin' in df.columns else None
        })
    )

    # Proportions of dot-marked turns per speaker
    speaker_level['prop_mark1'] = speaker_level['mark1'] / speaker_level['total_turns']
    speaker_level['prop_mark2'] = speaker_level['mark2'] / speaker_level['total_turns']

    return speaker_level.loc[:, ~speaker_level.columns.str.match(r'^None$')]
    
def compute_group_level(df):
    """
    Aggregate turns and markers at the group level.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'group', 'turns', 'speaker', and optionally 'mark1',
        'mark2', 'session', 'bin'.

    Returns
    -------
    pandas.DataFrame
        One row per group with totals, number of participants/sessions, and
        proportions of marked turns.
    """
    group_agg = {
        'turns': 'sum',
        'mark1': 'sum' if 'mark1' in df.columns else 'sum',
        'mark2': 'sum' if 'mark2' in df.columns else 'sum',
        'speaker': pd.Series.nunique,
    }

    if 'bin' in df.columns:
        group_agg['bin'] = 'count'

    if 'session' in df.columns:
        group_agg['session'] = pd.Series.nunique

    group_level = (
        df.groupby('group', as_index=False)
        .agg(group_agg)
        .rename(columns={
            'turns': 'total_turns',
            'mark1': 'total_mark1',
            'mark2': 'total_mark2',
            'session': 'num_sessions',
            'speaker': 'num_participants',
            'bin': 'bins_covered' if 'bin' in df.columns else None
        })
    )

    # Proportions at the group level (of all turns)
    group_level['prop_mark1'] = group_level['total_mark1'] / group_level['total_turns']
    group_level['prop_mark2'] = group_level['total_mark2'] / group_level['total_turns']

    # Drop columns with meaningless names if they exist
    return group_level.loc[:, ~group_level.columns.str.match(r'^None$')]

def compute_bin_level(df, grouping_cols):
    """
    Compute turn and marker proportions at the bin level.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'turns', 'mark1', 'mark2' and grouping identifiers.
    grouping_cols : list of str
        Columns used to group bins (e.g. ['group','session','bin']).

    Returns
    -------
    pandas.DataFrame
        Original dataframe with added columns:
        - proportion_of_bin_turns
        - prop_mark1 / prop_mark2
        - proportion_of_bin_mark1 / mark2 (if available)
    """
    bin_level = df.copy()
    # Totals per bin grouping (e.g., ['group','session','bin'])
    bin_totals_turns = df.groupby(grouping_cols)['turns'].transform('sum')
    bin_level['proportion_of_bin_turns'] = bin_level['turns'] / bin_totals_turns

    # Per-row proportions (marked turns within this speaker/bin segment)
    bin_level['prop_mark1'] = bin_level['mark1'] / bin_level['turns'].replace(0, pd.NA)
    bin_level['prop_mark2'] = bin_level['mark2'] / bin_level['turns'].replace(0, pd.NA)

    # Share of the bin's total marked turns (optional but useful)
    if 'mark1' in bin_level.columns:
        bin_totals_mark1 = df.groupby(grouping_cols)['mark1'].transform('sum').replace(0, pd.NA)
        bin_level['proportion_of_bin_mark1'] = bin_level['mark1'] / bin_totals_mark1
    if 'mark2' in bin_level.columns:
        bin_totals_mark2 = df.groupby(grouping_cols)['mark2'].transform('sum').replace(0, pd.NA)
        bin_level['proportion_of_bin_mark2'] = bin_level['mark2'] / bin_totals_mark2

    sort_cols = [col for col in [*grouping_cols, 'speaker'] if col in bin_level.columns]
    return bin_level.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)

def compute_session_level(turn_totals, disinterest_speaker: str = "0"):
    """
    Summarize turn-taking metrics at the session level.

    Parameters
    ----------
    turn_totals : pandas.DataFrame
        Must contain ['session','group','speaker','turns','mark1','mark2'].

    Returns
    -------
    pandas.DataFrame
        One row per session-group with totals, entropy of speaker turns,
        clinician-to-participant ratio, and marker proportions.
    """
    session_summary = (
        turn_totals.groupby(['session', 'group'])
        .agg(
            total_turns=('turns', 'sum'),
            total_mark1=('mark1', 'sum'),
            total_mark2=('mark2', 'sum'),
        )
        .reset_index()
    )

    entropy_data = (
        turn_totals.groupby(['session', 'group', 'speaker'])['turns'].sum()
        .groupby(['session', 'group'])
        .apply(lambda x: entropy(x), include_groups=False)
        .reset_index(name='turn_entropy')
    )

    ratios = (
        turn_totals.groupby(['session', 'group'])
        .apply(
            lambda group: clinician_to_participant_ratio(
                group,
                disinterest_speaker=disinterest_speaker,
            ),
            include_groups=False,
        )
        .reset_index(name='clinician_participant_ratio')
    )

    session_summary = session_summary.merge(entropy_data, on=['session', 'group'], how='left')
    session_summary = session_summary.merge(ratios, on=['session', 'group'], how='left')

    # Proportions of marked turns at session level
    session_summary['prop_mark1'] = session_summary['total_mark1'] / session_summary['total_turns']
    session_summary['prop_mark2'] = session_summary['total_mark2'] / session_summary['total_turns']

    return session_summary

def compute_participation_level(turn_totals, has_bin=False):
    """
    Summarize metrics at the participant-session level.

    Parameters
    ----------
    turn_totals : pandas.DataFrame
        Must include ['group','session','speaker','turns','mark1','mark2'].
    has_bin : bool, default False
        If True, include bin-level variability (mean, std, var, CV, change).

    Returns
    -------
    pandas.DataFrame
        Rows for each speaker-session with totals, proportions of session
        turns, marker stats, and optional bin statistics.
    """
    
    participation_level = (
        turn_totals.groupby(['group', 'session', 'speaker'], as_index=False)
        .agg({'turns': 'sum', 'mark1': 'sum', 'mark2': 'sum'})
        .rename(columns={'turns': 'total_turns', 'mark1': 'total_mark1', 'mark2': 'total_mark2'})
    )
    session_totals = participation_level.groupby(['session', 'group'])['total_turns'].transform('sum')
    participation_level['proportion_of_session_turns'] = participation_level['total_turns'] / session_totals

    # Per-speaker proportions of their own marked turns
    participation_level['prop_mark1'] = participation_level['total_mark1'] / participation_level['total_turns'].replace(0, pd.NA)
    participation_level['prop_mark2'] = participation_level['total_mark2'] / participation_level['total_turns'].replace(0, pd.NA)

    # Bin stats per participant-session
    if has_bin:
        bin_stats = (
            turn_totals.groupby(['session', 'group', 'speaker'])
            .agg(
                mean_turns=('turns', 'mean'),
                std_turns=('turns', 'std'),
                var_turns=('turns', 'var'),
                min_turns=('turns', 'min'),
                max_turns=('turns', 'max'),
                mean_mark1=('mark1', 'mean'),
                std_mark1=('mark1', 'std'),
                var_mark1=('mark1', 'var'),
                mean_mark2=('mark2', 'mean'),
                std_mark2=('mark2', 'std'),
                var_mark2=('mark2', 'var'),
            )
            .reset_index()
        )
        bin_stats['cv_turns'] = bin_stats['std_turns'] / bin_stats['mean_turns']

        changes = (
            turn_totals.sort_values(['session', 'group', 'speaker', 'bin'])
            .groupby(['session', 'group', 'speaker'])['turns']
            .agg(avg_change_turns=mean_absolute_change)
            .reset_index()
        )

        participation_level = participation_level.merge(bin_stats, on=['session', 'group', 'speaker'], how='left')
        participation_level = participation_level.merge(changes, on=['session', 'group', 'speaker'], how='left')
    
    return participation_level

# --- Transition Matrices and Ratios ---
def extract_sequence(turn_string):
    if isinstance(turn_string, (list, tuple, pd.Series)):
        return [str(token) for token in turn_string]
    return re.findall(r'\d', str(turn_string))

def build_transition_matrix(sequences):
    """
    Construct normalized transition matrix from turn sequences.

    Parameters
    ----------
    sequences : list of list of str
        Each sublist is a sequence of speaker IDs.

    Returns
    -------
    pandas.DataFrame
        Square matrix with P(to_speaker | from_speaker) probabilities.
    """
    transition_counts = defaultdict(lambda: defaultdict(int))
    for seq in sequences:
        for i in range(len(seq) - 1):
            from_speaker = seq[i]
            to_speaker = seq[i + 1]
            transition_counts[from_speaker][to_speaker] += 1

    # Flatten speaker list and sort
    speakers = sorted(set(transition_counts) | {k for d in transition_counts.values() for k in d})
    matrix = pd.DataFrame(0, index=speakers, columns=speakers, dtype=int)

    for from_spk, to_dict in transition_counts.items():
        for to_spk, count in to_dict.items():
            matrix.loc[from_spk, to_spk] = count

    return matrix.div(matrix.sum(axis=1), axis=0).fillna(0)

def compute_transition_metrics(events, disinterest_speaker: str = "0"):
    """
    Compute transition matrices and speaker ratios for each group.

    Parameters
    ----------
    events : pandas.DataFrame
        Normalized turn events with 'group', 'speaker', and sequence columns.

    Returns
    -------
    dict
        {
            'transition_matrices': {group_id: pandas.DataFrame},
            'speaker_ratios': pandas.DataFrame with per-group ratios
        }
    """
    speaker_matrices = {}
    speaker_ratios = []

    for group, group_df in events.groupby('group'):
        sequence_group_cols = [
            col for col in ['group', 'session', 'bin'] if col in group_df.columns
        ]
        sequences = []
        for _, sequence_df in group_df.groupby(sequence_group_cols, dropna=False):
            sequence = (
                sequence_df.sort_values('sequence_position', kind='mergesort')['speaker']
                .astype(str)
                .tolist()
            )
            if sequence:
                sequences.append(sequence)
        if not sequences:
            continue

        matrix = build_transition_matrix(sequences)
        speaker_matrices[str(group)] = matrix

        # Compute ratios
        speakers = matrix.columns.astype(str)
        participants = [s for s in speakers if s != disinterest_speaker]
        ptp = matrix.loc[participants, participants].to_numpy().sum() if participants else np.nan
        ptc = (
            matrix.loc[participants, disinterest_speaker].sum()
            if disinterest_speaker in matrix.columns
            else np.nan
        )
        cpp = (
            matrix.loc[disinterest_speaker, participants].sum()
            if disinterest_speaker in matrix.index
            else np.nan
        )

        speaker_ratios.append({
            'group': group,
            'participant_to_participant': ptp,
            'participant_to_clinician': ptc,
            'clinician_to_participant': cpp
        })

    return {
        'transition_matrices': speaker_matrices,
        'speaker_ratios': pd.DataFrame(speaker_ratios)
    }


def _events_from_dct_coding(
    df,
    *,
    sample_id_field: str = "sample_id",
    exclude_speakers=None,
):
    df = df.copy()
    if 'group' not in df.columns and sample_id_field in df.columns:
        df = df.rename(columns={sample_id_field: 'group'})

    required_cols = ['group', 'turns']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    has_session = 'session' in df.columns
    has_bin = 'bin' in df.columns
    rows = []
    for _, row in df.iterrows():
        if pd.isna(row['turns']):
            continue
        sequence_position = 0
        for match in re.finditer(r'(\d)(\.{1,2})?', str(row['turns'])):
            sequence_position += 1
            dots = match.group(2) or ""
            original_speaker = match.group(1)
            mapped_speaker, mapping_reason = _map_speaker_label(
                original_speaker,
                exclude_speakers,
            )
            event = {
                'sample_id': row.get('group'),
                'group': row.get('group'),
                'speaker': mapped_speaker,
                'original_speaker': original_speaker,
                'mapped_speaker': mapped_speaker,
                'mapping_reason': mapping_reason,
                'sequence_position': sequence_position,
                'mark1': 1 if dots == '.' else 0,
                'mark2': 1 if dots == '..' else 0,
                'source': 'dct_coding',
            }
            if has_session:
                event['session'] = row.get('session')
            if has_bin:
                event['bin'] = row.get('bin')
            rows.append(event)

    return pd.DataFrame(rows), has_session, has_bin


def _events_from_transcript_rows(
    df,
    *,
    sample_id_field: str = "sample_id",
    exclude_speakers=None,
):
    required_cols = [sample_id_field, 'speaker']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    transcript_df = df.copy()
    transcript_df['_source_row_order'] = range(len(transcript_df))
    sort_cols = [
        col
        for col in [sample_id_field, 'session', 'position', 'position_sub', '_source_row_order']
        if col in transcript_df.columns
    ]
    transcript_df = transcript_df.sort_values(sort_cols, kind='mergesort')
    has_session = 'session' in transcript_df.columns

    rows = []
    grouping_cols = [sample_id_field]
    if has_session:
        grouping_cols.append('session')
    for _, group_df in transcript_df.groupby(grouping_cols, dropna=False):
        for sequence_position, (_, row) in enumerate(group_df.iterrows(), start=1):
            if pd.isna(row.get('speaker')):
                continue
            original_speaker = str(row.get('speaker')).strip()
            mapped_speaker, mapping_reason = _map_speaker_label(
                original_speaker,
                exclude_speakers,
            )
            event = {
                'sample_id': row.get(sample_id_field),
                'group': row.get(sample_id_field),
                'speaker': mapped_speaker,
                'original_speaker': original_speaker,
                'mapped_speaker': mapped_speaker,
                'mapping_reason': mapping_reason,
                'sequence_position': sequence_position,
                'mark1': 0,
                'mark2': 0,
                'source': 'transcript_table',
            }
            if has_session:
                event['session'] = row.get('session')
            rows.append(event)

    return pd.DataFrame(rows), has_session, False


def _events_from_transcript_table(
    transcript_table,
    *,
    sample_id_field: str = "sample_id",
    exclude_speakers=None,
):
    transcript_df = extract_transcript_data(
        transcript_table,
        kind='joined',
        sample_id_field=sample_id_field,
    )
    return _events_from_transcript_rows(
        transcript_df,
        sample_id_field=sample_id_field,
        exclude_speakers=exclude_speakers,
    )


def _turn_totals_from_events(events, *, has_session=False, has_bin=False):
    grouping_cols = ['group']
    if has_session and 'session' in events.columns:
        grouping_cols.append('session')
    grouping_cols.append('speaker')
    if has_bin and 'bin' in events.columns:
        grouping_cols.append('bin')

    turn_totals = (
        events.groupby(grouping_cols, dropna=False)
        .agg(
            turns=('speaker', 'size'),
            mark1=('mark1', 'sum'),
            mark2=('mark2', 'sum'),
        )
        .reset_index()
    )
    return turn_totals


def _empty_analysis_result():
    return {
        'speaker_level': pd.DataFrame(),
        'group_level': pd.DataFrame(),
        'bin_level': pd.DataFrame(),
        'session_level': pd.DataFrame(),
        'participation_level': pd.DataFrame(),
        'transition_matrices': {},
        'speaker_ratios': pd.DataFrame(),
    }


def _analyze_turn_events(
    events,
    *,
    has_session=False,
    has_bin=False,
    disinterest_speaker: str = "0",
):
    if events.empty:
        logger.warning("No conversation turn events available for analysis.")
        return _empty_analysis_result()

    speaker_mapping = _speaker_mapping_from_events(events)
    _log_speaker_mapping(speaker_mapping)

    turn_totals = _turn_totals_from_events(
        events,
        has_session=has_session,
        has_bin=has_bin,
    )

    ct_data = {
        'speaker_level': compute_speaker_level(turn_totals),
        'group_level': compute_group_level(turn_totals),
        'speaker_label_mapping': speaker_mapping,
    }

    if has_bin:
        bin_grouping = ['group']
        if has_session:
            bin_grouping.append('session')
        bin_grouping.append('bin')
        turn_totals = compute_bin_level(turn_totals, grouping_cols=bin_grouping)
        ct_data['bin_level'] = turn_totals

    if has_session:
        ct_data['session_level'] = compute_session_level(
            turn_totals,
            disinterest_speaker=disinterest_speaker,
        )
        ct_data['participation_level'] = compute_participation_level(
            turn_totals,
            has_bin=has_bin,
        )

    ct_data.update(
        compute_transition_metrics(
            events,
            disinterest_speaker=disinterest_speaker,
        )
    )
    return ct_data


def _analyze_convo_turns_file(
    df,
    sample_id_field: str = "sample_id",
    exclude_speakers=None,
):
    """
    Analyze a single conversation turns dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain at least ['group','turns'] plus optional ['session','bin'].

    Returns
    -------
    dict
        {
            'speaker_level': DataFrame,
            'group_level': DataFrame,
            'bin_level': DataFrame (if applicable),
            'session_level': DataFrame (if applicable),
            'participation_level': DataFrame (if applicable),
            'transition_matrices': dict,
            'speaker_ratios': DataFrame
        }
    """
    events, has_session, has_bin = _events_from_dct_coding(
        df,
        sample_id_field=sample_id_field,
        exclude_speakers=exclude_speakers,
    )
    return _analyze_turn_events(
        events,
        has_session=has_session,
        has_bin=has_bin,
        disinterest_speaker=_excluded_speaker_token(exclude_speakers),
    )

# Summary statistics with coefficient of variation
def summarize(df, level_name):
    """
    Compute summary statistics with coefficient of variation for a level.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain numeric columns to summarize.
    level_name : str
        Name of the level (e.g. 'speaker','group') for labeling.

    Returns
    -------
    pandas.DataFrame
        Summary statistics (mean, std, min, max, cv) for each metric.
    """
    numeric = df.select_dtypes(include=[np.number])
    summary = numeric.agg(['mean', 'std', 'min', 'max']).transpose()
    summary['cv'] = summary['std'] / summary['mean']
    summary.reset_index(inplace=True)
    summary.columns = ['metric', 'mean', 'std', 'min', 'max', 'cv']
    summary.insert(0, 'level', level_name)
    return summary

def write_if_not_empty(df, writer, sheet_name):
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_excel(writer, index=False, sheet_name=sheet_name)

def analyze_digital_convo_turns(
    input_dir,
    output_dir,
    sample_id_field: str = "sample_id",
    dct_coding_filename: str = "conversation_turns.xlsx",
    transcript_table_filename: str = "transcript_tables.xlsx",
    exclude_speakers=None,
    use_transcript_tables: bool = False,
):
    """
    Run full analysis pipeline on conversation turn files.

    Parameters
    ----------
    input_dir : str or Path
        Directory to search for the configured conversation-turns coding file
        or transcript table.
    output_dir : str or Path
        Directory where analysis Excel outputs are written.

    Returns
    -------
    None
        Writes one *_Analysis.xlsx per input file, containing:
        - Turn counts (speaker/group/bin/session levels)
        - Participation summaries
        - Transition matrices and ratios
        - Summary statistics
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if use_transcript_tables:
        source_file = find_transcript_table(
            directories=[input_dir, output_dir],
            filename=transcript_table_filename,
            required=True,
            label="transcript table file for conversation turns analysis",
        )
        source_label = "transcript table"
        out_file_name = source_file.name.replace('.xlsx', '_turns_analysis.xlsx')
    else:
        source_file = find_one_matching_file(
            directories=[input_dir, output_dir],
            filename=dct_coding_filename,
            label="conversation turns coding file",
        )
        source_label = "conversation turns coding file"
        out_file_name = source_file.name.replace('.xlsx', '_analysis.xlsx')

    logger.info(f"Found {source_label}: {get_rel_path(source_file)}.")

    for ct_file in tqdm([source_file], desc="Analyzing conversation turns"):
        try:
            if use_transcript_tables:
                events, has_session, has_bin = _events_from_transcript_table(
                    ct_file,
                    sample_id_field=sample_id_field,
                    exclude_speakers=exclude_speakers,
                )
                ct_data = _analyze_turn_events(
                    events,
                    has_session=has_session,
                    has_bin=has_bin,
                    disinterest_speaker=_excluded_speaker_token(exclude_speakers),
                )
            else:
                xls = pd.ExcelFile(ct_file)
                if not xls.sheet_names:
                    logger.warning(f"No sheets found in file: {ct_file.name}")
                    continue
                df = xls.parse(xls.sheet_names[0])
                if df.empty:
                    logger.warning(f"Empty data in file: {ct_file.name}")
                    continue

                ct_data = _analyze_convo_turns_file(
                    df,
                    sample_id_field=sample_id_field,
                    exclude_speakers=exclude_speakers,
                )

            # Extract all data levels (with fallback to empty df)
            bin_level = ct_data.get('bin_level', pd.DataFrame())
            participation_level = ct_data.get('participation_level', pd.DataFrame())
            session_level = ct_data.get('session_level', pd.DataFrame())
            speaker_level = ct_data.get('speaker_level', pd.DataFrame())
            group_level = ct_data.get('group_level', pd.DataFrame())
            speaker_label_mapping = ct_data.get('speaker_label_mapping', pd.DataFrame())
            speaker_ratios = ct_data.get('speaker_ratios', pd.DataFrame())
            speaker_matrices = ct_data.get('transition_matrices', {})

            # Compile summary stats safely
            summary_levels = {
                'session': session_level,
                'participation': participation_level,
                'speaker': speaker_level,
                'group': group_level,
            }

            summary_frames = []
            for level_name, df_level in summary_levels.items():
                if isinstance(df_level, pd.DataFrame) and not df_level.empty:
                    try:
                        summary_frames.append(summarize(df_level, level_name))
                    except Exception as e:
                        logger.warning(f"Could not summarize {level_name} level: {e}")

            summary_stats_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()

            final_path = Path(output_dir, out_file_name)

            with pd.ExcelWriter(final_path, engine='xlsxwriter') as writer:
                write_if_not_empty(bin_level, writer, 'bin_level_turns')
                write_if_not_empty(participation_level, writer, 'participation_level_turns')
                write_if_not_empty(session_level, writer, 'session_level_summary')
                write_if_not_empty(speaker_level, writer, 'speaker_level_turns')
                write_if_not_empty(group_level, writer, 'group_level_summary')
                write_if_not_empty(summary_stats_all, writer, 'summary_statistics')
                write_if_not_empty(speaker_ratios, writer, 'speaker_level_ratios')
                write_if_not_empty(speaker_label_mapping, writer, 'speaker_label_mapping')

                for name, matrix in speaker_matrices.items():
                    sheet_name = f"speaker_matrix_{name}"[:31]  # Excel max sheet name = 31
                    matrix.to_excel(writer, sheet_name=sheet_name)

        except Exception as e:
            logger.error(f"Unexpected error with file {ct_file.name}: {e}")
