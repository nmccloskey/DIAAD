# `vocab analyze` Usage Guide

Use `diaad vocab analyze` when input samples are ready to be scored against the active target-vocabulary resources.

## Input Priority

Before dispatch, the current CLI requires transcript tables to be available or auto-generated. After that prerequisite check, the analysis function uses the following input priority.

DIAAD looks first for one workbook matching:

```text
unblind_utterance_data*.xlsx
```

This is the preferred path when earlier workflow steps have produced a combined utterance-level analysis file that includes unblinded sample identifiers, stimulus labels, utterances, speaking time, and optional filtering columns such as `c2_cu` or `word_count`.

If no unblinded utterance data file is found, DIAAD falls back to the configured transcript table file:

```text
transcript_tables.xlsx
```

## Required Fields

The input must contain the configured sample identifier column. The default is:

```text
sample_id
```

The input must also contain a stimulus or narrative column whose values match active resource IDs. If `project.stimulus_column` is configured, DIAAD uses that column. Otherwise it tries legacy names:

```text
narrative
scene
story
stimulus
```

Transcript-table fallback also requires an utterance column and a speaking-time column. Speaking-time candidates include:

```text
speaking_time
client_time
speech_time
time_s
time_sec
time_seconds
```

## Resource Matching

Only rows whose stimulus or narrative value matches an active resource ID are analyzed. Built-in IDs include:

```text
BrokenWindow
CatRescue
Cinderella
RefusedUmbrella
Sandwich
```

Custom resource IDs are active when `advanced.target_vocabulary_resource_path` points to a valid resource file or directory.

## Filtering And Token Matching

Rows with speaker labels listed in `project.exclude_speakers` are removed when a `speaker` column is present.

In unblinded input mode, DIAAD also looks for a `c2_cu` column or a `word_count` column and removes rows where that selected filter column is missing. If neither column is present, analysis continues without that filtering step.

Utterance text is normalized before matching. The formatter expands contractions, converts standalone digits to words, handles selected CHAT replacement patterns, removes common annotation containers, and drops unintelligible markers.

## Output Workbook

The `summary` sheet includes one row per sample, with fields such as:

```text
num_tokens
num_base_forms_produced
num_core_token_matches
lexicon_coverage
core_tokens_per_min
accuracy_pwa_percentile
accuracy_control_percentile
efficiency_pwa_percentile
efficiency_control_percentile
```

The `details` sheet includes one row per sample and base form, with the number of matched tokens and a binary score for whether that base form appeared.

## Common Problems

If no rows are produced, compare input stimulus values with active resource IDs.

If percentile columns are blank, the resource may have no norms, the sample may have no speaking time for efficiency norms, or DIAAD may not have been able to load the declared norm tables.

If coverage is unexpectedly low, inspect the `details` sheet and the resource `variant_map`.
