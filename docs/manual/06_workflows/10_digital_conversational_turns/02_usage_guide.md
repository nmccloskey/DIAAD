# Digital Conversational Turns Usage Guide

Digital Conversational Turns is a workflow for representing who takes turns, in what order, and with what marker annotations. It can be used before full transcription, instead of full transcription for some questions, or alongside transcript-table speaker-sequence analysis.

## Use DCT When

DCT is useful when a project needs:

- a lower-burden conversational dynamics measure;
- speaker participation summaries;
- transition probabilities between speakers;
- turn-sequence reliability evidence;
- a workflow that may be feasible before full transcription.

It is most informative in interactions with multiple speakers or participant categories. For dyadic clinician-client dialogs, a transcript-based POWERS workflow may answer different questions, and manual DCT may be less interesting unless turn dynamics are central.

## Manual Turn-String Workbooks

For manual DCT coding, prepare a primary workbook named by `advanced.dct_coding_filename`, which defaults to:

```text
conversation_turns.xlsx
```

For reliability evaluation, prepare a reliability workbook named by `advanced.dct_coding_reliability`, which defaults to:

```text
conversation_turns_reliability.xlsx
```

Required fields include the configured sample identifier and:

```text
turns
```

`session` and `bin` are recommended because they define more informative comparison and summary units.

## Turn-String Convention

Manual DCT strings use digits as speaker codes:

```text
0.1..23.0.12
```

By convention, `0` represents the clinician or other non-client interlocutor category. Digits `1` through `9` identify client or participant speakers.

Dots immediately following a digit are marker annotations:

```text
1.
1..
```

The parser is digit-by-digit. A string containing `10` is interpreted as speaker `1` followed by speaker `0`, not speaker ten.

## Evaluate Reliability

Run:

```bash
diaad turns evaluate
```

The results compare speaker counts and full turn-string sequences. Inspect count agreement, sequence similarity, and alignment files before deciding whether coders need retraining, adjudication, or a revised protocol.

## Analyze Manual DCT Workbooks

Run:

```bash
diaad turns analyze
```

The analysis can summarize speaker-level, group-level, bin-level, session-level, participation, ratio, and transition-matrix outputs when the relevant input columns are present.

## Analyze Transcript-Table Speaker Sequences

If no DCT workbook is found, `turns analyze` falls back to transcript tables. In this mode, each transcript-table speaker tag is treated as one sequence token. Speaker labels such as `INV`, `CHI`, or `MOT` remain intact.

Speakers listed in `project.exclude_speakers` are pooled into the non-client category rather than dropped. If any excluded speakers are configured, the first listed speaker becomes that category label.

Transcript fallback does not synthesize bins, so bin-level outputs are not produced unless the source contains bin information.

## Read Next

- `turns analyze` usage guide: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/04_analyze/02_usage_guide.md`
- `turns evaluate` usage guide: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/02_evaluate/02_usage_guide.md`
- Digital Conversational Turns research context: `docs/manual/04_modules/07_digital_conversational_turns/03_research_context.md`
- Transcript-based workflow baseline: `docs/manual/06_workflows/04_transcription_based_workflow_baseline/01_quickstart.md`
