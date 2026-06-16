# Transcription Reliability Usage Guide

Transcription reliability is a staged workflow: select samples, create independent reliability transcripts, compare them with originals, and decide whether review or reselection is needed.

## Plan The Sampling Frame

Use `project.reliability_fraction` to set the selected fraction:

```yaml
reliability_fraction: 0.2
```

`transcripts select` prefers the configured transcript table when one is available. If no transcript table is found, it can select directly from CHAT files. Use the table-based path when the transcript table is already the project's current sample frame.

Inspect both sheets in the selection workbook:

- `reliability_selection`;
- `all_transcripts`.

The `all_transcripts` sheet helps document what was eligible and what was selected.

## Prepare Independent Transcripts

The reliability transcript should not simply be a lightly edited copy of the original transcript. It should be prepared independently enough to test the consistency of the transcription procedure.

DIAAD may write blank reliability CHAT files during selection when source CHAT objects are available. These files contain setup headers and need completed transcript content before evaluation.

## Match Originals And Reliability Files

`transcripts evaluate` supports two common layouts.

Tagged reliability files:

```text
chat/
  sample_001.cha
  sample_001_reliability.cha
```

Reliability subdirectory:

```text
chat/
  sample_001.cha
  reliability/
    sample_001.cha
```

When reliability files are in the reliability subdirectory, DIAAD can create tagged renamed copies under a `renamed` subdirectory.

For robust matching, configure metadata fields so each original and reliability transcript pair resolves to the same metadata values.

## Set Transcript Processing Deliberately

The following settings change the text being compared:

| Setting | Typical effect |
|---|---|
| `exclude_speakers` | Omits selected CHAT speaker tiers before comparison. |
| `strip_clan` | Removes CLAN markup before comparison. |
| `prefer_correction` | Uses corrected forms in CLAN correction notation. |
| `lowercase` | Ignores casing differences. |

Set these before evaluating. Changing them afterward can change reliability scores without changing the transcripts.

## Read Results

The results workbook includes token counts, character counts, percent differences, Levenshtein metrics, and Needleman-Wunsch metrics.

The text report summarizes Levenshtein similarity bands. Treat those bands as practical review aids rather than universal standards.

For low or surprising values, open the corresponding alignment file under:

```text
global_alignments/
```

Alignment files can show whether disagreement comes from broad transcript mismatch, repeated-word differences, nonword spelling, missing sections, speaker exclusion, or normalization choices.

## Reselect As A Fallback

Use:

```bash
diaad transcripts reselect
```

when another reliability round is needed. Reselection reads prior reliability selection workbooks, avoids already selected samples when possible, and writes a replacement subset.

Document reselection in the project record. It changes the reliability history and should not be hidden as if the first subset had passed unchanged.

## Read Next

- `transcripts evaluate` usage guide: `docs/manual/04_modules/01_transcripts/05_commands/04_evaluate/02_usage_guide.md`
- Reliability functionality: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/02_usage_guide.md`
- Run provenance: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/02_usage_guide.md`
