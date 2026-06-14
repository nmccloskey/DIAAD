# `transcripts evaluate` Usage Guide

Use `diaad transcripts evaluate` after reliability transcripts have been completed. The command compares each reliability transcript with its corresponding original transcript.

## Input Layouts

The command supports tagged reliability files:

```text
diaad_data/input/
  chat/
    P1_picnic_pre.cha
    P1_picnic_pre_reliability.cha
```

It also supports a reliability subdirectory:

```text
diaad_data/input/
  chat/
    P1_picnic_pre.cha
    reliability/
      P1_picnic_pre.cha
```

When reliability files are in the reliability subdirectory without the tag, DIAAD creates renamed tagged copies under a `renamed` subdirectory so the matching logic can compare them with originals.

## Matching Rules

DIAAD matches reliability transcripts to originals by configured metadata fields. If no metadata fields are configured, matching falls back to file-stem-like values.

For reliable matching, configure metadata fields so each original/reliability pair resolves to the same metadata values and each pair is unique.

## Text Processing Settings

The comparison uses the following project and advanced settings:

| Setting | Default | Effect |
|---|---|---|
| `project.exclude_speakers` | `[]` | Speaker tier labels to omit before comparison. |
| `project.strip_clan` | `true` | Whether to remove CLAN markup before comparison. |
| `project.prefer_correction` | `true` | Whether to prefer corrected forms in CLAN correction notation. |
| `project.lowercase` | `true` | Whether to lowercase text before comparison. |
| `advanced.reliability_tag` | `_reliability` | Filename tag used to identify reliability transcripts. |
| `advanced.reliability_dirname` | `reliability` | Directory name used for untagged reliability transcript inputs. |

These settings affect the text that is compared, so keep them stable across reliability runs.

## Output Metrics

The workbook includes one row per matched pair, with columns for:

- configured metadata values;
- original and reliability filenames;
- token counts and percent difference;
- character counts and percent difference;
- Levenshtein distance and similarity;
- Needleman-Wunsch global alignment score and normalized score.

The text report summarizes Levenshtein similarity across all evaluated pairs. The `global_alignments/` directory contains pair-specific alignment text files for closer review.

## Common Problems

If no pairs are matched, check metadata fields, reliability tags, and the reliability directory layout.

If duplicate metadata values are logged, the command keeps the first indexed file for that key and skips later duplicate files. Make metadata extraction more specific or remove duplicate inputs.

If scores look unexpectedly low, inspect the text processing settings and open the alignment file for that pair before treating the score as a substantive reliability conclusion.
