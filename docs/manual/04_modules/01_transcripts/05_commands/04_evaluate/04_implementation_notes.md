# `transcripts evaluate` Implementation Notes

`transcripts evaluate` dispatches to `evaluate_transcription_reliability()`.

## Dispatch Path

The main path is:

1. `src/diaad/cli/commands.py` registers `transcripts evaluate`.
2. `src/diaad/cli/dispatch.py` dispatches it without requiring transcript tables.
3. `src/diaad/core/run_context.py` threads metadata fields, text-processing settings, `reliability_tag`, and `reliability_dirname`.
4. `src/diaad/core/run_wrappers.py` calls `evaluate_transcription_reliability()`.
5. `src/diaad/transcripts/transcription_reliability_evaluation.py` matches files, processes text, computes metrics, and writes outputs.

## Reliability File Handling

The evaluator recursively scans the input directory for `.cha` files. Files with the configured reliability tag in the stem are treated as reliability transcripts. Files without the tag are treated as originals, except for untagged reliability files discovered under the configured reliability directory.

For untagged files under the reliability directory, the evaluator writes tagged copies under:

```text
reliability/renamed/
```

and evaluates the tagged copies. The original untagged reliability files are excluded from the original-file set.

## Pair Matching

Originals and reliability files are indexed by metadata-value tuples derived from configured metadata fields. Duplicate keys are logged and skipped after the first file for that key. Reliability files without a matching original are logged and skipped.

## Text Processing

Text processing uses `pylangacq.Reader` to extract utterance text. It can exclude speaker tiers, process CLAN correction notation, strip CLAN markup, normalize whitespace, and lowercase text.

The main settings are:

- `project.exclude_speakers`
- `project.strip_clan`
- `project.prefer_correction`
- `project.lowercase`

## Outputs

The evaluator writes:

```text
transcription_reliability_evaluation/transcription_reliability_evaluation.xlsx
transcription_reliability_evaluation/transcription_reliability_report.txt
transcription_reliability_evaluation/global_alignments/*_transcription_reliability_alignment.txt
```

Workbook metrics include token and character count differences, Levenshtein distance/similarity, and Needleman-Wunsch score/normalized score.

## Relevant Sources

- `src/diaad/transcripts/transcription_reliability_evaluation.py`
- `src/diaad/core/run_context.py`
- `src/diaad/core/run_wrappers.py`
- `src/diaad/cli/dispatch.py`
