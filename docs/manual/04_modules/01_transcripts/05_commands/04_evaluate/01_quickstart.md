# `transcripts evaluate` Quickstart

`diaad transcripts evaluate` compares original CHAT transcripts with reliability CHAT transcripts and writes transcription reliability metrics.

## Run

```bash
diaad transcripts evaluate --config config
```

## Minimum Inputs

Place original and reliability `.cha` files under the configured input directory. DIAAD can identify reliability files in either of two ways:

- by filename tag, such as `_reliability`; or
- by placing reliability transcripts in a directory named `reliability`.

The default settings are:

```yaml
reliability_tag: _reliability
reliability_dirname: reliability
```

## Primary Outputs

By default, the command writes:

```text
diaad_data/output/diaad_YYMMDD_HHMM/transcription_reliability_evaluation/
  transcription_reliability_evaluation.xlsx
  transcription_reliability_report.txt
  global_alignments/
```

## Immediate Next Step

Review the workbook and report together. Use the alignment files for cases that need manual inspection; they show how each matched original/reliability pair was aligned.

## Read Next

- Research context: `docs/manual/04_modules/01_transcripts/05_commands/04_evaluate/03_research_context.md`
- Usage guide: `docs/manual/04_modules/01_transcripts/05_commands/04_evaluate/02_usage_guide.md`
- Transcripts module quickstart: `docs/manual/04_modules/01_transcripts/01_quickstart.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
