# Transcription Reliability Quickstart

Use transcription reliability when a project needs evidence that transcripts are being produced consistently enough for downstream discourse analysis.

## Before Selection

Review project settings that affect transcript comparison:

```yaml
strip_clan: true
prefer_correction: true
lowercase: true
exclude_speakers: []
reliability_fraction: 0.2
```

Also confirm `metadata_fields`, because DIAAD uses metadata to match reliability transcripts with originals when possible.

## Select A Subset

Run:

```bash
diaad transcripts select
```

Inspect:

```text
transcription_reliability_selection/transcription_reliability_samples.xlsx
```

## Complete Reliability Transcripts

Prepare independent reliability transcripts for the selected samples. These should usually be created from source audio or video rather than by editing the original transcript.

## Evaluate

Run:

```bash
diaad transcripts evaluate
```

Inspect:

```text
transcription_reliability_evaluation/
  transcription_reliability_results.xlsx
  transcription_reliability_report.txt
  global_alignments/
```

## Reselect If Needed

Use reselection only when another reliability round is needed:

```bash
diaad transcripts reselect
```

## Read Next

- `transcripts select`: `docs/manual/04_modules/01_transcripts/05_commands/03_select/01_quickstart.md`
- `transcripts evaluate`: `docs/manual/04_modules/01_transcripts/05_commands/04_evaluate/01_quickstart.md`
- `transcripts reselect`: `docs/manual/04_modules/01_transcripts/05_commands/05_reselect/01_quickstart.md`
- Transcript text normalization: `docs/manual/05_functionalities/07_transcript_text_normalization_speaker_exclusion/01_quickstart.md`
