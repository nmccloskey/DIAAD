# Transcription-Based Workflow Baseline Usage Guide

The transcription-based baseline is the recommended shared starting point for DIAAD workflows that analyze transcript content. It turns a set of transcript files into a stable, human-editable table structure that later coding, reliability, blinding, and analysis steps can share.

## Stage 0: Prepare Transcripts

DIAAD assumes the initial transcription process happens outside the transcript table command. A project may use automatic transcription, CHAT-oriented tools, manual transcription, or some combination of those steps.

Before running downstream DIAAD commands, decide what counts as the current transcript version. For research workflows, it is usually worth documenting:

- who reviewed the transcripts;
- whether review happened in one or more rounds;
- which transcript conventions were used;
- whether speaker labels are stable;
- whether filenames contain metadata DIAAD should extract.

## Stage 1: Configure Transcript Processing

Before reliability evaluation, review the transcript-processing settings in project configuration:

```yaml
strip_clan: true
prefer_correction: true
lowercase: true
exclude_speakers: []
```

These settings affect transcript comparison and some transcript-derived outputs. More detail appears in Transcript Text Normalization and Speaker Exclusion (`docs/manual/05_functionalities/07_transcript_text_normalization_speaker_exclusion/02_usage_guide.md`).

Also check:

- `metadata_fields`, so samples can be matched and grouped correctly;
- `reliability_fraction`, so selection reflects the project plan;
- `random_seed`, so random selections are reproducible;
- `advanced.transcript_table_filename`, so downstream discovery is predictable.

## Stage 2: Select Reliability Samples

Run:

```bash
diaad transcripts select
```

The command can use an existing transcript table if one is available. If no transcript table is found, it can build the sample frame from CHAT files. In either case, inspect:

```text
transcription_reliability_selection/transcription_reliability_samples.xlsx
```

The `reliability_selection` sheet is the selected subset. The `all_transcripts` sheet is the sample frame with selection indicators.

## Stage 3: Complete Independent Reliability Transcription

Reliability transcription should be independent of the transcript being evaluated. In many projects, this means the reliability transcriber works from the source audio or video rather than editing the existing transcript.

When DIAAD can access the CHAT files during selection, it writes blank reliability CHAT files with headers. Those files are setup artifacts, not completed reliability transcripts.

## Stage 4: Evaluate Reliability

After reliability transcripts are complete, run:

```bash
diaad transcripts evaluate
```

Inspect the output workbook, text report, and global alignment files. If reliability is lower than expected, look at the pair-specific alignment file before deciding whether the issue is a transcription error, a protocol ambiguity, a normalization setting, or a matching problem.

Use `transcripts reselect` only when another reliability round is needed:

```bash
diaad transcripts reselect
```

Reselection is a fallback and should be documented as part of the reliability history.

## Stage 5: Tabularize The Stable Transcript Set

Once the transcript set is ready to become the canonical DIAAD representation, run:

```bash
diaad transcripts tabularize
```

The transcript table workbook becomes the shared input for most transcript-derived workflows:

```text
transcript_tables/transcript_tables.xlsx
```

The table is both computer-friendly and human-editable. It follows database logic through stable identifiers, but it remains an `.xlsx` workbook that users can inspect and carefully revise.

## Stage 6: Branch Into Analysis Workflows

After transcript tabularization, projects may branch into:

- Complete Utterances coding;
- Word Counting;
- Target Vocabulary Coverage;
- POWERS coding;
- generic templates;
- speaking-time and rate workflows;
- transcript-table-informed Digital Conversational Turns.

Do not treat the table as ready for all branches until metadata, speaker labels, and utterance rows have been checked.

## Blinding Checkpoint

For manual coding workflows, consider whether configured identifiers should be encoded before coder-facing files are distributed. If the project uses blinding, decode back to original sample identifiers before DIAAD analysis when downstream joins require them. Re-encoding final exports can be useful for blinded statistical workflows.

This is an ideal pattern, not a universal requirement. In some dialogic workflows, sample identity may remain practically obvious even if formal identifiers are masked.

## Read Next

- Transcripts command pages: `docs/manual/04_modules/01_transcripts/05_commands/`
- Reliability functionality: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/02_usage_guide.md`
- Metadata extraction: `docs/manual/05_functionalities/08_metadata_extraction/02_usage_guide.md`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
