# Clinician-Client Dialogue POWERS Usage Guide

The clinician-client POWERS workflow is a dialogic transcript workflow. It uses the shared transcription baseline as its starting point, then branches into POWERS coding, reliability, analysis, and rate calculation.

Although this page uses clinician-client dialogue as the central example, the same workflow may fit other dialogic samples when the POWERS protocol is appropriate.

## Stage 1: Review The Transcript Table

After transcript tabularization, inspect:

- `sample_id` and any configured metadata fields;
- `utterance_id`;
- `speaker`;
- utterance order;
- transcript text;
- excluded-speaker settings;
- any revisions that may alter utterance identity or row order.

POWERS depends on utterance-level rows and speaker context. If transcript tables are revised after coding files are generated, decide whether affected POWERS workbooks must be regenerated or re-coded.

## Stage 2: Decide Where The Workflow Should Run

Dialogic samples can contain identifying content even after filenames or sample identifiers are encoded. A hosted web workflow may be convenient for data that are already de-identified, and the web service is designed around temporary session processing. Still, local execution can be preferable when users want more control over sensitive transcript content.

Three common options are:

- hosted web app for easily de-identifiable materials and itinerant use;
- local web app for a convenient interface with local files;
- local CLI for automated, scripted, or privacy-sensitive workflows.

Formal blinding may still be useful, but it may not be practically effective when coders recognize a client, clinician, session, or conversational content.

## Stage 3: Generate POWERS Workbooks

Run:

```bash
diaad powers files
```

The primary workbook contains:

```text
utterance_coding
section_e
```

The reliability workbook contains the selected reliability material. Selection is sample-based: if a sample is selected, all of its utterance rows are included.

When `project.automate_powers` is true and NLP support is available, DIAAD fills first-pass values for:

```text
speech_units
filled_pauses
content_words
num_nouns
tagged_utterance
```

The `tagged_utterance` column is a review aid, not a POWERS score.

## Stage 4: Complete Human Coding

Human review is required before analysis. Coders should inspect automated fields, correct them where needed, and complete fields that DIAAD does not automate, including:

```text
turn_type
circumlocutions
sem_paras
phon_errs
neologisms
comments
lg_pauses
collab_repair
POWERS_comment
```

The Section E sheet contains sample-level note or descriptor fields:

```text
type_of_day
amount_of_enjoyment
degree_of_difficulty
other_notes
```

In the current implementation, Section E is not summarized by `powers analyze`, evaluated by `powers evaluate`, or converted to rates by `powers rates`.

## Stage 5: Evaluate Reliability

After primary and reliability workbooks are complete, run:

```bash
diaad powers evaluate
```

The command evaluates continuous or count-like metrics such as speech units, content words, nouns, pauses, and error counts, plus categorical fields such as `turn_type` and `collab_repair`.

If another reliability round is needed, run:

```bash
diaad powers reselect
```

Reselection clears manual POWERS fields for newly selected rows and can reapply first-pass automation when automation is enabled. Review the new workbook before distributing it.

## Stage 6: Analyze And Rate Completed Coding

After coding and reliability review are complete, run:

```bash
diaad powers analyze
```

The analysis workbook can include utterance-, turn-, speaker-, and dialog-level summaries.

Then create a speaking-time workbook:

```bash
diaad templates times
```

Enter speaking time in seconds and run:

```bash
diaad powers rates
```

POWERS rates are inferred for count-like dialog-summary columns. Proportions, ratios, identifiers, and Section E fields are not rate-normalized.

## Common Problems

If automated columns are blank, check `project.automate_powers`, the installed NLP dependencies, and `advanced.spacy_model_name`.

If paired reliability rows are fewer than expected, check that sample and utterance identifiers match across primary and reliability workbooks.

If Section E values do not appear in analysis, reliability, or rates, that is expected in the current implementation.

## Read Next

- `powers files` usage guide: `docs/manual/04_modules/05_powers/05_commands/01_files/02_usage_guide.md`
- `powers evaluate` usage guide: `docs/manual/04_modules/05_powers/05_commands/04_evaluate/02_usage_guide.md`
- `powers analyze` usage guide: `docs/manual/04_modules/05_powers/05_commands/02_analyze/02_usage_guide.md`
- `powers rates` usage guide: `docs/manual/04_modules/05_powers/05_commands/03_rates/02_usage_guide.md`
