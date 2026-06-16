# Monologic Narrative Word Counting Usage Guide

Word Counting is a human-reviewed workflow. DIAAD can produce automated first-pass counts, but users should review counts against the project's word-counting protocol before analysis.

## Choose The Input Path

If Complete Utterance analysis has already run, `words files` can use:

```text
cu_coding_by_utterance.xlsx
```

This is often preferable because CU-derived information can help mark non-countable rows as `NA`.

If CU output is not available, `words files` falls back to the transcript table:

```text
transcript_tables.xlsx
```

This direct path is useful when a project needs word counts without CU coding.

## Generate Workbooks

Run:

```bash
diaad words files
```

Important settings include:

| Setting | Use |
|---|---|
| `project.reliability_fraction` | Sample fraction for reliability workbook. |
| `project.num_coders` | Coder assignment. |
| `project.exclude_speakers` | Speaker labels marked not countable. |
| `advanced.word_count_filename` | Primary workbook filename. |
| `advanced.auto_blind` | Optional blinding for coder-facing files. |

If the project uses blinding, encode before coder-facing distribution and decode before analysis when original identifiers are needed for joins.

## Review First-Pass Counts

DIAAD seeds the `word_count` column with automated first-pass counts. Review these values manually.

Protocol-specific review may need to consider:

- repetitions and part-word repetitions;
- nonword fillers;
- neologisms;
- unintelligible material;
- off-task commentary;
- direct prompt-only responses;
- utterance boundaries and speaker labels.

The point of the first pass is efficiency, not final authority.

## Evaluate Reliability

After primary and reliability word counts are complete, run:

```bash
diaad words evaluate
```

Inspect detailed agreement outputs before deciding whether counts need adjudication, recoding, or reselection.

Use:

```bash
diaad words reselect
```

only when a replacement or additional reliability round is needed.

## Analyze And Rate

Run:

```bash
diaad words analyze
```

The analysis writes utterance-level cleaned data and sample-level summaries with totals and per-utterance descriptive statistics.

For words per minute, create or complete a speaking-time workbook:

```bash
diaad templates times
diaad words rates
```

Speaking time is entered in seconds and converted to minutes internally.

## Relationship To Target Vocabulary Coverage

Word Counting and Target Vocabulary Coverage may both report token-like values, but they are not interchangeable.

Word Counting asks how much countable language was produced. TVC asks whether words from a predefined target vocabulary were produced and how completely that lexicon was covered.

## Read Next

- `words files` usage guide: `docs/manual/04_modules/04_word_counting/05_commands/01_files/02_usage_guide.md`
- `words analyze` usage guide: `docs/manual/04_modules/04_word_counting/05_commands/04_analyze/02_usage_guide.md`
- Reliability functionality: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/02_usage_guide.md`
- Target Vocabulary workflow: `docs/manual/06_workflows/09_monologic_narrative_target_vocabulary_coverage/01_quickstart.md`
