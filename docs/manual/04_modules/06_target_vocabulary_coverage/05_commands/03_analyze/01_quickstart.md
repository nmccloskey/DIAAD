# `vocab analyze` Quickstart

`diaad vocab analyze` computes target vocabulary coverage from utterance-level language data.

## Run

```bash
diaad vocab analyze --config config
```

## Minimum Inputs

The preferred input is an unblinded utterance-level workbook matching:

```text
unblind_utterance_data*.xlsx
```

If that file is not available, DIAAD falls back to the configured transcript table workbook, usually:

```text
transcript_tables.xlsx
```

Current CLI dispatch still checks that transcript tables are available, or can be auto-generated, before `vocab analyze` runs. The analysis function then prefers `unblind_utterance_data*.xlsx` if that file is available.

For custom resources, configure `advanced.target_vocabulary_resource_path`.

## Primary Output

By default, the command writes:

```text
target_vocab/
  target_vocab_data_YYMMDD_HHMM.xlsx
```

The output workbook contains:

```text
summary
details
```

## Immediate Next Step

Inspect the `summary` sheet for expected resource IDs in `narrative`, nonmissing `speaking_time`, and plausible `lexicon_coverage` values. Then inspect `details` for base forms that were unexpectedly missed or overmatched.

## Read Next

- `vocab analyze` research context: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/03_analyze/03_research_context.md`
- Target Vocabulary Coverage research context: `docs/manual/04_modules/06_target_vocabulary_coverage/03_research_context.md`
- Exact file-name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
