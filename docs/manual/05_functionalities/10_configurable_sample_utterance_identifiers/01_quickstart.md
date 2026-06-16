# Configurable Sample and Utterance Identifiers Quickstart

DIAAD uses sample and utterance identifiers to keep transcript tables, coding files, reliability subsets, blinding codebooks, and analysis outputs joinable.

## Defaults

Use the defaults unless you have a durable integration reason to change them:

```yaml
advanced:
  sample_id_column: sample_id
  utterance_id_column: utterance_id
```

The generated transcript table uses:

- `sample_id` for transcript/sample identifiers;
- `utterance_id` for utterance identifiers within each sample.

For utterance-level joins, treat the sample identifier and utterance identifier together as the safest coordinate.

## When To Change Them

Change identifier column names only when:

- your project already has stable external identifier names;
- you are integrating multiple DIAAD datasets that need expanded identifier fields;
- a downstream system requires different column names.

Make the change before tabularization and before creating manual coding files.

## Read Next

- Configuration: `docs/manual/02_operation/04_configuration.md`
- Transcript tabularization feature: `docs/manual/03_features/01_transcript_tabularization.md`
- Revision handling: `docs/manual/05_functionalities/11_revision_handling/01_quickstart.md`
