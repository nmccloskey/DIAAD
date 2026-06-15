# `vocab check` Quickstart

`diaad vocab check` validates the active target-vocabulary resources and writes a compact resource report.

## Run

```bash
diaad vocab check --config config
```

## Minimum Inputs

No custom input is required when using only bundled resources.

For custom resources, configure:

```yaml
target_vocabulary_resource_path: diaad_data/input/target_vocab/resources
```

The configured path can point to one JSON file or to a directory of JSON files.

## Primary Output

By default, the command writes:

```text
target_vocab/
  target_vocab_resource_check.txt
```

## Immediate Next Step

Review the active resource IDs. Confirm that every expected built-in or custom resource is present before running `vocab analyze`.

## Read Next

- `vocab file` usage guide: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/01_file/02_usage_guide.md`
- `vocab analyze` quickstart: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/03_analyze/01_quickstart.md`
- Target Vocabulary Coverage research context: `docs/manual/04_modules/06_target_vocabulary_coverage/03_research_context.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
