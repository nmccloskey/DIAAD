# `vocab file` Quickstart

`diaad vocab file` creates a blank JSON template for a custom target-vocabulary resource.

## Run

```bash
diaad vocab file --config config
```

## Minimum Inputs

No analysis input file is required. The command uses the configured output directory.

## Primary Output

By default, the command writes:

```text
target_vocab/
  target_vocabulary_resource_template.json
```

## Immediate Next Step

Edit the JSON template into a real resource, then configure `advanced.target_vocabulary_resource_path` to point to that JSON file or to a directory of resource JSON files.

## Read Next

- Target Vocabulary Coverage research context: `docs/manual/04_modules/06_target_vocabulary_coverage/03_research_context.md`
- `vocab check` quickstart: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/02_check/01_quickstart.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
