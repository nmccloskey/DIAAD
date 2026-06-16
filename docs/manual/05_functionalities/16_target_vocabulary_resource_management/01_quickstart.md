# Target Vocabulary Resource Management Quickstart

Target Vocabulary Coverage depends on resources. A resource defines the target words for a stimulus or task, accepted variants for those words, and optional norm-table metadata.

Use the built-in resources when your data use the bundled CoreLex-style narrative IDs:

```text
BrokenWindow
CatRescue
Cinderella
RefusedUmbrella
Sandwich
```

Use a custom resource when the project has its own prompt, task vocabulary, accepted variants, or local scoring rules.

## Basic Path

1. Generate a starter template if needed:

```bash
diaad vocab file
```

2. Edit the JSON resource so its `id` matches the stimulus or narrative value in the analysis input.

3. Point DIAAD to the custom resource file or directory:

```yaml
advanced:
  target_vocabulary_resource_path: path/to/target_vocab_resources
```

4. Check the active resource set:

```bash
diaad vocab check
```

5. Run analysis only after the resource check succeeds:

```bash
diaad vocab analyze
```

## What The Resource Controls

The resource controls which words can count as target-vocabulary matches. It does not control transcript tabularization, speaker exclusion, speaking-time entry, or whether a custom lexicon is valid for a research question.

The most important fields are:

```text
id
display_name
language
task_type
base_forms
variant_map
norms
```

`base_forms` lists the target vocabulary. `variant_map` lists additional accepted surface forms for specific base forms.

## Read Next

- Target Vocabulary Coverage module: `docs/manual/04_modules/06_target_vocabulary_coverage/`
- `vocab file`: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/01_file/`
- `vocab check`: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/02_check/`
- `vocab analyze`: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/03_analyze/`
- Configuration: `docs/manual/02_operation/04_configuration.md`
