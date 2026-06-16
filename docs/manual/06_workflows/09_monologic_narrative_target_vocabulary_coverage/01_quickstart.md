# Monologic Narrative Target Vocabulary Coverage Quickstart

Use this workflow when a narrative or task should be scored against a predefined target lexicon. DIAAD's Target Vocabulary Coverage workflow includes built-in CoreLex-style narrative resources and supports custom resources.

## Built-In Resource Path

For the bundled CoreLex-style resources, start by checking the active resources:

```bash
diaad vocab check
```

Then run:

```bash
diaad vocab analyze
```

If additional per-minute fields are needed:

```bash
diaad vocab rates
```

## Custom Resource Path

Create a resource template:

```bash
diaad vocab file
```

Edit the JSON resource, configure `advanced.target_vocabulary_resource_path`, then run:

```bash
diaad vocab check
diaad vocab analyze
```

## Key Requirements

The input must include a sample identifier and a stimulus or narrative value that matches an active resource ID.

Built-in IDs include:

```text
BrokenWindow
CatRescue
Cinderella
RefusedUmbrella
Sandwich
```

## Read Next

- Target Vocabulary Coverage module: `docs/manual/04_modules/06_target_vocabulary_coverage/01_quickstart.md`
- Target vocabulary resource management: `docs/manual/05_functionalities/16_target_vocabulary_resource_management/01_quickstart.md`
- `vocab analyze`: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/03_analyze/01_quickstart.md`
- Word Counting versus Target Vocabulary Coverage: `docs/manual/03_features/02_word_counting_vs_target_vocabulary_coverage.md`
