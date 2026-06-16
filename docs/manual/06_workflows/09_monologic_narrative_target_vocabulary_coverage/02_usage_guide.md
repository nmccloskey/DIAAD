# Monologic Narrative Target Vocabulary Coverage Usage Guide

Target Vocabulary Coverage, or TVC, scores samples against an active target-vocabulary resource. In monologic narrative workflows, this may mean using bundled CoreLex-style resources or project-specific custom lexicons.

## Decide Which Resource Set Applies

Use built-in resources when the input stimulus values match bundled IDs:

```text
BrokenWindow
CatRescue
Cinderella
RefusedUmbrella
Sandwich
```

Use custom resources when the project has its own task, prompt, intervention vocabulary, accepted variants, or scoring rules.

For custom resources:

```bash
diaad vocab file
```

Then edit the JSON and set:

```yaml
advanced:
  target_vocabulary_resource_path: path/to/resource_or_directory
```

## Check Before Analysis

Run:

```bash
diaad vocab check
```

The check confirms resource structure and lists active resource IDs. It does not validate that a custom resource is scientifically appropriate.

## Prepare Input

`vocab analyze` requires:

- the configured sample identifier;
- utterance text;
- a stimulus or narrative column matching active resource IDs;
- speaking-time values when using transcript-table fallback.

When available, DIAAD prefers an unblinded utterance-level input workbook matching:

```text
unblind_utterance_data*.xlsx
```

That path can be useful in integrated workflows where earlier steps have already produced utterance-level data with sample identifiers, stimulus labels, utterances, speaking time, and inclusion filters.

If no unblinded utterance data file is found, DIAAD falls back to the configured transcript table.

## Run Analysis

Run:

```bash
diaad vocab analyze
```

The output workbook contains:

```text
summary
details
```

Inspect `details` to see which base forms were matched. If coverage is unexpectedly low, compare observed forms with the resource `variant_map`.

## Add Rates

`vocab analyze` writes `core_tokens_per_min` when speaking time is available. To add additional inferred per-minute fields for count-like TVC summary columns, run:

```bash
diaad vocab rates
```

## Norms And Percentiles

Built-in CoreLex-style resources declare online norm tables. DIAAD retrieves those norm tables and computes percentiles locally; transcript data are not uploaded to the norm source.

Percentiles should be interpreted only when the norm source, task, population, and metric definition match the project.

## Read Next

- `vocab check` usage guide: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/02_check/02_usage_guide.md`
- `vocab analyze` usage guide: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/03_analyze/02_usage_guide.md`
- TVC rates: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/04_rates/02_usage_guide.md`
- Resource management: `docs/manual/05_functionalities/16_target_vocabulary_resource_management/02_usage_guide.md`
