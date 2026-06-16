# Target Vocabulary Resource Management Usage Guide

Target-vocabulary resources are the bridge between an elicitation task and DIAAD's coverage metrics. The resource tells DIAAD which target words exist for a stimulus, which variants should count for those words, and whether percentile lookups are available.

## Built-In And Custom Resources

With no custom resource path configured, DIAAD loads the bundled resource set:

```text
BrokenWindow
CatRescue
Cinderella
RefusedUmbrella
Sandwich
```

When `advanced.target_vocabulary_resource_path` is configured, DIAAD loads the built-ins first and then merges the custom resources. A custom resource with a new `id` adds another available stimulus. A custom resource with the same `id` as a built-in resource overrides that built-in definition for the run.

The configured path can point to one JSON file or to a directory of JSON files.

## Resource Fields

Each resource requires:

| Field | Purpose |
|---|---|
| `id` | Machine-readable resource identifier. Input stimulus or narrative values must match this value. |
| `display_name` | Human-readable label for the task or resource. |
| `language` | Resource language label. |
| `task_type` | Task or elicitation type label. |
| `base_forms` | Unique target vocabulary items. |
| `variant_map` | Accepted variants keyed by base form. |

The optional `norms` block declares online CSV norm tables for percentile calculations. Norm specifications are checked for shape, but their scientific appropriateness remains a project-level decision.

## Variants

Each base form counts as an accepted form for itself. `variant_map` adds accepted surface forms such as plurals, inflections, spelling variants, or task-specific alternatives.

Use variants carefully. A variant can map to only one base form, and it cannot also be listed as a separate base form. If two target words can plausibly share the same observed token, the coding protocol needs to decide which interpretation is allowed before analysis.

## Input Matching

`vocab analyze` filters input rows by the active resource IDs. In practice, this means the configured stimulus or narrative column must use the same identifiers as the resources.

For built-ins, those identifiers are:

```text
BrokenWindow
CatRescue
Cinderella
RefusedUmbrella
Sandwich
```

For custom resources, the identifier is whatever the project sets in the resource's `id` field.

## Checking Resources

Run:

```bash
diaad vocab check
```

The check report lists the configured custom path, active resource count, and active IDs. It also catches structural errors such as missing required fields, duplicate base forms, variants that point to nonexistent base forms, variants mapped to multiple base forms, and malformed norm specifications.

`vocab check` is structural validation. It does not prove that a custom target vocabulary is clinically, psychometrically, or theoretically valid.

## Read Next

- Word Counting Versus Target Vocabulary Coverage: `docs/manual/03_features/02_word_counting_vs_target_vocabulary_coverage.md`
- Target Vocabulary Coverage research context: `docs/manual/04_modules/06_target_vocabulary_coverage/03_research_context.md`
- `vocab check` usage guide: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/02_check/02_usage_guide.md`
- Configured filenames and file discovery: `docs/manual/05_functionalities/09_configured_filenames_file_discovery_input_selection/02_usage_guide.md`
