# `vocab check` Usage Guide

Use `diaad vocab check` before analysis when a project depends on custom target-vocabulary resources, or when you want to confirm which built-in resources are active.

## Active Resource Set

With no custom resource path configured, the active set is the bundled resource set:

```text
BrokenWindow
CatRescue
Cinderella
RefusedUmbrella
Sandwich
```

When `advanced.target_vocabulary_resource_path` is configured, DIAAD loads the bundled resources and the custom resources. Custom resources add to the active set unless they reuse a bundled resource ID. If a custom resource uses the same ID as a bundled resource, the custom resource overrides the bundled one for that run.

## What Is Validated

The checker validates resource structure and consistency:

- required top-level fields are present;
- string fields are non-empty;
- `base_forms` is non-empty and has no duplicates;
- every `variant_map` key is a base form;
- variants are non-empty strings;
- a variant is not mapped to multiple base forms;
- declared norms have the expected URL, format, and column mapping fields.

## What Is Not Validated

The checker does not establish that a resource is psychometrically valid. It also does not prove that a declared norm source is appropriate for a project, population, or task.

The command validates norm specification shape. Norm data are loaded during `vocab analyze` when relevant resources are present in the input data.

## Report Interpretation

The report lists:

```text
Custom resource path
Custom resource id or count
Active resource count
Active resource ids
```

Use this output to catch misspelled IDs, wrong paths, duplicate resource definitions, and unexpected overrides before running analysis.

## Common Problems

If the custom resource path is missing or points to a non-JSON file, fix `advanced.target_vocabulary_resource_path`.

If a variant appears under two base forms, edit the resource so each accepted variant maps to only one base form.

If `vocab analyze` later produces no rows, compare the active resource IDs with the stimulus or narrative values in the input data.
