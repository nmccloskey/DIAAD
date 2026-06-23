# `vocab file` Usage Guide

Use `diaad vocab file` when a project needs to author a custom target-vocabulary resource.

## What The Template Contains

The template follows the resource shape used by bundled target-vocabulary resources:

```text
resource_id
display_name
language
task_type
base_forms
variant_map
norms
```

`base_forms` is the target lexicon. `variant_map` lets a resource declare accepted surface forms for a base form, such as plural forms, inflections, or task-specific accepted variants.

The `norms` block is optional in a finished custom resource. If included, it must use the structure expected by DIAAD's resource validator.

## Built-In Resources Do Not Need This Command

DIAAD includes five bundled CoreLex-style narrative resources:

```text
BrokenWindow
CatRescue
Cinderella
RefusedUmbrella
Sandwich
```

Run `vocab file` only when the project needs a new custom resource or wants a starting point for overriding a bundled resource.

## After Editing

After editing the template:

1. Save it as a project resource JSON file.
2. Set `advanced.target_vocabulary_resource_path` to the file or resource directory.
3. Run `diaad vocab check`.
4. Run `diaad vocab analyze` only after the resource check succeeds.

## Common Problems

If a resource uses a custom `resource_id`, the input data must contain that same value in the configured stimulus or narrative column.

If a resource reuses a bundled resource ID, the custom resource overrides the bundled definition for that run.

If a resource declares norms, the norm specification must be structurally valid, but structural validity does not establish that the norms are appropriate for the project.
