# Generated Example I/O

DIAAD has two complementary documentation sources:

- authored manual pages under `docs/manual/`
- generated Example I/O pages under `src/diaad/examples/assets/rendered_docs/example_io/`

The generated pages are a manual view named `example_io`. They show runnable synthetic inputs, representative outputs, and preview tables or snippets for DIAAD commands. They are generated from packaged example specs and example artifacts, so they should not be edited as ordinary authored prose.

## What This View Is For

Authored manual pages should explain concepts, decisions, workflows, command usage, and implementation notes. Generated Example I/O pages should show what a command looks like when it runs on a small synthetic project.

This separation keeps the manual readable while making examples reproducible. If an example output changes, the source of truth is the example spec, generator, renderer, and tests, not a copied table inside an authored page.

## Composition Contract

Generated Example I/O pages include YAML front matter so a manual composition layer can place them beside authored views for the same object.

Command pages declare fields such as:

- `object_type: command`
- `object_types: [command]`
- `object_id` and `command_id`, using stable IDs such as `transcripts.tabularize`, `cus.files`, and `vocab.check`
- `canonical_command`, such as `transcripts tabularize`
- `module_id`
- `view: example_io`
- `view_label: Example I/O`
- `view_order: 50`
- `slot: examples`
- `source_manual: generated_example_io`
- `generated: true`

The full example dataset overview is different from a normal command page. It represents the omnibus `diaad examples` command and the full example workflow, so it declares workflow identity such as `object_type: workflow`, `object_id: full_example_dataset`, and `workflow_id: full_example_dataset`, while also carrying secondary command identity for `examples`.

## Intended Manual Shape

A composed manual can keep source files separate while presenting a unified command view. For example:

```text
Transcripts
  Tabularize
    Quickstart
    Usage Guide
    Implementation Notes
    Example I/O
```

In that layout, authored pages stay concise and conceptual, while the generated `example_io` view remains preview-driven and reproducible.

## User Examples and Maintainer Regeneration

Ordinary users can generate example files with `diaad examples` or command-specific example packages with `diaad examples --for-command "<command>"`. The web app exposes related example downloads through its example-file action.

Documentation maintainers can regenerate the packaged Example I/O markdown with the examples renderer. Details for `--render-docs` belong in the later Examples module or command implementation notes. This feature page only establishes the composition rule: generated pages are maintained by generation, not by manual editing.

## Maintenance Rule

When example content changes, update the example specs, generator, renderer, or tests, then regenerate the Example I/O pages. Do not duplicate generated example tables, workbook previews, manifests, or generated README content inside authored manual pages.

## Related Manual Pages

- [Functional overview](../01_overview/03_functional_overview.md)
- [Command-line operation](../02_operation/02_command_line.md)
- [Web app operation](../02_operation/03_webapp.md)
- [Testing](../02_operation/05_testing.md)

## Draft Review Notes

Before publication, confirm whether authored manual pages will receive comparable front matter. This page currently treats generated `example_io` metadata as a composition contract for generated documentation only.
