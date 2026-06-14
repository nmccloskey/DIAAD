# Generated Example I/O View

## Purpose

DIAAD has two complementary documentation sources:

- authored manual pages under `docs/manual/`
- generated Example I/O pages under `src/diaad/examples/assets/rendered_docs/example_io/`

The generated pages are a manual view named `example_io`. They show runnable
synthetic inputs, representative outputs, and output previews for individual
DIAAD commands. They are generated from packaged example specs and example
artifacts, so they should not be edited as ordinary authored manual prose.

## Composition Contract

Generated Example I/O pages include YAML front matter so a manual composition
layer can place them beside authored pages for the same object. Command pages
use stable command IDs such as `transcripts.tabularize`, `cus.files`, and
`vocab.check`.

Each command Example I/O page declares:

- `object_type: command`
- `object_id` and `command_id` matching the stable command ID
- `view: example_io`
- `view_label: Example I/O`
- `slot: examples`
- `source_manual: generated_example_io`
- `generated: true`

The full example dataset overview is different from a normal command page. It
represents the omnibus `diaad examples` command and the full example workflow,
so it declares `object_type: workflow`, `workflow_id: full_example_dataset`,
`command_id: examples`, and `command_subtype: omnibus`.

## Intended Manual Shape

A composed manual can keep source files separate while presenting a unified
command view. For example:

```text
Transcripts
  Tabularize
    Quickstart
    Usage Guide
    Implementation Notes
    Example I/O
```

In that layout, authored pages stay concise and conceptual, while the generated
`example_io` view can remain preview-driven and reproducible.

## Maintenance Rule

When example content changes, update the example specs, generator, renderer, or
tests, then regenerate the Example I/O pages. Do not duplicate generated
example tables or output previews into authored manual pages.
