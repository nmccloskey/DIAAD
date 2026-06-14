# Examples Module Usage Guide

The Examples module has two user-facing package modes: full-dataset examples and command-specific examples. It also has a maintainer-facing documentation regeneration mode.

## Full Dataset Examples

`diaad examples` creates a full synthetic example dataset. This package is best when you want to browse DIAAD's overall file organization and see how commands relate across modules.

The full package includes input fixtures and expected outputs for many commands. It is a teaching fixture rather than a template that every user project must copy exactly.

## Command-Specific Examples

Use `--for-command` when you want a smaller package for one command or a small set of commands:

```bash
diaad examples --for-command "cus analyze" --config config
```

Command-specific packages are meant to show the minimal configuration, input, output, and logs needed to understand that command. They use runtime-shaped output paths so the examples look like normal DIAAD runs.

## Web App Downloads

The web app does not expose `examples` as an ordinary selectable dispatch command. Instead, it provides a dedicated example-file download action. If commands are selected in the web command menu, the example download can create selected-command examples; otherwise it can create a full example dataset.

## Force And Regeneration

Use `--force` when an existing example package should be replaced. Use it cautiously if you have copied or edited example outputs for local notes.

`--render-docs` is for documentation maintenance. It regenerates packaged Example I/O markdown under `src/diaad/examples/assets/rendered_docs/example_io/`; it is not needed for ordinary use.

## Do Not Edit Generated Views By Hand

Generated Example I/O pages are maintained through specs, generator code, renderer code, and tests. Edit those sources and regenerate rather than copying generated tables into authored manual pages.
